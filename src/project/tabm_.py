"""Train and evaluate an MLP or TabM (an efficient ensemble of MLPs).

This is a focused variant of `project.mlp` that additionally supports TabM
(https://github.com/yandex-research/tabm). Numerical feature embeddings are
intentionally *not* supported here. Muon (via the aux-Adam wrappers) is
supported for both the plain MLP and the TabM backbones.
"""

import dataclasses
import datetime
import logging
import math
import time
import typing
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

import numpy as np
import scipy.special
import tabm
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

import lib.data
import lib.env
import lib.experiment
import lib.metrics
import lib.optim.utils
import lib.preprocessing
import lib.utils.collections
import lib.utils.dataclasses
import lib.utils.misc
import lib.utils.time
import lib.utils.torch
import project.ranking
from lib.types import DataKey, KWArgs, PartKey

_LOGGER = logging.getLogger(__name__)

_WORST_SCORE = -999999.0

# TabM architecture types that pack the ensemble into the very first layer
# (and thus require `start_scaling_init=None`).
_PACKED_ARCH_TYPES = ('tabm-packed',)


class _OneHotEncoding(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                nn.functional.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


class _Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: int | None,
        type: Literal['mlp', 'tabm'],
        backbone: KWArgs,
    ) -> None:
        assert n_num_features > 0 or cat_cardinalities
        super().__init__()

        self.cat_module = (
            _OneHotEncoding(cat_cardinalities) if cat_cardinalities else None
        )

        # The per-feature chunks of the input. They are used by TabM to
        # initialize the very first scaling (one scaling value per input feature,
        # shared within a feature's one-hot block).
        d_features: list[int] = [1] * n_num_features + list(cat_cardinalities)
        d_in = n_num_features + sum(cat_cardinalities)
        d_out = 1 if n_classes is None or n_classes == 2 else n_classes

        self._type = type
        if type == 'mlp':
            self.ensemble_view = None
            self.backbone = tabm.MLPBackbone(d_in=d_in, **backbone)
            self.output = nn.Linear(backbone['d_block'], d_out)

        elif type == 'tabm':
            arch_type = backbone.get('arch_type', 'tabm')
            is_packed = arch_type in _PACKED_ARCH_TYPES
            self.ensemble_view = tabm.EnsembleView(k=backbone['k'])
            self.backbone = tabm.make_tabm_backbone(
                d_in=d_in,
                **backbone,
                start_scaling_init=None if is_packed else 'random-signs',
                start_scaling_init_chunks=None if is_packed else d_features,
            )
            self.output = tabm.LinearEnsemble(
                self.backbone.get_original_output_shape()[0],
                d_out,
                k=self.backbone.k,
            )

        else:
            raise ValueError(f'Unknown model {type=}')

        self._n_num_features = n_num_features

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x_list: list[Tensor] = []

        if x_num is None:
            assert self._n_num_features == 0
        else:
            assert self._n_num_features > 0
            x_list.append(x_num)

        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x_list.append(self.cat_module(x_cat).to(torch.get_default_dtype()))

        x = torch.cat(x_list, dim=-1)

        if self.ensemble_view is not None:
            # (B, D) -> (B, K, D), or (B, K, D) -> (B, K, D) (different batches
            # per ensemble member, see `share_training_batches`).
            x = self.ensemble_view(x)
        x = self.backbone(x)
        x = self.output(x)
        return x


class _ApplyModel(Protocol):
    def __call__(
        self,
        model: nn.Module,
        data: dict[DataKey, dict[PartKey, Tensor]],
        *,
        part: PartKey,
        idx: Tensor,
    ) -> Tensor: ...


def _apply_model_impl(
    model: nn.Module,
    data: dict[DataKey, dict[PartKey, Tensor]],
    *,
    part: PartKey,
    idx: Tensor,
) -> Tensor:
    device = next(model.parameters()).device
    x_storage = data['x_num'] if 'x_num' in data else None
    c_storage = data['x_cat'] if 'x_cat' in data else None
    if x_storage is not None and x_storage[part].device != device:
        idx = idx.to(x_storage[part].device)
    x_num = x_storage[part][idx] if x_storage is not None else None
    x_cat = c_storage[part][idx] if c_storage is not None else None
    if x_num is not None:
        if x_num.device != device:
            x_num = x_num.to(device, non_blocking=True)
        # `x_num` may be stored in a reduced precision (see `Config.num_dtype`).
        # Upcast the (small) batch back to the default dtype so the forward pass
        # and the one-hot concatenation operate in a single dtype; under AMP the
        # matmuls are downcast again, so compute throughput is preserved.
        default_dtype = torch.get_default_dtype()
        if x_num.dtype != default_dtype:
            x_num = x_num.to(default_dtype)
    if x_cat is not None and x_cat.device != device:
        x_cat = x_cat.to(device, non_blocking=True)
    return (
        model(x_num, x_cat)
        .squeeze(-1)  # Remove the last dimension for regression predictions.
        .float()
    )


class _ComputeMetrics(Protocol):
    def __call__(
        self, *, y_true: np.ndarray, y_pred: np.ndarray
    ) -> lib.metrics.MetricDict: ...


def _are_valid_predictions(y_pred: np.ndarray) -> bool:
    return np.isfinite(y_pred).all().item()


def _ensemble_reduce(y_pred: np.ndarray, task_type: lib.data.TaskType) -> np.ndarray:
    """Average the `k` per-member predictions of a TabM ensemble.

    The input has the per-member dimension at axis 1:
    - regression:   `(N, k)`        -> `(N,)`     (averaged labels)
    - binclass:     `(N, k)` logits -> `(N,)`     (averaged probabilities)
    - multiclass:   `(N, k, C)` logits -> `(N, C)` (averaged probabilities)

    For classification the averaging is done over probabilities (not logits),
    which matches the official TabM evaluation.
    """
    if task_type.is_regression:
        return y_pred.mean(axis=1)
    elif task_type.is_binclass:
        return scipy.special.expit(y_pred).mean(axis=1)
    else:
        return scipy.special.softmax(y_pred, axis=-1).mean(axis=1)


class _EvaluateImplOutput(NamedTuple):
    metrics: dict[PartKey, Any]
    predictions: dict[PartKey, np.ndarray]


@lib.utils.torch.adjust_gpu_memory_usage('batch_size')
def _evaluate_impl(
    *,
    apply_model: _ApplyModel,
    model: nn.Module,
    data: dict[DataKey, dict[PartKey, Tensor]],
    task_labels: dict[PartKey, np.ndarray],
    task_type: lib.data.TaskType,
    task_score: lib.data.TaskScore,
    ranking_qids: dict[PartKey, np.ndarray] | None,
    compute_metrics: _ComputeMetrics,
    is_tabm: bool,
    parts: list[PartKey],
    prediction_transform: Callable[[np.ndarray], np.ndarray] | None,
    batch_size: int,
    device: torch.device,
) -> _EvaluateImplOutput:
    model.eval()

    predictions = {}
    metrics = {}
    for part in parts:
        y_pred = (
            torch.cat(
                [
                    apply_model(model, data, part=part, idx=idx)
                    for idx in torch.arange(
                        len(task_labels[part]), device=device
                    ).split(batch_size)
                ]
            )
            .cpu()
            .numpy()
        )
        # For TabM, reduce the per-member predictions to a single ensemble
        # prediction before (optionally) inverse-transforming and scoring.
        if is_tabm:
            y_pred = _ensemble_reduce(y_pred, task_type)
        if prediction_transform is not None:
            y_pred = prediction_transform(y_pred)
        predictions[part] = y_pred

        if _are_valid_predictions(y_pred):
            metrics[part] = compute_metrics(y_true=task_labels[part], y_pred=y_pred)
            if ranking_qids is not None:
                metrics[part].update(
                    project.ranking.compute_metrics(
                        y_true=task_labels[part],
                        y_score=y_pred,
                        qid=ranking_qids[part],
                    )
                )
            metrics[part]['score'] = (
                1 if task_score.higher_is_better else -1
            ) * metrics[part][task_score.value]
        else:
            metrics[part] = {'score': _WORST_SCORE}

    return _EvaluateImplOutput(metrics, predictions)


def _load_task(
    dataset_dir: str | Path, split_id: lib.data.SplitIDLike, **split_options
) -> tuple[
    dict[PartKey, np.ndarray],
    lib.data.TaskType,
    lib.data.TaskScore,
    dict[PartKey, np.ndarray] | None,
]:
    task_labels = lib.data.load_data(
        dataset_dir, split_id, keys=['y'], **split_options
    )['y']

    dataset_info = lib.data.load_info(dataset_dir)
    task_type = lib.data.TaskType(dataset_info['task']['type'])
    task_score = lib.data.TaskScore(dataset_info['task']['score'])
    ranking_qids = project.ranking.load_qids(dataset_dir, split_id, **split_options)

    return task_labels, task_type, task_score, ranking_qids


def _make_optimizer(
    model: _Model, *, type: str, muon: dict[str, Any] | None = None, **kwargs
) -> torch.optim.Optimizer:
    optimizer_cls = lib.utils.misc.import_(type)

    parameter_groups = lib.optim.utils.ParameterGroups()
    if muon is None:
        # AdamW-style: a single optimizer class that takes ``**kwargs`` (lr,
        # weight_decay, ...) as global defaults; per-group keys are optional.
        nonmuon_hparams = {}
        catchall_hparams = {}
        optimizer_kwargs = kwargs
    else:
        # Aux-Adam Muon (e.g. ``vendor.muon.SingleDeviceMuonWithAuxAdam``) takes
        # only ``param_groups`` and reads every hyperparameter off each group, so
        # the Adam (non-muon) groups must carry lr/betas/eps/weight_decay
        # themselves and no kwargs may be passed to the constructor.
        nonmuon_hparams = {
            'use_muon': False,
            **{x: kwargs[x] for x in ['lr', 'betas', 'eps']},
        }
        catchall_hparams = {**nonmuon_hparams, 'weight_decay': kwargs['weight_decay']}
        optimizer_kwargs = {}
        muon_hparams = {
            **{x: kwargs[x] for x in ['lr', 'weight_decay']},
            **muon,
        }
        # Muon optimizes the 2D+ weight matrices of the backbone's linear layers.
        # Vectors (e.g. the output layer of a (k, d, 1) `LinearEnsemble`) are
        # excluded via the `1 not in shape` check and fall back to Adam.
        parameter_groups.add(
            [
                typing.cast(nn.Parameter, x.weight)
                for x in model.backbone.modules()
                if isinstance(
                    x, nn.Linear | tabm.LinearBatchEnsemble | tabm.LinearEnsemble
                )
                and 1 not in x.weight.shape[-2:]
            ],
            use_muon=True,
            **muon_hparams,
        )
    # Parameters that should not be weight-decayed: biases, normalization
    # parameters and the TabM scaling adapters (`r`/`s` of `LinearBatchEnsemble`,
    # `LinearEnsemble`-style packed ensembles share weights so have none).
    parameter_groups.add(
        (
            p
            for m in model.modules()
            for pn, p in m.named_parameters(recurse=False)
            if (
                isinstance(m, nn.BatchNorm1d | nn.LayerNorm | nn.InstanceNorm1d)
                or pn.endswith('bias')
                or pn in ('r', 's')
            )
        ),
        **nonmuon_hparams,
        weight_decay=0.0,
    )
    parameter_groups.add(model.parameters(), **catchall_hparams)

    return optimizer_cls(parameter_groups.unwrap(), **optimizer_kwargs)


def _make_loss_fn(task_type: lib.data.TaskType):
    return {
        lib.data.TaskType.BINCLASS: nn.functional.binary_cross_entropy_with_logits,
        lib.data.TaskType.MULTICLASS: nn.functional.cross_entropy,
        lib.data.TaskType.REGRESSION: nn.functional.mse_loss,
    }[task_type]


def _make_compute_metrics(
    task_type: lib.data.TaskType, *, prediction_type: lib.metrics.PredictionType
):
    return {
        lib.data.TaskType.BINCLASS: partial(
            lib.metrics.compute_binclass_metrics, prediction_type=prediction_type
        ),
        lib.data.TaskType.MULTICLASS: partial(
            lib.metrics.compute_multiclass_metrics, prediction_type=prediction_type
        ),
        lib.data.TaskType.REGRESSION: lib.metrics.compute_regression_metrics,
    }[task_type]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int
    data: KWArgs
    # The `model` sub-config must contain `type` ("mlp" or "tabm") and a
    # `backbone` sub-config. For `type="tabm"`, `backbone` must additionally
    # contain `k` (the ensemble size) and may contain `arch_type` (one of
    # "tabm", "tabm-mini", "tabm-packed"; default "tabm").
    model: KWArgs
    optimizer: KWArgs
    batch_size: int
    eval_batch_size: int | None = None
    # TabM-only: if True (default), the same batch is fed to all `k` ensemble
    # members. If False, each member receives an independently sampled batch
    # (see the official TabM notebook). Must be True for `type="mlp"`.
    share_training_batches: bool = True
    patience: int
    n_epochs: int | None = None
    gradient_clipping_norm: float | None = None
    # If one decides to use `amp_dtype`, "bfloat16" is a reasonable default value.
    amp_dtype: Literal['bfloat16', 'float16'] | None = None
    compile: bool = False
    save_checkpoint: bool = False
    # If True, the feature tensors (x_num, x_cat) are kept in pinned host memory
    # and transferred to GPU batch-wise inside `apply_model`, instead of being
    # fully materialized on the device. Useful for datasets that do not fit in
    # GPU memory (e.g. maps-routing-big).
    pin_memory: bool = False
    # Storage dtype for the numerical features (`x_num`). Casting to a lower
    # precision (e.g. "bfloat16") roughly halves the resident memory of the
    # numerical features. Each batch is upcast back to the default dtype before
    # the forward pass, so model weights and compute are unaffected.
    num_dtype: Literal['bfloat16', 'float16'] | None = None

    def get_eval_batch_size(self) -> int:
        return (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else 2048
            if self.compile
            else 32768
        )


@lib.utils.torch.set_num_threads(1)
def main(exp: str | Path, *, force: bool = False) -> lib.experiment.ReportDict | None:
    # >>> Start

    if not lib.experiment.try_start(exp, main, force=force):
        print(f'The experiment is already completed: {exp}')
        return None

    exp = lib.experiment.check(exp)
    config = lib.utils.dataclasses.from_dict(Config, lib.experiment.load_config(exp))
    report = lib.experiment.create_report(exp, main)
    print(f'{lib.experiment.format_config(config)}\n')

    is_tabm = config.model['type'] == 'tabm'
    if not is_tabm:
        assert config.share_training_batches, (
            'share_training_batches must be True for type="mlp"'
        )

    # >>> Setup

    lib.utils.torch.seed(config.seed)
    device = lib.utils.torch.get_device()
    _LOGGER.debug(f'Device: {device}')

    # >>> Data

    data = lib.data.build_data(**config.data)
    assert 'x_bin' not in data
    # The task labels are loaded separately from the rest of the data.
    # They are used for computing metrics, and thus must never be modified.
    task_labels, task_type, task_score, ranking_qids = _load_task(
        config.data['dataset_dir'],
        config.data.get('split_id', lib.data.DEFAULT_SPLIT_ID),
        **lib.data.split_options(config.data),
    )

    if task_type.is_regression:
        regression_label_transformer = lib.preprocessing.StandardScaler1d()
        data['y'] = lib.preprocessing.fit_transform_dict(
            regression_label_transformer, data['y']
        )
    else:
        regression_label_transformer = None

    data_device = device if not config.pin_memory else torch.device('cpu')
    num_storage_dtype: torch.dtype | None = (
        None if config.num_dtype is None else getattr(torch, config.num_dtype)
    )

    def _place(key: DataKey, array: Any) -> Tensor:
        t = torch.as_tensor(array)
        # Cast the numerical features to the reduced storage dtype *on the host*
        # before moving them to the device, so the device never materializes the
        # full-precision copy (the float32 materialization is what triggers OOM
        # on large datasets such as maps-routing-big).
        if key == 'x_num' and num_storage_dtype is not None:
            t = t.to(num_storage_dtype)
        return t.to(data_device)

    data = {
        key: {part: _place(key, array) for part, array in data[key].items()}
        for key in data.keys()
    }
    if config.pin_memory:
        for key in data:
            for part in data[key]:
                data[key][part] = data[key][part].pin_memory()
    Y_train = (
        data['y']['train']
        .to(torch.long if task_type.is_multiclass else torch.float)
        .to(device)
    )

    # >>> Model

    model: _Model = _Model(
        n_num_features=data['x_num']['train'].shape[-1] if 'x_num' in data else 0,
        cat_cardinalities=(
            [len(torch.unique(x)) for x in data['x_cat']['train'].T]
            if 'x_cat' in data
            else []
        ),
        n_classes=(
            len(np.unique(np.concat([x for x in task_labels.values()])))
            if task_type.is_binclass or task_type.is_multiclass
            else None
        ),
        **config.model,
    )
    report['n_parameters'] = sum(x.numel() for x in model.parameters())
    tabm_k = typing.cast(int, model.backbone.k) if is_tabm else None  # type: ignore[attr-defined]
    # For TabM, the ensemble is averaged over probabilities (see
    # `_ensemble_reduce`), so metrics are computed from probabilities.
    prediction_type = (
        lib.metrics.PredictionType.LABELS
        if task_type.is_regression
        else lib.metrics.PredictionType.PROBS
        if is_tabm
        else lib.metrics.PredictionType.LOGITS
    )
    report['prediction_type'] = prediction_type.value
    model.to(device)

    # >>> Training

    optimizer = _make_optimizer(model, **config.optimizer)
    base_loss_fn = _make_loss_fn(task_type)
    compute_metrics = _make_compute_metrics(task_type, prediction_type=prediction_type)

    if is_tabm:
        assert tabm_k is not None
        share_training_batches = config.share_training_batches

        def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
            # y_pred: (B, k) for regression/binclass, (B, k, C) for multiclass.
            return base_loss_fn(
                y_pred.flatten(0, 1),
                (
                    y_true.repeat_interleave(tabm_k, dim=0)
                    if share_training_batches
                    else y_true.flatten(0, 1)
                ),
            )
    else:
        loss_fn = base_loss_fn

    step = 0
    train_size = len(task_labels['train'])
    eval_batch_size = config.get_eval_batch_size()

    # The following generator is used only for creating training batches,
    # so the random seed fully determines the sequence of training objects.
    batch_generator = torch.Generator(device).manual_seed(config.seed)
    timer = lib.utils.time.Timer()
    remaining_patience = config.patience
    best_checkpoint = None

    # >>> Numerical logging
    #
    # A numerical log is a list of dictionaries of the same structure. Appending new
    # entries should _not_ involve slow operations, such as host-device synchronization.
    # At the end of the script, the logs are transformed as needed and saved on disk.

    steps_numlog = []
    epochs_numlog = []

    # >>> Efficiency

    amp_dtype: torch.dtype | None = (
        None if config.amp_dtype is None else getattr(torch, config.amp_dtype)
    )
    grad_scaler = (
        torch.amp.GradScaler(device.type)  # pyright: ignore[reportPrivateImportUsage]
        if amp_dtype is torch.float16
        else None
    )
    autocast = None if amp_dtype is None else torch.autocast(device.type, amp_dtype)
    if config.compile:
        # At the moment of writing, the argument `mode` is not used,
        # because it results in training issues.
        model.compile()
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    # >>> Functions

    apply_model = _apply_model_impl
    if autocast is not None:
        apply_model = autocast(apply_model)

    # >>> Batch size

    batch_size = config.batch_size
    if batch_size is None:
        raise ValueError('batch_size must be provided')
    report['batch_size'] = batch_size
    report['epoch_size'] = epoch_size = math.ceil(train_size / batch_size)

    # The following order of `evaluation_mode` and `partial` preserves
    # typing-related hints in VSCode.
    evaluate = evaluation_mode()(
        partial(
            _evaluate_impl,
            apply_model=apply_model,
            model=model,
            data=data,
            task_labels=task_labels,
            task_type=task_type,
            task_score=task_score,
            ranking_qids=ranking_qids,
            compute_metrics=compute_metrics,
            is_tabm=is_tabm,
            prediction_transform=(
                None
                if regression_label_transformer is None
                else regression_label_transformer.inverse_transform
            ),
            device=device,
        )
    )

    def make_checkpoint() -> dict[str, Any]:
        return deepcopy(
            {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'batch_generator': batch_generator.get_state(),
                'random_state': lib.utils.torch.get_random_state(),
                'remaining_patience': remaining_patience,
                'report': report,
                'timer': timer,
                **(
                    {}
                    if grad_scaler is None
                    else {'grad_scaler': grad_scaler.state_dict()}
                ),
            }
        )

    # >>> Training loop

    timer.run()
    while config.n_epochs is None or step // epoch_size < config.n_epochs:
        model.train()

        batch_losses = []
        epoch_step_offset = len(steps_numlog)
        epoch_training_start_time = time.perf_counter()
        # For TabM with `share_training_batches=False`, each ensemble member gets
        # an independently shuffled batch, so the batch index has shape (B, k).
        # See the official TabM notebook for the rationale.
        batches = (
            torch.randperm(train_size, generator=batch_generator, device=device).split(
                batch_size
            )
            if config.share_training_batches
            else torch.rand(
                (train_size, typing.cast(int, tabm_k)),
                generator=batch_generator,
                device=device,
            )
            .argsort(dim=0)
            .split(batch_size, dim=0)
        )
        for batch_idx in tqdm(
            batches,
            desc=str(exp),
            leave=False,
        ):
            optimizer.zero_grad()
            loss = loss_fn(
                apply_model(model, data, part='train', idx=batch_idx),
                Y_train[batch_idx],
            )
            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()

            if config.gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), config.gradient_clipping_norm
                )

            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()

            step += 1
            batch_losses.append(loss.detach())
            # NOTE: the loss value is filled in below, in one batched host-device
            # sync per epoch, instead of holding a GPU tensor per step for the
            # entire trial (which used to keep thousands of small CUDA
            # allocations alive at once and leak GPU memory across steps).
            steps_numlog.append({'step': step, 'time': timer.elapsed()})
        epoch_training_time = time.perf_counter() - epoch_training_start_time

        batch_losses_cpu = torch.stack(batch_losses).cpu().tolist()
        for i, loss_value in enumerate(batch_losses_cpu):
            steps_numlog[epoch_step_offset + i]['loss'] = loss_value

        (metrics, predictions), eval_batch_size = evaluate(
            parts=lib.data.evaluation_parts(task_labels), batch_size=eval_batch_size
        )
        val_score_improved = (
            'metrics' not in report
            or metrics['val']['score'] > report['metrics']['val']['score']
        )

        epochs_numlog.append(
            {'step': step, 'time': timer.elapsed(), 'metrics': metrics}
        )
        print(
            f'{"*" if val_score_improved else " "}'
            f' [epoch] {step // epoch_size:<3}'
            f' [val] {metrics["val"]["score"]:.3f}'
            f' [test] {metrics["test"]["score"]:.3f}'
            f' [loss] {sum(batch_losses_cpu) / len(batch_losses_cpu):.4f}'
            f' [time] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}'
            f' [it/s] {math.trunc(epoch_size / epoch_training_time):>3}'
        )

        if val_score_improved:
            remaining_patience = config.patience
            report['best_step'] = step
            report['metrics'] = metrics
            best_checkpoint = make_checkpoint()
        else:
            remaining_patience -= 1

        if remaining_patience < 0 or not _are_valid_predictions(predictions['val']):
            break

    report['time'] = timer.elapsed()
    print()

    # >>> Final evaluation

    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint['model'])
    (metrics, predictions), eval_batch_size = evaluate(
        parts=lib.data.evaluation_parts(task_labels, ['train', 'val', 'test']),
        batch_size=eval_batch_size,
    )
    report['eval_batch_size'] = eval_batch_size
    report['metrics'] = metrics

    # >>> Artifacts

    if config.save_checkpoint:
        torch.save(make_checkpoint(), exp / 'checkpoint.pt')

    numlog_dir = exp / 'numlog'
    numlog_dir.mkdir()
    np.savez(
        numlog_dir / 'steps.npz',
        **{
            k: torch.stack(v).cpu().numpy() if isinstance(v[0], Tensor) else np.array(v)
            for k, v in lib.utils.collections.transpose_dicts(
                lib.utils.collections.flatten_dict(x) for x in steps_numlog
            ).items()
        },  # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-argument-type]
    )
    np.savez(
        numlog_dir / 'epochs.npz',
        **{
            k: np.array(v)
            for k, v in lib.utils.collections.transpose_dicts(
                lib.utils.collections.flatten_dict(x) for x in epochs_numlog
            ).items()
        },  # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-argument-type]
    )

    # >>> Finish

    print(f'Report\n{lib.utils.misc.add_frame(lib.experiment.format_report(report))}\n')
    lib.experiment.finish(exp, main, report=report)
    return report


if __name__ == '__main__':
    lib.env.initialize()

    import cyclopts

    cyclopts.run(main, result_action='return_value')
