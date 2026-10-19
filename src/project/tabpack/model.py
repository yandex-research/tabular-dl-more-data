"""Train TabPack: a pack of MLPs trained simultaneously with online ensembling.

Adapted from `yandex-research/tabpack` (arXiv 2607.05380). Unlike `project.mlp`
and `project.tabm_`, TabPack's own hyperparameter tuning is internal: a
`HyperparameterSampler` resamples the configs of "dead" (early-stopped) pack
members while the rest of the pack keeps training. Because of that, this
module is invoked as a single experiment (`main(exp, *, force=False)`) rather
than through `lib.tools.tune`.
"""

import dataclasses
import itertools
import json
import logging
import math
import statistics
import time
import typing
from collections.abc import Callable, Generator, Mapping, Sequence
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, NotRequired, Protocol

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

import lib.data
import lib.env
import lib.experiment
import lib.metrics
import project.ranking
import lib.optim.utils
import lib.preprocessing
import lib.tools.tune
import lib.utils.collections
import lib.utils.dataclasses
import lib.utils.misc
import lib.utils.time
import lib.utils.torch
from lib.types import DataKey, KWArgs, PartKey

from . import ensemble_utils as pack_ensemble_utils
from . import ensemble_utils_torch as pack_ensemble_utils_torch
from . import nn as pack_nn
from . import optim as pack_optim
from . import utils as pack_utils
from .metrics import Metrics, calculate_metrics_pack
from .nn import BATCH_DIM, PACK_DIM, get_pack_size

type ConfigDict = KWArgs
type PredictionType = lib.metrics.PredictionType

_LOGGER = logging.getLogger(__name__)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Model
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _make_num_module(type: str, **kwargs) -> nn.Module:
    classes = (
        pack_nn.LinearEmbeddingsPack,
        pack_nn.LinearReLUEmbeddingsPack,
        pack_nn.CosineEmbeddingsPack,
    )
    cls = {x.__name__: x for x in classes}[type]
    return cls(**kwargs)


class ModelPack(pack_nn.ModulePack):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        pack_size: int,
        num_embeddings: None | KWArgs = None,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        if num_embeddings is None:
            self.num_module = None
            d_num_feature = 1
        else:
            self.num_module = _make_num_module(
                n_features=n_num_features, pack_size=pack_size, **num_embeddings
            )
            d_num_feature = num_embeddings.get('max_d_embedding')
            if d_num_feature is None:
                d_num_feature = num_embeddings['d_embedding']
            assert isinstance(d_num_feature, int)
        self.cat_module = (
            pack_nn.OneHotEncoding(cat_cardinalities) if cat_cardinalities else None
        )
        self.pack_view = pack_nn.PackView(pack_size=pack_size)
        self.backbone = pack_nn.MLPBackbonePack(
            d_in=d_num_feature * n_num_features + sum(cat_cardinalities),
            pack_size=pack_size,
            **backbone_kwargs,
        )
        self.output = pack_nn.LinearPack(
            backbone_kwargs['d_block'],
            1 if n_classes is None or n_classes == 2 else n_classes,
            max_in_features=backbone_kwargs.get('max_d_block'),
            pack_size=pack_size,
        )

        self._n_num_features = n_num_features
        self._cat_cardinalities = cat_cardinalities

    @property
    def pack_size(self) -> int:
        return self.backbone.pack_size

    def forward(
        self, x_num: None | Tensor, x_cat: None | Tensor, pack_idx: None | Tensor = None
    ) -> Tensor:
        assert x_num is not None or x_cat is not None

        x_list: list[Tensor] = []
        pack_view_used = False

        if x_num is None:
            assert self._n_num_features == 0
            assert self.num_module is None
        else:
            assert self._n_num_features > 0
            if self.num_module is None:
                x_list.append(x_num)
            else:
                x_num_ = self.pack_view(x_num, pack_idx)
                pack_view_used = True
                x_num_ = self.num_module(x_num_, pack_idx)
                x_list.append(x_num_.flatten(-2))
                del x_num_

        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x_cat_ = self.cat_module(x_cat).to(torch.get_default_dtype())
            if pack_view_used:
                x_cat_ = self.pack_view(x_cat_, pack_idx)
            x_list.append(x_cat_)
            del x_cat_

        x: Tensor = torch.cat(x_list, dim=-1)
        if not pack_view_used:
            x = self.pack_view(x, pack_idx)
        x = self.backbone(x, pack_idx)
        x = self.output(x, pack_idx)
        return x


class ApplyModel(Protocol):
    def __call__(
        self,
        model: nn.Module,
        data: dict[DataKey, dict[PartKey, Tensor]],
        *,
        part: PartKey,
        batch_idx: Tensor,
    ) -> Tensor: ...


def apply_model_impl(
    model: nn.Module,
    data: dict[DataKey, dict[PartKey, Tensor]],
    *,
    part: PartKey,
    batch_idx: Tensor,
    device: torch.device,
) -> Tensor:
    x_num_storage = data['x_num'][part] if 'x_num' in data else None
    x_cat_storage = data['x_cat'][part] if 'x_cat' in data else None
    storage = x_num_storage if x_num_storage is not None else x_cat_storage
    if storage is not None and batch_idx.device != storage.device:
        # `batch_idx` is generated on the GPU, but with `pin_memory` the
        # features are kept on the (pinned) CPU, so indices must follow them
        # before slicing; the slice is moved back to the GPU below.
        batch_idx = batch_idx.to(storage.device)
    x_num = x_num_storage[batch_idx] if x_num_storage is not None else None
    x_cat = x_cat_storage[batch_idx] if x_cat_storage is not None else None
    if x_num is not None:
        if x_num.device != device:
            x_num = x_num.to(device, non_blocking=True)
        # `x_num` may be stored in a reduced precision (see `Config.num_dtype`).
        # Upcast the (small) batch back to the default dtype so the forward
        # pass operates in a single dtype; under AMP the matmuls are downcast
        # again, so compute throughput is preserved.
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


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Hyperparameter sampler
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class HyperparameterSampler:
    """A simple wrapper around `optuna.study.Study`."""

    _WARMUP_SEED_SHIFT = 1000
    _BASIC_SAMPLERS = (
        'BruteForceSampler',
        'GridSampler',
        'RandomSampler',
        'QMCSampler',
    )

    @staticmethod
    def _make_sampler(type: str, **kwargs) -> optuna.samplers.BaseSampler:
        return getattr(optuna.samplers, type)(**kwargs)

    def __init__(
        self,
        *,
        space: dict[str, Any],
        type: str = 'TPESampler',
        strict_n_startup_trials: bool = False,
        study_kwargs: None | KWArgs = None,
        **kwargs,
    ) -> None:
        self._study = optuna.create_study(
            sampler=HyperparameterSampler._make_sampler(type, **kwargs),
            **({} if study_kwargs is None else study_kwargs),
        )
        self._strict_n_startup_trials = strict_n_startup_trials
        self._space = space
        self._trials: dict[int, optuna.trial.Trial] = {}

    @property
    def n_startup_trials(self) -> None | int:
        return getattr(self._study.sampler, '_n_startup_trials', None)

    def _load_n_finished_trials(self) -> int:
        return len(
            self._study.get_trials(
                False,
                (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
            )
        )

    def ask(self, index: int) -> dict[str, Any]:
        assert index not in self._trials
        trial = self._study.ask()
        if self.n_startup_trials is not None:
            trial.set_user_attr(
                'startup', self._load_n_finished_trials() < self.n_startup_trials
            )
        self._trials[index] = trial
        return lib.tools.tune._sample_config(trial, self._space, [])

    def tell(self, index: int, value: float) -> bool:
        trial = self._trials[index]
        should_skip = (
            self._strict_n_startup_trials
            and self.n_startup_trials is not None
            and trial.user_attrs.get('startup', False)
            and self._load_n_finished_trials() >= self.n_startup_trials
        )
        should_tell = not should_skip
        if should_tell:
            self._study.tell(self._trials[index], value)
        return should_tell


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Pack
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _find_arrays_and_tensors(x: Any) -> Generator[np.ndarray | Tensor]:
    if isinstance(x, np.ndarray | Tensor):
        yield x
    elif x is None or x is ... or isinstance(x, bool | int | float | str | bytes):
        return
    elif isinstance(x, Sequence):
        for item in x:
            yield from _find_arrays_and_tensors(item)
    elif isinstance(x, Mapping):
        for item in itertools.chain(x.keys(), x.values()):
            yield from _find_arrays_and_tensors(item)
    elif hasattr(x, '__dict__'):
        yield from _find_arrays_and_tensors(vars(x))


class StatePack:
    _BEST_STEP_INIT = np.iinfo(np.int64).min

    def __init__(self, *, pack_size: int, configs: None | list[ConfigDict]) -> None:
        assert pack_size >= 0

        # Static properties.
        self._ids = np.arange(pack_size)
        self._configs = deepcopy(configs)

        # Dynamic properties.
        self._steps = np.zeros(pack_size, dtype=np.int64)
        self._n_consequtive_bad_updates = np.zeros(pack_size, dtype=np.int64)
        self._best_metrics = {}
        self._best_steps = np.full(pack_size, StatePack._BEST_STEP_INIT)
        self._best_predictions = {}
        self._best_predictions_torch = {}
        self._best_model_state_dicts = {}

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def configs(self) -> None | list[dict]:
        return self._configs

    @property
    def steps(self) -> np.ndarray:
        return self._steps

    @property
    def n_consequtive_bad_updates(self) -> np.ndarray:
        return self._n_consequtive_bad_updates

    @property
    def best_metrics(self) -> dict[PartKey, Metrics]:
        return self._best_metrics

    @property
    def best_steps(self) -> np.ndarray:
        return self._best_steps

    @property
    def best_predictions(self) -> dict[PartKey, np.ndarray]:
        return self._best_predictions

    @property
    def best_predictions_torch(self) -> dict[PartKey, Tensor]:
        return self._best_predictions_torch

    @property
    def best_model_state_dicts(self) -> dict[str, Tensor]:
        return self._best_model_state_dicts

    @property
    def pack_size(self) -> int:
        return len(self.ids)

    def validate(self) -> None:
        # Validate the pack size.
        pack_size = self.pack_size
        if self.configs is not None:
            assert len(self.configs) == pack_size
        assert all(
            x.shape[PACK_DIM] == pack_size for x in _find_arrays_and_tensors(self)
        )

        # Validate the static properties.
        assert np.all(self.ids >= 0)
        assert len(np.unique(self.ids)) == pack_size

        # Validate the dynamic properties.
        assert np.all(self.steps >= 0)
        assert np.all(self.n_consequtive_bad_updates >= 0)
        assert np.all(self.best_steps <= self.steps)

    # NOTE
    # All the following methods MUTATE `self`.

    def step(self) -> None:
        self._steps += 1

    def remove(self, pack_idx: np.ndarray) -> None:
        assert len(pack_idx) > 0
        assert (self.steps[pack_idx] > 0).all()

        device = next(iter(self.best_model_state_dicts.values())).device
        keep_pack_idx_torch = pack_nn.make_keep_pack_idx(
            self.pack_size, torch.as_tensor(pack_idx)
        )
        keep_pack_idx = keep_pack_idx_torch.numpy()
        keep_pack_idx_torch = keep_pack_idx_torch.to(device)
        del pack_idx

        # NOTE
        # All static and dynamic properties corresponding to `pack_idx`
        # must be removed.

        # Remove the static properties.
        self._ids = self.ids[keep_pack_idx].copy()
        if self.configs is not None:
            self.configs[:] = [self.configs[i] for i in map(int, keep_pack_idx)]

        # Remove the dynamic properties.
        self._steps = self.steps[keep_pack_idx].copy()
        self._n_consequtive_bad_updates = self.n_consequtive_bad_updates[
            keep_pack_idx
        ].copy()
        for part_metrics in self.best_metrics.values():
            for key in list(part_metrics):
                part_metrics[key] = part_metrics[key][keep_pack_idx].copy()
        self._best_steps = self.best_steps[keep_pack_idx].copy()
        for part in list(self.best_predictions):
            self.best_predictions[part] = self.best_predictions[part][
                keep_pack_idx
            ].copy()
        for part in list(self.best_predictions):
            self.best_predictions_torch[part] = self.best_predictions_torch[part][
                keep_pack_idx_torch
            ].clone()
        for key, value in list(self.best_model_state_dicts.items()):
            self.best_model_state_dicts[key] = value[keep_pack_idx_torch]

    def update(
        self,
        metrics: dict[PartKey, Metrics],
        *,
        predictions: dict[PartKey, np.ndarray],
        predictions_torch: dict[PartKey, Tensor],
        model_state_dict: dict[str, Tensor],
    ) -> None:
        assert (self.steps > 0).all()

        if self.best_metrics:
            device = next(iter(model_state_dict.values())).device
            improved_mask = metrics['val']['score'] > self.best_metrics['val']['score']
            improved_pack_idx = np.nonzero(improved_mask)[0]

            if len(improved_pack_idx) > 0:
                improved_pack_idx_torch = torch.tensor(improved_pack_idx, device=device)
                self.n_consequtive_bad_updates[improved_pack_idx] = 0

                for part in self.best_metrics:
                    for key in self.best_metrics[part]:
                        self.best_metrics[part][key][improved_pack_idx] = metrics[part][
                            key
                        ][improved_pack_idx]
                self.best_steps[improved_pack_idx] = self.steps[improved_pack_idx]
                for part in self.best_predictions:
                    self.best_predictions[part][improved_pack_idx] = predictions[part][
                        improved_pack_idx
                    ]
                for part in self.best_predictions_torch:
                    self.best_predictions_torch[part][improved_pack_idx_torch] = (
                        predictions_torch[part][improved_pack_idx_torch]
                    )
                for key, value in model_state_dict.items():
                    self.best_model_state_dicts[key][improved_pack_idx_torch] = value[
                        improved_pack_idx_torch
                    ]
            self.n_consequtive_bad_updates[~improved_mask] += 1

        else:
            assert np.all(self.n_consequtive_bad_updates == 0)
            self._best_metrics = metrics
            self._best_steps[:] = self.steps
            self._best_predictions = deepcopy(predictions)
            self._best_predictions_torch = deepcopy(predictions_torch)
            self._best_model_state_dicts = deepcopy(model_state_dict)


class FinalStatePack:
    def __init__(self) -> None:
        self._ids = np.array([], dtype=np.int64)
        self._steps = np.array([], dtype=np.int64)
        self._predictions = {}
        self._predictions_torch = {}

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def steps(self) -> np.ndarray:
        return self._steps

    @property
    def predictions(self) -> dict[str, np.ndarray]:
        return self._predictions

    @property
    def predictions_torch(self) -> dict[str, Tensor]:
        return self._predictions_torch

    def __len__(self) -> int:
        return len(self.ids)

    def extend(
        self,
        *,
        ids: np.ndarray,
        steps: np.ndarray,
        predictions: dict[str, np.ndarray],
        predictions_torch: dict[str, Tensor],
    ) -> None:
        self._ids = np.concat([self._ids, ids])
        self._steps = np.concat([self._steps, steps])
        self._predictions = (
            {
                k: np.concat([self.predictions[k], predictions[k]])
                for k in predictions.keys()
            }
            if self.predictions
            else predictions
        )
        self._predictions_torch = (
            {
                k: torch.cat([self.predictions_torch[k], predictions_torch[k]])
                for k in predictions_torch.keys()
            }
            if self.predictions_torch
            else predictions_torch
        )


def pack_validate(
    model: ModelPack, optimizer: torch.optim.Optimizer, state: StatePack
) -> None:
    pack_size = model.pack_size
    assert state.pack_size == pack_size

    for x in itertools.chain(model.parameters(), model.buffers()):
        if isinstance(x, pack_nn.ParameterPack | pack_nn.BufferPack):
            assert get_pack_size(x) == pack_size

    assert pack_size == state.pack_size
    for x in itertools.chain.from_iterable(
        group['params'] for group in optimizer.param_groups
    ):
        assert get_pack_size(x) == pack_size

    state.validate()


def pack_remove(
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    state: StatePack,
    *,
    pack_idx: np.ndarray,
) -> None:
    assert len(pack_idx) > 0
    pack_idx_torch = torch.as_tensor(pack_idx, device=next(model.parameters()).device)
    old_to_new = pack_nn.module_pack_remove(model, pack_idx_torch)
    pack_optim.optimizer_pack_remove(optimizer, pack_idx_torch, old_to_new)
    state.remove(pack_idx)


def compute_stop_pack_idx(
    state: StatePack, *, epoch_size: int, n_epochs: int, patience: int
) -> None | np.ndarray:
    early_stop_mask = (
        state.n_consequtive_bad_updates > patience if patience >= 0 else None
    )
    epoch_stop_mask = state.steps // epoch_size >= n_epochs if n_epochs >= 0 else None
    stop_mask = (
        early_stop_mask
        if epoch_stop_mask is None
        else epoch_stop_mask
        if early_stop_mask is None
        else (early_stop_mask | epoch_stop_mask)
    )
    if stop_mask is None:
        return None
    stop_pack_idx = np.nonzero(stop_mask)[0]
    return None if len(stop_pack_idx) == 0 else stop_pack_idx


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Experiments
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class ExperimentDict(typing.TypedDict):
    config: NotRequired[KWArgs]
    report: lib.experiment.ReportDict


def assemble_experiments(
    *,
    state: StatePack,
    final_metrics: dict[PartKey, Metrics],
    pack_idx: np.ndarray,
    time_elapsed: float,
) -> list[ExperimentDict]:
    experiments = []
    for i, index in enumerate(pack_idx):
        experiment = {}
        if state.configs is not None:
            experiment['config'] = state.configs[index]
        experiment['report'] = {
            'id': int(state.ids[index]),
            'best_step': int(state.best_steps[index]),
            'metrics': {
                part: {k: float(v[i]) for k, v in part_metrics.items()}
                for part, part_metrics in final_metrics.items()
            },
            # NOTE
            # The time computed below is not representative of the time
            # that running this one experiment would take in isolation.
            'time': time_elapsed,
        }
        experiments.append(experiment)
    return experiments


def get_experiment_val_score(experiment: ExperimentDict) -> float:
    return experiment['report']['metrics']['val']['score']


def get_best_experiment(
    current_best_experiment: None | ExperimentDict,
    new_experiments: list[ExperimentDict],
) -> tuple[ExperimentDict, bool]:
    assert new_experiments

    best_experiment = current_best_experiment
    del current_best_experiment

    best_experiment_improved = False
    for experiment in new_experiments:
        if best_experiment is None or get_experiment_val_score(
            experiment
        ) > get_experiment_val_score(best_experiment):
            best_experiment = experiment
            best_experiment_improved = True

    assert best_experiment is not None
    return best_experiment, best_experiment_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Ensembles
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class OnlineEnsemble:
    def __init__(
        self,
        *,
        type: str,
        options: None | KWArgs = None,
        algorithm_score_fn: None | pack_ensemble_utils_torch.EnsembleScoreFn = None,
        update_type: Literal['final', 'best', 'latest'],
        update_part: PartKey,
        include_current_ensemble_in_pool: bool = False,
        prediction_type: lib.metrics.PredictionType,
        score_fn: pack_ensemble_utils_torch.EnsembleScoreFn,
        patience: int,
    ) -> None:
        # Algorithm details
        self._ensemble_fn = pack_ensemble_utils_torch.get_ensemble_fn(type)
        self._ensemble_fn_kwargs = {} if options is None else options
        self._ensemble_fn_kwargs['score_fn'] = (
            score_fn if algorithm_score_fn is None else algorithm_score_fn
        )
        if type == 'autotopk' or (
            type == 'greedy'
            and options is not None
            and options.get('init_autotopk', False)
        ):
            self._ensemble_fn_kwargs['prediction_type'] = prediction_type
        self._update_type = update_type
        self._include_current_ensemble_in_pool = include_current_ensemble_in_pool

        # Current ensemble
        self._ids = np.array([], dtype=np.int64)
        self._steps = np.array([], dtype=np.int64)
        self._predictions: dict[PartKey, np.ndarray] = {}
        self._predictions_torch: dict[PartKey, Tensor] = {}
        self._weights: None | np.ndarray = None
        self._score: None | float = None

        # Task properties
        self._prediction_type = prediction_type

        # Update settings
        self._update_part = update_part
        self._score_fn = score_fn
        self._patience = patience
        self._remaining_patience = patience

        #
        self._time_elapsed = 0.0

    @property
    def ids(self):
        return self._ids

    @property
    def steps(self):
        return self._steps

    @property
    def predictions(self):
        return self._predictions

    @property
    def weights(self):
        return self._weights

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @property
    def is_running(self) -> bool:
        return self._remaining_patience >= 0

    def _prepare_pool(
        self,
        running_ids: np.ndarray,
        running_steps: np.ndarray,
        running_latest_predictions: dict[str, np.ndarray],
        running_latest_predictions_torch: dict[str, Tensor],
        running_best_predictions: dict[str, np.ndarray],
        running_best_predictions_torch: dict[str, Tensor],
        finished_ids: np.ndarray,
        finished_steps: np.ndarray,
        finished_predictions: dict[str, np.ndarray],
        finished_predictions_torch: dict[str, Tensor],
    ):
        if self._update_type == 'final':
            pool_ids = finished_ids
            pool_steps = finished_steps
            pool_predictions = finished_predictions
            pool_predictions_torch = finished_predictions_torch

        else:
            if self._update_type == 'best':
                running_predictions = running_best_predictions
                running_predictions_torch = running_best_predictions_torch
            else:
                assert self._update_type == 'latest'
                running_predictions = running_latest_predictions
                running_predictions_torch = running_latest_predictions_torch

            if len(finished_ids) == 0:
                pool_ids = running_ids
                pool_steps = running_steps
                pool_predictions = running_predictions
                pool_predictions_torch = running_predictions_torch

            else:
                pool_ids = np.concat([finished_ids, running_ids])
                pool_steps = np.concat([finished_steps, running_steps])
                pool_predictions = {
                    k: np.concat([finished_predictions[k], running_predictions[k]])
                    for k in running_predictions.keys()
                }
                pool_predictions_torch = {
                    k: torch.cat(
                        [finished_predictions_torch[k], running_predictions_torch[k]]
                    )
                    for k in running_predictions_torch.keys()
                }

            del running_predictions, running_predictions_torch
        del finished_predictions, finished_predictions_torch
        del running_best_predictions, running_best_predictions_torch
        del running_latest_predictions, running_latest_predictions_torch

        if self._include_current_ensemble_in_pool and len(self.ids) > 0:
            pool_ids = np.concat([self._ids, pool_ids])
            pool_steps = np.concat([self._steps, pool_steps])
            if self._predictions:
                pool_predictions = {
                    k: np.concat([self._predictions[k], pool_predictions[k]])
                    for k in pool_predictions.keys()
                }
            if self._predictions_torch:
                pool_predictions_torch = {
                    k: torch.cat(
                        [self._predictions_torch[k], pool_predictions_torch[k]]
                    )
                    for k in pool_predictions_torch.keys()
                }

        return pool_ids, pool_steps, pool_predictions, pool_predictions_torch

    def update(self, **kwargs) -> bool:
        assert self.is_running
        if self._update_type == 'final' and len(kwargs['finished_ids']) == 0:
            return False

        start_time = time.perf_counter()

        (pool_ids, pool_steps, pool_predictions, pool_predictions_torch) = (
            self._prepare_pool(**kwargs)
        )
        ensemble = self._ensemble_fn(
            pool_predictions_torch[self._update_part], **self._ensemble_fn_kwargs
        )
        ensemble_idx, ensemble_weights = (
            (ensemble, None) if isinstance(ensemble, Tensor) else ensemble
        )

        ensemble_part_prediction = (
            pack_ensemble_utils_torch.compute_ensemble_prediction(
                pool_predictions_torch[self._update_part][ensemble_idx],
                ensemble_weights,
                self._prediction_type,
            )
        )
        score = self._score_fn(ensemble_part_prediction[None]).item()
        improved = self._score is None or score > self._score

        if improved:
            self._remaining_patience = self._patience

            ensemble_idx_numpy = ensemble_idx.cpu().numpy()
            self._ids = pool_ids[ensemble_idx_numpy]
            self._steps = pool_steps[ensemble_idx_numpy]
            self._predictions = {
                k: v[ensemble_idx_numpy] for k, v in pool_predictions.items()
            }
            self._predictions_torch = {
                k: v[ensemble_idx] for k, v in pool_predictions_torch.items()
            }
            self._weights = (
                ensemble_weights
                if ensemble_weights is None
                else ensemble_weights.cpu().numpy()
            )
            self._score = score

        else:
            self._remaining_patience -= 1

        self._time_elapsed += time.perf_counter() - start_time
        return improved


def _calculate_ensemble_metrics(
    predictions: dict[PartKey, np.ndarray],
    *,
    task_labels: dict[PartKey, np.ndarray],
    task_type: lib.data.TaskType,
    task_score: lib.data.TaskScore,
    ranking_qids: dict[PartKey, np.ndarray] | None,
    prediction_type: lib.metrics.PredictionType,
) -> dict[PartKey, dict[str, float]]:
    # `calculate_metrics_pack` expects a leading pack dimension, so a dummy
    # pack of size 1 is added and then squeezed out of the result.
    metrics = {}
    for part, y_pred in predictions.items():
        part_metrics = calculate_metrics_pack(
            y_true=task_labels[part],
            y_pred=y_pred[None],
            task_type=task_type,
            prediction_type=prediction_type,
            score=task_score,
            ranking_qid=None if ranking_qids is None else ranking_qids[part],
        )
        metrics[part] = {k: float(v[0]) for k, v in part_metrics.items()}
    return metrics


def update_online_ensembles(
    online_ensembles: dict[str, OnlineEnsemble],
    *,
    task_labels: dict[PartKey, np.ndarray],
    task_type: lib.data.TaskType,
    task_score: lib.data.TaskScore,
    ranking_qids: dict[PartKey, np.ndarray] | None,
    step: int,
    timer: lib.utils.time.Timer,
    **kwargs,
) -> tuple[dict[str, lib.experiment.ReportDict], bool]:
    reports = {}
    first_online_ensemble_improved = False
    for i_ensemble, (ensemble_name, ensemble) in enumerate(online_ensembles.items()):
        if not ensemble.is_running:
            continue

        ensemble_improved = ensemble.update(**kwargs)
        if i_ensemble == 0:
            first_online_ensemble_improved = ensemble_improved

        if ensemble_improved:
            ensemble_predictions = {
                k: pack_ensemble_utils.compute_ensemble_prediction(
                    v, ensemble.weights, ensemble._prediction_type
                )
                for k, v in ensemble._predictions.items()
            }
            metrics = _calculate_ensemble_metrics(
                ensemble_predictions,
                task_labels=task_labels,
                task_type=task_type,
                task_score=task_score,
                ranking_qids=ranking_qids,
                prediction_type=ensemble._prediction_type,
            )

            reports[ensemble_name] = {
                'ids': ensemble.ids.tolist(),
                **(
                    {}
                    if ensemble.weights is None
                    else {'weights': ensemble.weights.tolist()}
                ),
                'step': step,
                'steps': ensemble.steps.tolist(),
                'time': timer.elapsed(),
                'ensemble_time': ensemble.time_elapsed,
                'metrics': metrics,
            }

    return reports, first_online_ensemble_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Training
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def generate_training_batches(
    *,
    train_size: int,
    batch_size: int,
    batch_generator: torch.Generator,
    pack_size: int,
) -> list[Tensor]:
    """Generate training batches for one epoch."""
    random_values = torch.rand(
        (pack_size, train_size),
        generator=batch_generator,
        device=batch_generator.device,
    )
    batches = random_values.argsort(dim=BATCH_DIM).split(batch_size, dim=BATCH_DIM)
    batches = list(batches)
    return batches


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Evaluation
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class _EvaluateOutput(NamedTuple):
    metrics: dict[PartKey, Metrics]
    predictions: dict[PartKey, np.ndarray]
    predictions_torch: dict[PartKey, Tensor]


@lib.utils.torch.adjust_gpu_memory_usage('batch_size')
def _evaluate(
    apply_model: ApplyModel,
    model: nn.Module,
    data: dict[DataKey, dict[PartKey, Tensor]],
    *,
    parts: list[PartKey],
    task_labels: dict[PartKey, np.ndarray],
    task_type: lib.data.TaskType,
    task_score: lib.data.TaskScore,
    ranking_qids: dict[PartKey, np.ndarray] | None,
    regression_label_transformer: None | lib.preprocessing.StandardScaler1d,
    prediction_type: lib.metrics.PredictionType,
    batch_size: int,
    device: torch.device,
) -> _EvaluateOutput:
    model.eval()

    metrics = {}
    predictions = {}
    predictions_torch = {}

    for part in parts:
        y_pred_torch = torch.cat(
            [
                apply_model(model, data, part=part, batch_idx=batch_idx)
                for batch_idx in torch.arange(
                    len(task_labels[part]), device=device
                ).split(batch_size)
            ],
            dim=BATCH_DIM,
        )

        if task_type.is_regression:
            assert regression_label_transformer is not None
            y_pred_torch = (
                y_pred_torch * regression_label_transformer.scale_
                + regression_label_transformer.mean_
            )

        elif task_type.is_binclass:
            y_pred_torch = torch.special.expit(y_pred_torch)

        else:
            assert task_type.is_multiclass
            y_pred_torch = torch.special.softmax(y_pred_torch, dim=-1)

        y_pred = y_pred_torch.cpu().numpy()

        assert np.isfinite(y_pred).all()
        metrics[part] = calculate_metrics_pack(
            y_true=task_labels[part],
            y_pred=y_pred,
            task_type=task_type,
            prediction_type=prediction_type,
            score=task_score,
            ranking_qid=None if ranking_qids is None else ranking_qids[part],
        )

        predictions[part] = y_pred
        predictions_torch[part] = y_pred_torch

    return _EvaluateOutput(metrics, predictions, predictions_torch)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Utilities
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
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


def _make_Y_train(
    data: dict[DataKey, dict[PartKey, Tensor]], task_type: lib.data.TaskType
) -> Tensor:
    return data['y']['train'].to(torch.long if task_type.is_multiclass else torch.float)


def _make_online_ensembles(
    online_ensemble_configs: dict[str, KWArgs],
    *,
    task_labels: dict[PartKey, np.ndarray],
    task_type: lib.data.TaskType,
    task_score: lib.data.TaskScore,
    prediction_type: lib.metrics.PredictionType,
    update_part: PartKey,
    device: torch.device,
) -> dict[str, OnlineEnsemble]:
    y_true = (
        torch.as_tensor(task_labels[update_part])
        .to(
            torch.long
            if (task_type.is_binclass or task_type.is_multiclass)
            else torch.float
        )
        .to(device)
    )

    score_fn = pack_ensemble_utils_torch.make_emsemble_score_fn(
        y_true=y_true,
        task_type=task_type,
        prediction_type=prediction_type,
        score=task_score,
    )
    loss_score_fn = (
        pack_ensemble_utils_torch.make_emsemble_score_fn(
            y_true=y_true,
            task_type=task_type,
            prediction_type=prediction_type,
            score=lib.data.TaskScore.CROSS_ENTROPY,
        )
        if task_type.is_binclass or task_type.is_multiclass
        else None
    )

    online_ensembles = {}
    for name, ensemble_config in online_ensemble_configs.items():
        algorithm_score_fn = ensemble_config.get('algorithm_score_fn')
        if algorithm_score_fn == 'loss':
            ensemble_config['algorithm_score_fn'] = loss_score_fn
        online_ensembles[name] = OnlineEnsemble(
            update_part=update_part,
            prediction_type=prediction_type,
            score_fn=score_fn,
            **ensemble_config,
        )

    return online_ensembles


def _make_muon_scale(linear: pack_nn.LinearPack) -> Tensor:
    pack_size = linear.pack_size
    out_f = (
        linear.out_features.float()
        if linear.out_features is not None
        else torch.full(
            (pack_size,),
            float(linear.max_out_features),
            device=linear.weight.device,
        )
    )
    in_f = (
        linear.in_features.float()
        if linear.in_features is not None
        else torch.full(
            (pack_size,),
            float(linear.max_in_features),
            device=linear.weight.device,
        )
    )
    return (out_f / in_f).clamp_(min=1.0).sqrt_()


def _make_optimizer(model: ModelPack, *, type: str, **kwargs) -> torch.optim.Optimizer:
    parameter_groups = lib.optim.utils.ParameterGroups()

    # Muon optimizes the 2D (+ pack dimension) weight matrices of the
    # backbone's linear layers.
    if type == 'MuonAdamWPack':
        for block in model.backbone._iter_blocks():
            muon_kwargs = {'muon_update_scale': _make_muon_scale(block.linear)}
            if block.linear.sparsity is not None:
                # A hard-sparse first layer doesn't fit Muon's orthogonalized
                # update, so those pack members always train it with AdamW
                # instead, regardless of the pack's optimizer type.
                muon_kwargs['muon_slot_mask'] = block.linear.sparsity <= 0.0
            parameter_groups.add([block.linear.weight], muon=True, **muon_kwargs)

    # Parameters that should not be weight-decayed: biases and the packed
    # numerical-embedding layer (mirrors the embedding-related special case
    # from the upstream `_default_zero_weight_decay_condition`).
    parameter_groups.add(
        (
            p
            for m in model.modules()
            for pn, p in m.named_parameters(recurse=False)
            if pn.endswith('bias') or isinstance(m, pack_nn.LinearEmbeddingsPack)
        ),
        weight_decay=0.0,
    )
    parameter_groups.add(model.parameters())

    optimizer_cls = {
        'AdamWPack': pack_optim.AdamWPack,
        'MuonAdamWPack': pack_optim.MuonAdamWPack,
    }[type]
    return optimizer_cls(parameter_groups.unwrap(), pack_size=model.pack_size, **kwargs)


def _make_loss_fn_pack(
    task_type: lib.data.TaskType,
) -> Callable[[Tensor, Tensor], Tensor]:
    base_loss_fn = {
        lib.data.TaskType.REGRESSION: nn.functional.mse_loss,
        lib.data.TaskType.BINCLASS: nn.functional.binary_cross_entropy_with_logits,
        lib.data.TaskType.MULTICLASS: nn.functional.cross_entropy,
    }[task_type]

    def loss_fn_pack(y_pred: Tensor, y_true: Tensor) -> Tensor:
        pack_size = get_pack_size(y_pred)
        losses = base_loss_fn(
            y_pred.flatten(0, 1), y_true.flatten(0, 1), reduction='none'
        )
        losses = losses.unflatten(0, (pack_size, y_true.shape[BATCH_DIM]))
        losses = losses.flatten(BATCH_DIM)
        return losses.mean(BATCH_DIM)

    return loss_fn_pack


def _get_mean_scores(
    current_mean_scores: None | dict[PartKey, float],
    experiments: list[ExperimentDict],
    state: StatePack,
    eval_parts: list[PartKey],
) -> tuple[dict[PartKey, float], bool]:
    scores = {}
    for experiment in experiments:
        for part in eval_parts:
            scores.setdefault(part, []).append(
                experiment['report']['metrics'][part]['score']
            )
    for part in eval_parts:
        scores.setdefault(part, []).extend(
            state.best_metrics[part]['score'][state.steps > 0].tolist()
        )
    assert scores

    mean_scores = {k: statistics.mean(v) for k, v in scores.items()}
    return mean_scores, (
        current_mean_scores is None or mean_scores['val'] > current_mean_scores['val']
    )


def _get_best_scores(
    current_best_experiment: None | ExperimentDict,
    current_best_scores: None | dict[PartKey, float],
    state: StatePack,
) -> tuple[dict[PartKey, float], bool]:
    if current_best_experiment is not None and (
        current_best_scores is None
        or get_experiment_val_score(current_best_experiment)
        > current_best_scores['val']
    ):
        current_best_scores = {
            part: current_best_experiment['report']['metrics'][part]['score']
            for part in state.best_metrics.keys()
        }
        best_scores_improved = True
    else:
        best_scores_improved = False

    if state.pack_size == 0 or (state.steps == 0).all():
        assert current_best_scores is not None
        return current_best_scores, best_scores_improved

    else:
        val_scores = state.best_metrics['val']['score']
        best_index = np.argmax(val_scores)
        if (
            current_best_scores is None
            or val_scores[best_index] > current_best_scores['val']
        ):
            return (
                {
                    part: float(part_metrics['score'][best_index])
                    for part, part_metrics in state.best_metrics.items()
                },
                True,
            )
        else:
            assert current_best_scores is not None
            return current_best_scores, best_scores_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Main
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int
    data: KWArgs

    # Model
    n_models: int
    model: KWArgs

    # Training
    optimizer: KWArgs
    batch_size: int
    n_epochs: int
    patience: int

    # Evaluation
    eval_parts: tuple[PartKey, ...] = ('val', 'test')
    eval_batch_size: int = 32768

    # Ensembles
    online_ensembles: None | KWArgs = None

    # Configs
    configs: None | list[ConfigDict] = None
    sampler: None | KWArgs = None

    # Efficiency
    amp_dtype: None | Literal['bfloat16'] = None
    timeout: None | int = None
    # Storage dtype for the numerical features (`x_num`). Casting to a lower
    # precision (e.g. "bfloat16") roughly halves the resident GPU memory of the
    # numerical features. Each batch is upcast back to the default dtype before
    # the forward pass, so model weights and compute are unaffected.
    num_dtype: None | Literal['bfloat16', 'float16'] = None
    # If True, the feature tensors (x_num, x_cat) are kept in pinned host
    # memory and transferred to GPU batch-wise inside `apply_model`, instead of
    # being fully materialized on the device. Useful for datasets whose
    # `x_cat` does not fit in GPU memory (avazu-x4-001's is ~7 GB in int64).
    pin_memory: bool = False

    # Report
    track_experiments: None | bool = None
    track_best_experiment: None | bool = None
    track_online_ensemble_history: bool = False
    track_memory_usage: bool = False

    # Output
    save_final_predictions: bool = True
    save_all_predictions: bool = False
    save_checkpoint: bool = False


def _validate_config(config: Config) -> None:
    if config.configs is not None:
        assert len(config.configs) == config.n_models

    if (
        config.sampler is not None
        and config.sampler.get('type') not in HyperparameterSampler._BASIC_SAMPLERS
    ):
        raise ValueError(
            'Given the provided sampler config, pack_size must be set explicitly,'
            ' because it can have non-trivial impact on the results'
        )


def _prepare_configs(
    config: Config, hyperparameter_sampler: None | HyperparameterSampler
) -> None | list[ConfigDict]:
    if config.configs is None:
        if hyperparameter_sampler is None:
            return None
        else:
            return [hyperparameter_sampler.ask(x) for x in range(config.n_models)]
    else:
        assert hyperparameter_sampler is None
        return config.configs


def _prepare_model_config(
    config: Config, model_configs_T: None | dict[str, list[Any]]
) -> ConfigDict:
    def infer_max_dimension_(
        subconfig: ConfigDict,
        key: str,
        value: None | int | list[int],
        value_space: None | list[Any],
        value_list: None | list[int],
    ) -> None:
        max_key = f'max_{key}'
        if value is None:
            if value_space is None:
                assert value_list is not None
                assert value_list
                subconfig[max_key] = max(value_list)

                assert max_key in subconfig, (
                    f'The model argument `{max_key}` cannot be inferred from the'
                    ' config, so it must be provided explicitly'
                )
            else:
                # value_space: ["_tune_", "int", min_value, max_value[, step]]
                subconfig[max_key] = value_space[3]
        else:
            assert max_key not in subconfig, (
                f'When the model argument `{key}` is provided explicitly,'
                f' the model argument `{max_key}` must be omitted, which is not true'
            )
            assert value_space is None, (
                f'The model argument `{key}` is presented as both a specific value and'
                ' as a part of the sampler space, which is not allowed'
            )
            subconfig[max_key] = max(value) if isinstance(value, list) else None

    model_config = deepcopy(config.model)
    sampler_space = {} if config.sampler is None else config.sampler.get('space', {})

    # Infer the maximum feature embedding dimension.
    model_num_embeddings_config = model_config.get('num_embeddings')
    if model_num_embeddings_config is not None:
        d_embedding_value = model_num_embeddings_config.get('d_embedding')
        d_embedding_value_space = (
            sampler_space.get('model', {}).get('num_embeddings', {}).get('d_embedding')
        )
        d_embedding_value_list = (
            None
            if model_configs_T is None
            or 'num_embeddings' not in model_configs_T
            or 'd_embedding' not in model_configs_T['num_embeddings'][0]
            else [x['d_embedding'] for x in model_configs_T['num_embeddings']]
        )
        if (
            d_embedding_value is not None
            or d_embedding_value_space is not None
            or d_embedding_value_list is not None
        ):
            infer_max_dimension_(
                model_num_embeddings_config,
                'd_embedding',
                d_embedding_value,
                d_embedding_value_space,
                d_embedding_value_list,
            )

    # Infer the maximum backbone dimensions.
    for key in ['n_blocks', 'd_block']:
        infer_max_dimension_(
            model_config,
            key,
            model_config.get(key),
            sampler_space.get('model', {}).get(key),
            None if model_configs_T is None else model_configs_T.get(key),
        )

    if model_configs_T is not None:
        # Merge the configs to the main config.
        pack_utils.dict_merge_recursively(
            model_config,
            (
                model_configs_T
                | {
                    key: pack_utils.transpose_list_of_dicts(model_configs_T[key])
                    for key in ('num_embeddings', 'activation')
                    if key in model_configs_T
                }
            ),
        )

    return model_config


def main(exp: str | Path, *, force: bool = False) -> lib.experiment.ReportDict | None:
    # >>> Start

    if not lib.experiment.try_start(exp, main, force=force):
        print(f'The experiment is already completed: {exp}')
        return None

    exp = lib.experiment.check(exp)
    config = lib.utils.dataclasses.from_dict(Config, lib.experiment.load_config(exp))
    _validate_config(config)
    report = lib.experiment.create_report(exp, main)
    print(f'{lib.experiment.format_config(config)}\n')

    # >>> Setup

    lib.utils.torch.seed(config.seed)
    device = lib.utils.torch.get_device()
    _LOGGER.debug(f'Device: {device}')

    if config.track_memory_usage:
        torch.cuda.reset_peak_memory_stats(device)

    amp_dtype: torch.dtype | None = (
        None if config.amp_dtype is None else getattr(torch, config.amp_dtype)
    )
    autocast = None if amp_dtype is None else torch.autocast(device.type, amp_dtype)

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

    num_storage_dtype: torch.dtype | None = (
        None if config.num_dtype is None else getattr(torch, config.num_dtype)
    )
    data_device = device if not config.pin_memory else torch.device('cpu')

    def _place(key: DataKey, array: Any) -> Tensor:
        t = torch.as_tensor(array)
        # Cast the numerical features to the reduced storage dtype *on the host*
        # before moving them to the device, so the device never materializes the
        # full-precision copy (the float32 materialization is what triggers OOM
        # on large datasets such as maps-routing-big, especially with a whole
        # pack of models trained simultaneously).
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
    Y_train = _make_Y_train(data, task_type).to(device)
    train_size = len(task_labels['train'])

    # >>> Hyperparameter sampler

    if config.sampler is None:
        hyperparameter_sampler = None
    else:
        hyperparameter_sampler = HyperparameterSampler(
            seed=config.seed,
            study_kwargs={'study_name': str(exp), 'direction': 'maximize'},
            **config.sampler,
        )
        _LOGGER.info('Using hyperparameter sampler')

    # >>> State

    state = StatePack(
        pack_size=config.n_models,
        configs=_prepare_configs(config, hyperparameter_sampler),
    )
    final_state = FinalStatePack()
    # Save all configs before training.
    all_configs = deepcopy(state.configs) if state.configs is not None else None
    configs_T = (
        None
        if state.configs is None
        else pack_utils.transpose_list_of_dicts(deepcopy(state.configs))
    )

    # >>> Model

    model = ModelPack(
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
        pack_size=state.pack_size,
        **_prepare_model_config(
            config,
            (
                None
                if configs_T is None or 'model' not in configs_T
                else pack_utils.transpose_list_of_dicts(configs_T.pop('model'))
            ),
        ),
    )
    model.to(device)

    # NOTE
    # Predictions must be stored in aggregation-friendly units
    # for ensembling purposes (raw logits do _not_ meet this requirement).
    prediction_type = (
        lib.metrics.PredictionType.LABELS
        if task_type.is_regression
        else lib.metrics.PredictionType.PROBS
    )
    apply_model = partial(apply_model_impl, device=device)
    apply_model = apply_model if autocast is None else autocast(apply_model)

    # >>> Ensembles

    online_ensemble_configs = config.online_ensembles
    if online_ensemble_configs is None:
        online_ensembles = None
        online_ensemble_history = None
    else:
        online_ensembles = _make_online_ensembles(
            online_ensemble_configs,
            task_labels=task_labels,
            task_type=task_type,
            task_score=task_score,
            prediction_type=prediction_type,
            update_part='val',
            device=device,
        )
        online_ensemble_history = {}

    # >>> Training

    optimizer = _make_optimizer(
        model,
        **config.optimizer,
        **(
            {}
            if configs_T is None or 'optimizer' not in configs_T
            else pack_utils.transpose_list_of_dicts(configs_T.pop('optimizer'))
        ),
    )

    assert not configs_T, (
        'The following fields of the generated configs were not used:'
        f' {", ".join(configs_T)}'
    )
    del configs_T

    loss_fn = _make_loss_fn_pack(task_type)
    epoch_size = math.ceil(train_size / config.batch_size)

    # >>> Evaluation

    evaluate = torch.inference_mode()(
        partial(
            _evaluate,
            apply_model,
            model,
            data,
            task_labels=task_labels,
            task_type=task_type,
            task_score=task_score,
            ranking_qids=ranking_qids,
            regression_label_transformer=regression_label_transformer,
            prediction_type=prediction_type,
            device=device,
        )
    )

    eval_parts = lib.data.evaluation_parts(task_labels, config.eval_parts)
    assert 'val' in eval_parts
    eval_batch_size = config.eval_batch_size

    # >>> Global state

    step = 0
    batch_generator = torch.Generator(device).manual_seed(config.seed)
    experiments: list[ExperimentDict] = []

    # >>> Numerical logs
    #
    # A numerical log is a list of Python objects of the same type (usually,
    # dictionaries holding numerical data, e.g. arrays, tensors, etc.).
    #
    # During training, simply append new records and avoid expensive
    # operations, such as CUDA-to-CPU conversions. After the training, the
    # logs are transformed as needed and saved on disk.
    steps_numlog = []
    epochs_numlog = []
    pack_epochs_numlog = []

    # `mean_scores` and `best_scores` are maintained only for printing
    # and rely on incomplete experiments (i.e. on the data from `state`).
    mean_scores = None
    mean_scores_improved = False
    best_scores = None
    best_scores_improved = False
    first_online_ensemble_scores = None
    first_online_ensemble_improved = False

    # >>> Report

    report['n_models'] = 0
    if hyperparameter_sampler is not None:
        report['n_trials'] = 0
    report['prediction_type'] = prediction_type.value
    report['epoch_size'] = epoch_size
    if online_ensembles is not None:
        report['online_ensembles'] = {
            k: {'report': {}} for k in online_ensembles.keys()
        }
    track_experiments = (
        hyperparameter_sampler is None
        if config.track_experiments is None
        else config.track_experiments
    )
    track_best_experiment = (
        # If individual models have different hyperparameters,
        # save the best experiment by default.
        state.configs is not None
        if config.track_best_experiment is None
        else config.track_best_experiment
    )

    # >>> Training loop

    print()
    timer = lib.utils.time.Timer()
    timer.run()
    pack_validate(model, optimizer, state)

    while (
        report['n_models'] < config.n_models
        and (config.timeout is None or timer.elapsed() < config.timeout)
        and (
            online_ensembles is None
            or any(x.is_running for x in online_ensembles.values())
        )
    ):
        # >>> Validation
        n_remaining_models = config.n_models - report['n_models']
        assert model.pack_size <= n_remaining_models

        # >>> Training phase
        epoch_training_start_time = time.perf_counter()
        model.train()

        batches = generate_training_batches(
            train_size=train_size,
            batch_size=config.batch_size,
            batch_generator=batch_generator,
            pack_size=state.pack_size,
        )
        batch_losses = []
        batch_sizes = []
        for batch_idx in tqdm(
            batches,
            desc=str(exp),
            leave=False,
        ):
            losses = loss_fn(
                apply_model(model, data, part='train', batch_idx=batch_idx),
                Y_train[batch_idx],
            )
            # The scale of the gradients should not depend on the number of
            # models, so the individual losses are summed, not averaged.
            loss = losses.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            state.step()

            loss_detached = loss.detach()
            batch_losses.append(loss_detached)
            batch_sizes.append(batch_idx.shape[BATCH_DIM])
            steps_numlog.append(
                {
                    'step': step,
                    'time': timer.elapsed(),
                    'batch_size': batch_sizes[-1],
                    'loss': loss_detached,
                }
            )

        epoch_training_duration = time.perf_counter() - epoch_training_start_time
        del batches, batch_idx, losses, loss, loss_detached

        if config.track_memory_usage:
            report['memory-usage'] = torch.cuda.memory_stats(device)
            break

        # >>> Evaluation phase

        (eval_metrics, eval_predictions, eval_predictions_torch), eval_batch_size = (
            evaluate(parts=eval_parts, batch_size=eval_batch_size, ranking_qids=None)
        )
        report['eval_batch_size'] = eval_batch_size
        state.update(
            eval_metrics,
            predictions=eval_predictions,
            predictions_torch=eval_predictions_torch,
            model_state_dict=model.state_dict(),
        )

        pack_epochs_numlog.append(
            deepcopy(
                {
                    'step': state.steps,
                    'time': np.full((state.pack_size,), timer.elapsed()),
                    'size': np.array([state.pack_size]),
                    'id': state.ids,
                    'metrics': eval_metrics,
                    **(
                        {'predictions': eval_predictions}
                        if config.save_all_predictions
                        else {}
                    ),
                }
            )
        )

        # Determine which models to stop.
        pack_size_before_stopping = model.pack_size
        stop_pack_idx = compute_stop_pack_idx(
            state,
            epoch_size=epoch_size,
            n_epochs=config.n_epochs,
            patience=config.patience,
        )

        if stop_pack_idx is None:
            latest_predictions = eval_predictions
            latest_predictions_torch = eval_predictions_torch

        else:
            stop_pack_idx_torch = torch.as_tensor(stop_pack_idx)
            n_models_to_stop = len(stop_pack_idx)

            keep_pack_idx_torch = pack_nn.make_keep_pack_idx(
                state.pack_size, stop_pack_idx_torch
            )
            keep_pack_idx = keep_pack_idx_torch.numpy()

            stop_pack_idx_torch = stop_pack_idx_torch.to(device)
            keep_pack_idx_torch = keep_pack_idx_torch.to(device)

            # Filter predictions for ensembles.
            latest_predictions = {
                k: v[keep_pack_idx] for k, v in eval_predictions.items()
            }
            latest_predictions_torch = {
                k: v[keep_pack_idx_torch] for k, v in eval_predictions_torch.items()
            }

            # Evaluate the best checkpoints of the stopped models.
            pack_nn.module_pack_load_state_dict(
                model, state.best_model_state_dicts, pack_idx=stop_pack_idx_torch
            )
            with pack_nn.module_pack_select(model, stop_pack_idx_torch):
                (
                    (final_metrics, final_predictions, final_predictions_torch),
                    eval_batch_size,
                ) = evaluate(
                    parts=lib.data.evaluation_parts(
                        task_labels, ['train', 'val', 'test']
                    ),
                    batch_size=eval_batch_size,
                    ranking_qids=None,
                )
            report['eval_batch_size'] = eval_batch_size

            # Assemble the new results.
            final_state.extend(
                ids=state.ids[stop_pack_idx],
                steps=state.best_steps[stop_pack_idx],
                predictions=final_predictions,
                predictions_torch=final_predictions_torch,
            )
            new_experiments = assemble_experiments(
                state=state,
                final_metrics=final_metrics,
                pack_idx=stop_pack_idx,
                time_elapsed=timer.elapsed(),
            )
            experiments.extend(new_experiments)
            if hyperparameter_sampler is not None:
                for experiment in new_experiments:
                    report['n_trials'] += hyperparameter_sampler.tell(
                        experiment['report']['id'],
                        get_experiment_val_score(experiment),
                    )

            # Remove the stopped models.
            pack_remove(model, optimizer, state, pack_idx=stop_pack_idx)

            # After the removal, all existing pack indices become invalid.
            del stop_pack_idx

            # Update the report.
            report['n_models'] += n_models_to_stop
            report['step'] = step
            if track_experiments:
                report['experiments'] = experiments
            if track_best_experiment:
                report['best'], _ = get_best_experiment(
                    report.get('best'), new_experiments
                )
            report['time'] = timer.elapsed()

            # Make the update visible.
            lib.experiment.dump_report(exp, report)

            del final_predictions, final_predictions_torch

        del eval_predictions, eval_predictions_torch

        # Update online ensembles.
        if online_ensembles is None:
            first_online_ensemble_improved = False
        else:
            ensemble_reports, first_online_ensemble_improved = update_online_ensembles(
                online_ensembles,
                task_labels=task_labels,
                task_type=task_type,
                task_score=task_score,
                ranking_qids=None,
                step=step,
                timer=timer,
                running_ids=state.ids,
                running_steps=state.steps,
                running_best_predictions=state.best_predictions,
                running_best_predictions_torch=state.best_predictions_torch,
                finished_ids=final_state.ids,
                finished_steps=final_state.steps,
                finished_predictions=final_state.predictions,
                finished_predictions_torch=final_state.predictions_torch,
                running_latest_predictions=latest_predictions,
                running_latest_predictions_torch=latest_predictions_torch,
            )

            assert online_ensemble_history is not None
            for ensemble_name, ensemble_report in ensemble_reports.items():
                report['online_ensembles'][ensemble_name]['report'] = ensemble_report
                online_ensemble_history.setdefault(ensemble_name, []).append(
                    deepcopy(ensemble_report)
                )
                if config.track_online_ensemble_history:
                    report['online_ensembles'][ensemble_name]['report']['history'] = (
                        online_ensemble_history[ensemble_name]
                    )
                del ensemble_name, ensemble_report

        del latest_predictions, latest_predictions_torch

        # Compute statistics.
        training_throughput = math.trunc(epoch_size / epoch_training_duration)
        total_training_throughput = training_throughput * pack_size_before_stopping
        epoch_mean_loss = statistics.fmean(
            torch.stack(batch_losses).tolist(), batch_sizes
        )
        if track_experiments:
            mean_scores, mean_scores_improved = _get_mean_scores(
                mean_scores, experiments, state, eval_parts
            )
            del mean_scores_improved  # Currently unused.
        if track_best_experiment:
            best_scores, best_scores_improved = _get_best_scores(
                report.get('best'), best_scores, state
            )
        if online_ensembles is not None:
            # Here, we rely on the ordered nature of Python dicts.
            first_online_ensemble_experiment = next(
                iter(report['online_ensembles'].values())
            )
            first_online_ensemble_scores = {
                part: part_metrics['score']
                for part, part_metrics in (
                    first_online_ensemble_experiment['report']
                    .get('metrics', {})
                    .items()
                )
            }

        epochs_numlog.append(
            {'step': step, 'time': timer.elapsed(), 'loss': epoch_mean_loss}
        )

        # Print metrics.
        mean_scores_message = (
            None
            if mean_scores is None
            else ' '.join(
                f'[{part[0]}] {score:.3f}' for part, score in mean_scores.items()
            )
        )
        best_scores_message = (
            None
            if best_scores is None
            else ' '.join(
                f'[{part[0]}*] {score:.3f}' for part, score in best_scores.items()
            )
        )
        first_online_ensemble_scores_message = (
            None
            if first_online_ensemble_scores is None or not first_online_ensemble_scores
            else ' '.join(
                f'[{part[0]}$] {score:.3f}'
                for part, score in first_online_ensemble_scores.items()
            )
        )
        print(
            f'{"$" if first_online_ensemble_improved else " "}'
            f'{"*" if best_scores_improved else " "}'
            f' [E] {step // epoch_size:<3}'
            f' [T] {timer.elapsed():.0f}s'
            f' [L] {epoch_mean_loss / pack_size_before_stopping:.3f}'
            f' [M] {report["n_models"]:<3}'
            f'{"" if mean_scores_message is None else f" {mean_scores_message}"}'
            f'{"" if best_scores_message is None else f" {best_scores_message}"}'
            f'{"" if first_online_ensemble_scores_message is None else f" {first_online_ensemble_scores_message}"}'  # noqa: E501
            f' [it/s] {training_throughput:<3} | {total_training_throughput:<5}'
        )

        # >>> Validation
        pack_validate(model, optimizer, state)

    pack_validate(model, optimizer, state)

    # Ranking metrics are reporting-only; compute them once for each final ensemble.
    if ranking_qids is not None and online_ensembles is not None:
        for name, ensemble in online_ensembles.items():
            if ensemble.predictions:
                report['online_ensembles'][name]['report']['metrics'] = (
                    _calculate_ensemble_metrics(
                        {
                            part: pack_ensemble_utils.compute_ensemble_prediction(
                                prediction, ensemble.weights, prediction_type
                            )
                            for part, prediction in ensemble.predictions.items()
                        },
                        task_labels=task_labels,
                        task_type=task_type,
                        task_score=task_score,
                        ranking_qids=ranking_qids,
                        prediction_type=prediction_type,
                    )
                )
    report['time'] = timer.elapsed()

    # >>> Main artifacts

    if config.save_checkpoint:
        torch.save(
            {
                'report': report,
                'step': step,
                'random_state': lib.utils.torch.get_random_state(),
                'batch_generator': batch_generator.get_state(),
                'hyperparameter_sampler': hyperparameter_sampler,
                'timer': timer,
            },
            exp / 'checkpoint.pt',
        )

    # NOTE
    # The order of values in pack-related artifacts follows the order of
    # finishing, NOT the order of IDs.

    # Patch experiments to include all configs (even for unfinished models).
    experiments_with_all_configs = []
    for model_id in range(config.n_models):
        finished_experiment = next(
            (x for x in experiments if x['report']['id'] == model_id), None
        )

        if finished_experiment is not None:
            experiments_with_all_configs.append(finished_experiment)
        else:
            experiment = {}
            if all_configs is not None:
                experiment['config'] = all_configs[model_id]
            experiments_with_all_configs.append(experiment)

    exp.joinpath('experiments.json').write_text(
        json.dumps(experiments_with_all_configs, indent=4)
    )
    if config.save_final_predictions:
        np.savez(exp / 'predictions.npz', **final_state.predictions)

    if online_ensembles is not None:
        assert online_ensemble_history is not None
        exp.joinpath('online_ensemble_history.json').write_text(
            json.dumps(online_ensemble_history, indent=4)
        )

    # >>> Save numerical logs

    np.savez(
        exp / 'numlog.npz',
        **lib.utils.collections.flatten_dict(
            {
                'steps': pack_utils.numpy_stack(pack_utils.to_numpy(steps_numlog)),
                'epochs': pack_utils.numpy_stack(pack_utils.to_numpy(epochs_numlog)),
                'pack': {
                    'epochs': pack_utils.numpy_concatenate(pack_epochs_numlog),
                },
            }
        ),
    )

    # >>> Finish

    print(f'Report\n{lib.utils.misc.add_frame(lib.experiment.format_report(report))}\n')
    lib.experiment.finish(exp, main, report=report)
    return report


if __name__ == '__main__':
    lib.env.initialize()

    import cyclopts

    cyclopts.run(main, result_action='return_value')
