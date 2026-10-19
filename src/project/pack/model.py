"""Train equally shaped MLPs as one packed fixed-step ablation.

Unlike :mod:`project.tabpack.model`, this runner does not tune, resample models or
build online ensembles.  One config describes exactly one model shape and one batch
size.  Pack members are the Cartesian product of learning rates, weight decays and
seeds.

Every available validation/test split is evaluated. ``best_by_validation`` in the
aggregate and per-member reports stores independent validation-selected results;
the legacy ``metrics`` and ``best_steps`` fields remain selected by ``val``.
"""

import dataclasses
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange

import lib.data
import lib.env
import lib.experiment
import lib.metrics
import lib.preprocessing
import lib.utils.dataclasses
import lib.utils.time
import lib.utils.torch
from lib.types import DataKey, KWArgs, PartKey
from project.tabpack import model as tabpack_model
from project.tabpack import nn as pack_nn
from project.tabpack.metrics import Metrics, calculate_metrics_pack

from .sampler import PackedEpochSampler

_CHECKPOINT_NAME = 'checkpoint.pt'
_WORST_SCORE = -999999.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    data: KWArgs
    model: KWArgs
    optimizer: KWArgs
    batch_size: int

    learning_rates: tuple[float, ...]
    seeds: tuple[int, ...]
    n_steps: int
    eval_every_n_steps: int

    weight_decays: tuple[float, ...] | None = None
    eval_parts: tuple[PartKey, ...] = ('val',)
    eval_batch_size: int = 32768
    checkpoint_every_n_steps: int = 1000
    gradient_clipping_norm: float | None = None
    amp_dtype: Literal['bfloat16'] | None = None
    pin_memory: bool = False
    num_dtype: Literal['bfloat16', 'float16'] | None = None

    def members(self) -> list[dict[str, int | float]]:
        weight_decays = (
            self.weight_decays
            if self.weight_decays is not None
            else (float(self.optimizer.get('weight_decay', 0.0)),)
        )
        # Hyperparameters are outer axes so all seeds of one configuration are
        # adjacent in reports.
        return [
            {'lr': lr, 'weight_decay': weight_decay, 'seed': seed}
            for lr in self.learning_rates
            for weight_decay in weight_decays
            for seed in self.seeds
        ]


def _validate_config(config: Config) -> None:
    if set(config.model) != {'backbone'}:
        raise ValueError(
            'Packed ablations currently support only model.backbone (plain MLP)'
        )
    backbone = config.model['backbone']
    required_backbone_keys = {'n_blocks', 'd_block', 'dropout'}
    if not required_backbone_keys <= set(backbone):
        missing = ', '.join(sorted(required_backbone_keys - set(backbone)))
        raise ValueError(f'Missing model.backbone fields: {missing}')
    if float(backbone['dropout']) != 0.0:
        raise ValueError(
            'Packed ablations currently require dropout=0 so member seeds have '
            'standalone-equivalent RNG semantics'
        )
    if config.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if config.n_steps <= 0:
        raise ValueError('n_steps must be positive')
    if config.eval_every_n_steps <= 0:
        raise ValueError('eval_every_n_steps must be positive')
    if config.checkpoint_every_n_steps <= 0:
        raise ValueError('checkpoint_every_n_steps must be positive')
    if config.eval_batch_size <= 0:
        raise ValueError('eval_batch_size must be positive')
    if not config.learning_rates or any(x <= 0.0 for x in config.learning_rates):
        raise ValueError('learning_rates must contain positive values')
    if not config.seeds or any(x < 0 for x in config.seeds):
        raise ValueError('seeds must contain non-negative values')
    if len(set(config.learning_rates)) != len(config.learning_rates):
        raise ValueError('learning_rates must be unique')
    if config.weight_decays is not None:
        if not config.weight_decays or any(x < 0.0 for x in config.weight_decays):
            raise ValueError('weight_decays must contain non-negative values')
        if len(set(config.weight_decays)) != len(config.weight_decays):
            raise ValueError('weight_decays must be unique')
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError('seeds must be unique')
    if 'val' not in config.eval_parts:
        raise ValueError('eval_parts must contain val')
    if config.gradient_clipping_norm is not None:
        if config.gradient_clipping_norm <= 0.0:
            raise ValueError('gradient_clipping_norm must be positive')


def _make_model(
    *,
    config: Config,
    n_num_features: int,
    cat_cardinalities: list[int],
    n_classes: int | None,
    pack_size: int,
) -> tabpack_model.ModelPack:
    return tabpack_model.ModelPack(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        pack_size=pack_size,
        **config.model['backbone'],
    )


def _initialize_members(
    model: tabpack_model.ModelPack,
    *,
    seeds: list[int],
    make_single_model,
) -> None:
    """Initialize each member exactly as a separately seeded pack of size one."""
    if len(seeds) != model.pack_size:
        raise ValueError('The number of seeds must equal the pack size')

    with lib.utils.torch.preserve_random_state():
        for pack_index, seed in enumerate(seeds):
            lib.utils.torch.seed(seed)
            single_model = make_single_model()
            pack_nn.module_pack_load_state_dict(
                model,
                single_model.state_dict(),
                pack_idx=torch.tensor([pack_index]),
                state_dict_idx=torch.tensor([0]),
            )


def _normalize_optimizer_config(
    config: KWArgs,
    learning_rates: list[float],
    weight_decays: list[float],
) -> tuple[str, KWArgs]:
    config = deepcopy(config)
    optimizer_type = config.pop('type')
    # The LR axis is defined once, at Config.learning_rates.
    config.pop('lr', None)
    config['lr'] = learning_rates
    # Likewise, a top-level weight_decays axis overrides the scalar optimizer
    # default. Config.members() expands that default to a length-one axis for
    # backward compatibility with existing experiments.
    config.pop('weight_decay', None)
    config['weight_decay'] = weight_decays

    if optimizer_type in ('torch.optim.AdamW', 'AdamWPack'):
        optimizer_type = 'AdamWPack'
        beta1, beta2 = config.pop('betas', (0.9, 0.999))
        config.setdefault('beta1', beta1)
        config.setdefault('beta2', beta2)
        config.setdefault('eps', 1e-8)
        config.setdefault('shared_step', True)
    elif optimizer_type in (
        'vendor.muon.SingleDeviceMuonWithAuxAdam',
        'MuonAdamWPack',
    ):
        optimizer_type = 'MuonAdamWPack'
        beta1, beta2 = config.pop('betas', (0.9, 0.999))
        config.setdefault('beta1', beta1)
        config.setdefault('beta2', beta2)
        config.setdefault('eps', 1e-8)
        muon = config.pop('muon', {})
        if 'muon_lr' not in config:
            if 'lr' not in muon:
                raise ValueError('Muon requires optimizer.muon.lr or optimizer.muon_lr')
            config['muon_lr'] = muon.pop('lr')
        if muon:
            raise ValueError(f'Unsupported Muon options: {sorted(muon)}')
        config.setdefault('shared_step', True)
    else:
        raise ValueError(f'Unsupported packed optimizer: {optimizer_type}')

    return optimizer_type, config


def _make_optimizer(
    model: tabpack_model.ModelPack, config: Config
) -> torch.optim.Optimizer:
    optimizer_type, optimizer_kwargs = _normalize_optimizer_config(
        config.optimizer,
        [float(member['lr']) for member in config.members()],
        [float(member['weight_decay']) for member in config.members()],
    )
    return tabpack_model._make_optimizer(  # noqa: SLF001
        model, type=optimizer_type, **optimizer_kwargs
    )


def _move_optimizer_group_tensors_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Move per-member optimizer hyperparameters after a CPU checkpoint load."""
    for group in optimizer.param_groups:
        for key, value in group.items():
            if key != 'params' and isinstance(value, Tensor):
                group[key] = value.to(device)


def _clip_grad_norm_(model: nn.Module, max_norm: float) -> Tensor:
    """Clip every member independently and return its pre-clipping norm."""
    squared_norm: Tensor | None = None
    parameters = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not isinstance(parameter, pack_nn.ParameterPack):
            raise TypeError('Packed models must contain only ParameterPack parameters')
        parameters.append(parameter)
        value = parameter.grad.detach().float().flatten(1).square().sum(1)
        squared_norm = value if squared_norm is None else squared_norm + value

    if squared_norm is None:
        raise RuntimeError('No gradients to clip')
    norm = squared_norm.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for parameter in parameters:
        assert parameter.grad is not None
        parameter.grad.mul_(scale.view(-1, *((1,) * (parameter.ndim - 1))))
    return norm


def _update_best(
    *,
    step: int,
    metrics: dict[PartKey, Metrics],
    best_metrics: dict[PartKey, Metrics] | None,
    best_steps: np.ndarray,
    validation_part: PartKey = 'val',
) -> tuple[dict[PartKey, Metrics], np.ndarray]:
    if best_metrics is None:
        pack_size = len(metrics[validation_part]['score'])
        improved = np.ones(pack_size, dtype=bool)
        best_metrics = deepcopy(metrics)
    else:
        improved = (
            metrics[validation_part]['score'] > best_metrics[validation_part]['score']
        )
        for part, part_metrics in metrics.items():
            for key, values in part_metrics.items():
                best_metrics[part][key][improved] = values[improved]

    best_steps[improved] = step
    return best_metrics, improved


def _calculate_metrics_allowing_invalid(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: lib.data.TaskType,
    prediction_type: lib.metrics.PredictionType,
    task_score: lib.data.TaskScore,
) -> Metrics:
    pack_size = y_pred.shape[pack_nn.PACK_DIM]
    valid = np.isfinite(y_pred).reshape(pack_size, -1).all(1)
    if not valid.any():
        return {'score': np.full(pack_size, _WORST_SCORE)}

    valid_metrics = calculate_metrics_pack(
        y_true=y_true,
        y_pred=y_pred[valid],
        task_type=task_type,
        prediction_type=prediction_type,
        score=task_score,
    )
    metrics = {}
    for key, values in valid_metrics.items():
        output = np.full(pack_size, np.nan)
        output[valid] = values
        metrics[key] = output
    metrics['score'][~valid] = _WORST_SCORE
    return metrics


@lib.utils.torch.adjust_gpu_memory_usage('batch_size')
def _evaluate(
    *,
    apply_model,
    model: tabpack_model.ModelPack,
    data: dict[DataKey, dict[PartKey, Tensor]],
    task_labels: dict[PartKey, np.ndarray],
    task_type: lib.data.TaskType,
    task_score: lib.data.TaskScore,
    regression_label_transformer: lib.preprocessing.StandardScaler1d | None,
    prediction_type: lib.metrics.PredictionType,
    parts: tuple[PartKey, ...],
    batch_size: int,
    device: torch.device,
) -> dict[PartKey, Metrics]:
    model.eval()
    metrics = {}
    with torch.inference_mode():
        for part in parts:
            # Move every output chunk to CPU immediately.  The stock TabPack evaluator
            # keeps the full pack prediction tensor on the GPU, which is unnecessary
            # for ablations and can dominate memory on large validation sets.
            y_pred = torch.cat(
                [
                    apply_model(model, data, part=part, batch_idx=batch_idx).cpu()
                    for batch_idx in torch.arange(
                        len(task_labels[part]), device=device
                    ).split(batch_size)
                ],
                dim=pack_nn.BATCH_DIM,
            ).numpy()

            if task_type.is_regression:
                assert regression_label_transformer is not None
                y_pred = (
                    y_pred * regression_label_transformer.scale_
                    + regression_label_transformer.mean_
                )
            elif task_type.is_binclass:
                y_pred = torch.from_numpy(y_pred).sigmoid().numpy()
            else:
                assert task_type.is_multiclass
                y_pred = torch.from_numpy(y_pred).softmax(dim=-1).numpy()

            metrics[part] = _calculate_metrics_allowing_invalid(
                y_true=task_labels[part],
                y_pred=y_pred,
                task_type=task_type,
                prediction_type=prediction_type,
                task_score=task_score,
            )
    return metrics


def _metrics_to_json(metrics: dict[PartKey, Metrics]) -> dict[str, Any]:
    return {
        part: {key: values.tolist() for key, values in part_metrics.items()}
        for part, part_metrics in metrics.items()
    }


def _summarize_validations(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Select each member's best step independently for every validation split.

    Replay the checkpointed evaluation history so old checkpoints also retain
    secondary validation optima observed before resuming. All test metrics come
    from the selected step, never from a separate test-score maximization.
    """
    best_metrics: dict[PartKey, dict[PartKey, Metrics]] = {}
    best_steps: dict[PartKey, np.ndarray] = {}
    for record in history:
        metrics = {
            part: {key: np.asarray(values) for key, values in part_metrics.items()}
            for part, part_metrics in record['metrics'].items()
        }
        for part in metrics:
            if part != 'val' and not part.startswith('val_'):
                continue
            if part not in best_steps:
                best_steps[part] = np.full(
                    len(metrics[part]['score']), -1, dtype=np.int64
                )
            best_metrics[part], _ = _update_best(
                step=record['step'],
                metrics=metrics,
                best_metrics=best_metrics.get(part),
                best_steps=best_steps[part],
                validation_part=part,
            )
    return {
        part: {
            'best_steps': best_steps[part].tolist(),
            'metrics': _metrics_to_json(metrics),
        }
        for part, metrics in best_metrics.items()
    }


def _make_experiments(
    *,
    config: Config,
    best_steps: np.ndarray,
    metrics: dict[PartKey, Metrics],
    time_elapsed: float,
    best_by_validation: dict[str, Any],
) -> list[dict[str, Any]]:
    experiments = []
    for index, member in enumerate(config.members()):
        member_optimizer = deepcopy(config.optimizer)
        member_optimizer['lr'] = member['lr']
        member_optimizer['weight_decay'] = member['weight_decay']
        experiments.append(
            {
                'config': {
                    'seed': member['seed'],
                    'data': config.data,
                    'model': config.model,
                    'optimizer': member_optimizer,
                    'batch_size': config.batch_size,
                    'n_steps': config.n_steps,
                    'eval_every_n_steps': config.eval_every_n_steps,
                },
                'report': {
                    'id': index,
                    'best_step': int(best_steps[index]),
                    'metrics': {
                        part: {
                            key: float(values[index])
                            for key, values in part_metrics.items()
                        }
                        for part, part_metrics in metrics.items()
                    },
                    'best_by_validation': {
                        validation_part: {
                            'best_step': selected['best_steps'][index],
                            'metrics': {
                                part: {
                                    key: values[index]
                                    for key, values in part_metrics.items()
                                }
                                for part, part_metrics in selected['metrics'].items()
                            },
                        }
                        for validation_part, selected in best_by_validation.items()
                    },
                    # Packed wall time is shared and is not a standalone runtime.
                    'time': time_elapsed,
                },
            }
        )
    return experiments


def _dump_checkpoint(exp: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint_path = exp / _CHECKPOINT_NAME
    temporary_path = exp / f'{_CHECKPOINT_NAME}.tmp'
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def _load_checkpoint(exp: Path) -> dict[str, Any] | None:
    path = exp / _CHECKPOINT_NAME
    return (
        torch.load(path, map_location='cpu', weights_only=False)
        if path.exists()
        else None
    )


@lib.utils.torch.set_num_threads(1)
def main(
    exp: str | Path, *, force: bool = False, resume: bool = False
) -> lib.experiment.ReportDict | None:
    if not lib.experiment.try_start(exp, main, force=force, resume=resume):
        print(f'The experiment is already completed: {exp}')
        return None

    exp = lib.experiment.check(exp)
    config = lib.utils.dataclasses.from_dict(Config, lib.experiment.load_config(exp))
    _validate_config(config)
    members = config.members()
    member_seeds = [int(member['seed']) for member in members]
    pack_size = len(members)

    # The model initializers below restore the global state.  This seed controls only
    # shared setup and any future non-member-specific randomness.
    lib.utils.torch.seed(member_seeds[0])
    device = lib.utils.torch.get_device()

    # Data preprocessing is shared by the whole pack.
    data = lib.data.build_data(**config.data)
    assert 'x_bin' not in data
    task_labels, task_type, task_score, _ = tabpack_model._load_task(  # noqa: SLF001
        config.data['dataset_dir'],
        config.data.get('split_id', lib.data.DEFAULT_SPLIT_ID),
        **lib.data.split_options(config.data),
    )
    missing_parts = set(config.eval_parts) - task_labels.keys()
    if missing_parts:
        raise ValueError(f'Unknown evaluation splits: {sorted(missing_parts)}')
    evaluation_parts = tuple(lib.data.evaluation_parts(task_labels, config.eval_parts))
    if task_type.is_regression:
        regression_label_transformer = lib.preprocessing.StandardScaler1d()
        data['y'] = lib.preprocessing.fit_transform_dict(
            regression_label_transformer, data['y']
        )
    else:
        regression_label_transformer = None

    num_storage_dtype = (
        None if config.num_dtype is None else getattr(torch, config.num_dtype)
    )
    data_device = torch.device('cpu') if config.pin_memory else device

    def place(key: DataKey, array: Any) -> Tensor:
        tensor = torch.as_tensor(array)
        if key == 'x_num' and num_storage_dtype is not None:
            tensor = tensor.to(num_storage_dtype)
        return tensor.to(data_device)

    data = {
        key: {part: place(key, array) for part, array in part_data.items()}
        for key, part_data in data.items()
    }
    if config.pin_memory:
        for part_data in data.values():
            for part in part_data:
                part_data[part] = part_data[part].pin_memory()

    y_train = tabpack_model._make_Y_train(data, task_type).to(device)  # noqa: SLF001
    train_size = len(task_labels['train'])
    n_num_features = data['x_num']['train'].shape[-1] if 'x_num' in data else 0
    cat_cardinalities = (
        (
            torch.stack([part.max(dim=0).values for part in data['x_cat'].values()])
            .max(dim=0)
            .values
            + 1
        ).tolist()
        if 'x_cat' in data
        else []
    )
    n_classes = (
        len(np.unique(np.concat(list(task_labels.values()))))
        if task_type.is_binclass or task_type.is_multiclass
        else None
    )

    make_model = partial(
        _make_model,
        config=config,
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
    )
    model = make_model(pack_size=pack_size)
    _initialize_members(
        model,
        seeds=member_seeds,
        make_single_model=partial(make_model, pack_size=1),
    )
    model.to(device)
    optimizer = _make_optimizer(model, config)
    loss_fn = tabpack_model._make_loss_fn_pack(task_type)  # noqa: SLF001

    sampler = PackedEpochSampler(
        train_size=train_size,
        batch_size=config.batch_size,
        seeds=member_seeds,
        device=device,
    )
    prediction_type = (
        lib.metrics.PredictionType.LABELS
        if task_type.is_regression
        else lib.metrics.PredictionType.PROBS
    )
    autocast = (
        None
        if config.amp_dtype is None
        else torch.autocast(device.type, getattr(torch, config.amp_dtype))
    )
    apply_model = partial(tabpack_model.apply_model_impl, device=device)
    if autocast is not None:
        apply_model = autocast(apply_model)

    report = lib.experiment.create_report(exp, main)
    report.update(
        {
            'n_models': pack_size,
            'pack_size': pack_size,
            'n_parameters': sum(x.numel() for x in model.parameters()) // pack_size,
            'batch_size': config.batch_size,
            'epoch_size': sampler.epoch_size,
            'prediction_type': prediction_type.value,
        }
    )
    timer = lib.utils.time.Timer()
    step = 0
    eval_batch_size = config.eval_batch_size
    best_metrics: dict[PartKey, Metrics] | None = None
    best_steps = np.full(pack_size, -1, dtype=np.int64)
    eval_history: list[dict[str, Any]] = []
    loss_sum = torch.zeros(pack_size, device=device)
    loss_count = 0

    checkpoint = _load_checkpoint(exp) if resume else None
    if checkpoint is not None:
        report = checkpoint['report']
        timer = checkpoint['timer']
        step = checkpoint['step']
        eval_batch_size = checkpoint['eval_batch_size']
        best_metrics = checkpoint['best_metrics']
        best_steps = checkpoint['best_steps']
        eval_history = checkpoint['eval_history']
        if step < config.n_steps:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            _move_optimizer_group_tensors_to_device(optimizer, device)
            sampler.load_state_dict(checkpoint['sampler'])
            lib.utils.torch.set_random_state(checkpoint['random_state'])
            loss_sum = checkpoint['loss_sum'].to(device)
            loss_count = checkpoint['loss_count']
        print(f'Resuming {exp} from step {step}')

    def make_checkpoint() -> dict[str, Any]:
        report['step'] = step
        report['time'] = timer.elapsed()
        return {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'sampler': sampler.state_dict(),
            'random_state': lib.utils.torch.get_random_state(),
            'report': report,
            'timer': timer,
            'step': step,
            'eval_batch_size': eval_batch_size,
            'best_metrics': best_metrics,
            'best_steps': best_steps,
            'eval_history': eval_history,
            'loss_sum': loss_sum,
            'loss_count': loss_count,
        }

    timer.run()
    progress = trange(
        step,
        config.n_steps,
        initial=step,
        total=config.n_steps,
        desc=str(exp),
    )
    for _ in progress:
        model.train()
        batch_idx = sampler.next(shared=True)
        y_batch = y_train[batch_idx]
        if batch_idx.ndim == 1:
            # A single seed gives every member the same examples. Gather features
            # and encode categories once; ModelPack broadcasts the shared input.
            y_batch = y_batch.unsqueeze(0).expand(pack_size, -1)
        losses = loss_fn(
            apply_model(model, data, part='train', batch_idx=batch_idx),
            y_batch,
        )
        optimizer.zero_grad()
        losses.sum().backward()
        if config.gradient_clipping_norm is not None:
            _clip_grad_norm_(model, config.gradient_clipping_norm)
        optimizer.step()

        step += 1
        loss_sum += losses.detach()
        loss_count += 1

        should_evaluate = (
            step % config.eval_every_n_steps == 0 or step == config.n_steps
        )
        if should_evaluate:
            metrics, eval_batch_size = _evaluate(
                apply_model=apply_model,
                model=model,
                data=data,
                task_labels=task_labels,
                task_type=task_type,
                task_score=task_score,
                regression_label_transformer=regression_label_transformer,
                prediction_type=prediction_type,
                parts=evaluation_parts,
                batch_size=eval_batch_size,
                device=device,
            )
            best_metrics, improved = _update_best(
                step=step,
                metrics=metrics,
                best_metrics=best_metrics,
                best_steps=best_steps,
            )
            mean_loss = (loss_sum / loss_count).cpu().numpy()
            eval_history.append(
                {
                    'step': step,
                    'time': timer.elapsed(),
                    'loss': mean_loss.tolist(),
                    'metrics': _metrics_to_json(metrics),
                }
            )
            val_scores = metrics['val']['score']
            progress.set_postfix(
                val_best=float(np.nanmax(val_scores)),
                val_median=float(np.nanmedian(val_scores)),
                val_worst=float(np.nanmin(val_scores)),
                improved=int(improved.sum()),
            )
            loss_sum.zero_()
            loss_count = 0

        if step % config.checkpoint_every_n_steps == 0 or step == config.n_steps:
            checkpoint = make_checkpoint()
            lib.experiment.dump_report(exp, report)
            _dump_checkpoint(exp, checkpoint)

    assert best_metrics is not None

    report['step'] = step
    report['time'] = timer.elapsed()
    report['eval_batch_size'] = eval_batch_size
    report['best_steps'] = best_steps.tolist()
    # The periodic history already contains train loss and val/test metrics.  Keep
    # the metrics from each member's best validation step; no final prediction pass
    # or best-weight restoration is needed for this dynamics-only sweep.
    report['metrics'] = _metrics_to_json(best_metrics)
    report['best_by_validation'] = _summarize_validations(eval_history)
    report['evaluation_parts'] = list(evaluation_parts)
    report['experiments'] = _make_experiments(
        config=config,
        best_steps=best_steps,
        metrics=best_metrics,
        time_elapsed=timer.elapsed(),
        best_by_validation=report['best_by_validation'],
    )
    (exp / 'experiments.json').write_text(
        json.dumps(report['experiments'], indent=4) + '\n'
    )
    (exp / 'eval_history.json').write_text(json.dumps(eval_history, indent=4) + '\n')
    # Replace the resumable training checkpoint (model + optimizer) with a compact
    # completed-state checkpoint containing logs only.  This remains sufficient to
    # finish safely after interruption without retaining trained weights.
    _dump_checkpoint(
        exp,
        {
            'training_complete': True,
            'report': report,
            'timer': timer,
            'step': step,
            'eval_batch_size': eval_batch_size,
            'best_metrics': best_metrics,
            'best_steps': best_steps,
            'eval_history': eval_history,
        },
    )
    lib.experiment.finish(exp, main, report=report)
    return report


if __name__ == '__main__':
    lib.env.initialize()

    import cyclopts

    cyclopts.run(main, result_action='return_value')
