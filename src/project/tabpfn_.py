"""Fit and evaluate TabPFN v3 with the standard experiment lifecycle.

Requires a TabPFN installation compatible with the supplied v3 checkpoints.
Checkpoints are read from the working directory, checkpoints/tabpfn-v3,
or data/tabpfn-v3;
override via model.model_path.
"""

import dataclasses
import importlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import lib.data
import lib.env
import lib.experiment
import lib.metrics
import lib.preprocessing
import lib.utils.dataclasses
import lib.utils.torch
from lib.types import KWArgs, PartKey

_LOGGER = logging.getLogger(__name__)
CKPT_PATHS = {
    'regression': 'tabpfn-v3-regressor-v3_default.ckpt',
    'binclass': 'tabpfn-v3-classifier-v3_20260417_binary.ckpt',
    'multiclass': 'tabpfn-v3-classifier-v3_20260417_multiclass.ckpt',
}


def _checkpoint_path(task: str) -> Path:
    filename = CKPT_PATHS[task]
    candidates = [
        Path(filename),
        Path('checkpoints/tabpfn-v3') / filename,
        Path('data/tabpfn-v3') / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        'TabPFN checkpoint not found. Place the pretrained weights in '
        'checkpoints/tabpfn-v3 or set model.model_path. Searched: '
        + ', '.join(str(path.resolve()) for path in candidates)
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int
    data: KWArgs
    model: KWArgs = dataclasses.field(default_factory=dict)
    eval_parts: list[PartKey] = dataclasses.field(default_factory=lambda: ['test'])
    concat_val: bool = False


def _make_x(data: dict[str, dict[PartKey, np.ndarray]], part: PartKey) -> np.ndarray:
    arrays = [data[key][part] for key in ('x_num', 'x_cat') if key in data]
    if not arrays:
        raise ValueError('TabPFN requires at least one feature')
    return np.column_stack(arrays)


def main(exp: str | Path, *, force: bool = False) -> lib.experiment.ReportDict | None:
    if not lib.experiment.try_start(exp, main, force=force):
        print(f'The experiment is already completed: {exp}')
        return None
    exp = lib.experiment.check(exp)
    config = lib.utils.dataclasses.from_dict(Config, lib.experiment.load_config(exp))
    report = lib.experiment.create_report(exp, main)
    print(f'{lib.experiment.format_config(config)}\n')
    if not config.eval_parts or len(set(config.eval_parts)) != len(config.eval_parts):
        raise ValueError('eval_parts must be nonempty and contain unique parts')
    if config.concat_val and 'val' in config.eval_parts:
        raise ValueError('Cannot evaluate validation data included in training')
    if config.data.get('cat_transform') is not None:
        raise ValueError('TabPFN expects ordinal categories: use cat_transform=None')
    if {'random_state', 'categorical_features_indices', 'device'} & config.model.keys():
        raise ValueError('Seed, categorical indices and device are set by the runner')

    # Import lazily so the rest of the project does not require TabPFN.
    tabpfn = importlib.import_module('tabpfn')
    lib.utils.torch.seed(config.seed)
    device = lib.utils.torch.get_device()
    _LOGGER.info('Device: %s', device)
    use_cuda = device.type == 'cuda'
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    info = lib.data.load_info(config.data['dataset_dir'])
    task_type = lib.data.TaskType(info['task']['type'])
    task_score = lib.data.TaskScore(info['task']['score'])
    data = lib.data.build_data(**config.data)
    # Keep original labels untouched for metrics.
    labels = data['y']
    eval_parts = lib.data.evaluation_parts(labels, config.eval_parts)
    if config.concat_val and 'val' in eval_parts:
        raise ValueError(
            'concat_val is incompatible with evaluation on all validations'
        )
    for part in eval_parts:
        if part not in labels:
            raise ValueError(f'Unknown evaluation part: {part}')
    transformer = None
    if task_type.is_regression:
        transformer = lib.preprocessing.StandardScaler1d()
        y = lib.preprocessing.fit_transform_dict(transformer, labels)
    else:
        y = labels
    x_train = _make_x(data, 'train')
    y_train = y['train']
    if config.concat_val:
        x_train = np.concatenate([x_train, _make_x(data, 'val')])
        y_train = np.concatenate([y_train, y['val']])
    n_num = data['x_num']['train'].shape[1] if 'x_num' in data else 0
    cat_indices = list(range(n_num, x_train.shape[1]))
    _LOGGER.info('Train rows: %s; features: %s', *x_train.shape)
    prediction_type = (
        lib.metrics.PredictionType.LABELS
        if task_type.is_regression
        else lib.metrics.PredictionType.PROBS
    )
    report['prediction_type'] = prediction_type.value
    model_kwargs = dict(config.model)
    model_kwargs.setdefault('ignore_pretraining_limits', True)
    if 'model_path' not in model_kwargs:
        model_kwargs['model_path'] = str(_checkpoint_path(task_type.value))
    _LOGGER.info('TabPFN checkpoint: %s', model_kwargs['model_path'])
    model_class = (
        tabpfn.TabPFNRegressor if task_type.is_regression else tabpfn.TabPFNClassifier
    )
    t0 = time.perf_counter()
    model: Any = model_class(
        random_state=config.seed,
        categorical_features_indices=cat_indices,
        device=device,
        **model_kwargs,
    )
    model.fit(x_train, y_train)
    predictions = {}
    for part in eval_parts:
        x = _make_x(data, part)
        if task_type.is_regression:
            assert transformer is not None
            predictions[part] = transformer.inverse_transform(model.predict(x))
        else:
            probs = model.predict_proba(x)
            predictions[part] = probs[:, 1] if task_type.is_binclass else probs
    if use_cuda:
        torch.cuda.synchronize(device)
    report['time'] = time.perf_counter() - t0

    metrics = {}
    for part, prediction in predictions.items():
        kwargs = {'y_true': labels[part], 'y_pred': prediction}
        if task_type.is_regression:
            metrics[part] = lib.metrics.compute_regression_metrics(**kwargs)
        elif task_type.is_binclass:
            metrics[part] = lib.metrics.compute_binclass_metrics(
                **kwargs, prediction_type=prediction_type
            )
        else:
            metrics[part] = lib.metrics.compute_multiclass_metrics(
                **kwargs, prediction_type=prediction_type
            )
        metrics[part]['score'] = (1 if task_score.higher_is_better else -1) * metrics[
            part
        ][task_score.value]
    report['metrics'] = metrics
    if use_cuda:
        report['gpu-cost'] = {
            'max-mem-alloc-mb': torch.cuda.max_memory_allocated(device) / 2**20,
            'max-mem-reserved-mb': torch.cuda.max_memory_reserved(device) / 2**20,
        }
    np.savez(exp / 'predictions.npz', allow_pickle=False, **predictions)
    lib.experiment.finish(exp, main, report=report)
    return report


if __name__ == '__main__':
    lib.env.initialize()

    import cyclopts

    cyclopts.run(main, result_action='return_value')
