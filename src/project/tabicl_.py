"""Fit and evaluate TabICL with the standard experiment lifecycle."""

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int
    data: KWArgs
    model: KWArgs = dataclasses.field(default_factory=dict)
    eval_parts: list[PartKey] = dataclasses.field(default_factory=lambda: ['test'])
    concat_val: bool = False
    save_model: bool = False


def _make_x(
    data: dict[str, dict[PartKey, np.ndarray]],
    part: PartKey,
) -> np.ndarray:
    arrays = [data[key][part] for key in ('x_num', 'x_cat') if key in data]
    if not arrays:
        raise ValueError('TabICL requires at least one feature')
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
        raise ValueError('TabICL expects ordinal categories: use cat_transform=None')
    if {'random_state', 'device'} & config.model.keys():
        raise ValueError('Seed and device are set by the runner')

    # Import lazily so the rest of the project does not require TabICL.
    tabicl = importlib.import_module('tabicl')
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
    # Keep saved seed configs compatible while evaluating only test holdouts.
    eval_parts = [p for p in labels if p == 'test' or p.startswith('test_')]
    if not eval_parts:
        raise ValueError('No test holdouts available')
    report['eval_parts'] = eval_parts
    _LOGGER.info('Evaluation parts: %s', eval_parts)
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
    _LOGGER.info('Train rows: %s; features: %s', *x_train.shape)
    prediction_type = (
        lib.metrics.PredictionType.LABELS
        if task_type.is_regression
        else lib.metrics.PredictionType.PROBS
    )
    report['prediction_type'] = prediction_type.value
    model_kwargs = dict(config.model)
    model_kwargs.setdefault('use_amp', True)
    model_kwargs.setdefault('kv_cache', 'repr')
    model_kwargs.setdefault('batch_size', 1)
    model_kwargs.setdefault('offload_mode', 'cpu')
    report['inference'] = {
        'prediction_mode': 'per_test_split',
        'kv_cache': model_kwargs['kv_cache'],
        'batch_size': model_kwargs['batch_size'],
        'offload_mode': model_kwargs['offload_mode'],
    }
    model_class = (
        tabicl.TabICLRegressor if task_type.is_regression else tabicl.TabICLClassifier
    )
    t0 = time.perf_counter()
    model: Any = model_class(
        random_state=config.seed,
        device=device,
        **model_kwargs,
    )
    _LOGGER.info('Fit with kv_cache=%s', model_kwargs['kv_cache'])
    model.fit(x_train, y_train)
    del x_train, y_train
    if config.save_model:
        _LOGGER.info('Saving fitted model and cache: %s', exp / 'model.pkl')
        model.save(
            exp / 'model.pkl',
            save_kv_cache=True,
            save_training_data=not bool(model_kwargs['kv_cache']),
            save_model_weights=False,
        )
    predictions = {}
    for part in eval_parts:
        _LOGGER.info('Predict %s: %s rows', part, len(labels[part]))
        x_eval = _make_x(data, part)
        if task_type.is_regression:
            assert transformer is not None
            prediction = transformer.inverse_transform(model.predict(x_eval))
        else:
            probs = model.predict_proba(x_eval)
            prediction = probs[:, 1] if task_type.is_binclass else probs
        predictions[part] = prediction
        del x_eval
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
