"""Fit and evaluate Kumo Tabular with cached, batched inference."""

import dataclasses
import importlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

import lib.data
import lib.env
import lib.experiment
import lib.metrics
import lib.utils.dataclasses
import lib.utils.torch
from project.kumo_weights import verify as verify_weights
from lib.types import KWArgs, PartKey

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int
    data: KWArgs
    model: KWArgs = dataclasses.field(default_factory=dict)
    eval_parts: list[PartKey] = dataclasses.field(default_factory=lambda: ['test'])
    concat_val: bool = False
    num_estimators: int = 8
    query_batch_size: int = 1024
    use_amp: bool = True


def _make_x(sdm, data, parts, device, rows=slice(None)):
    """Retain category values and semantic types, even for separate query batches."""
    columns, blocks = {}, {}
    for key, stype in (('x_num', 'numerical'), ('x_cat', 'categorical')):
        if key not in data:
            continue
        arrays = [data[key][part][rows] for part in parts]
        array = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
        if not array.shape[1]:
            continue
        tensor = torch.as_tensor(
            array,
            dtype=torch.float32 if key == 'x_num' else torch.int64,
            device=device,
        )
        blocks[stype] = (
            tensor if key == 'x_num' else sdm.CategoricalTensor.from_tensor(tensor)
        )
        columns[stype] = [f'{key}_{i}' for i in range(array.shape[1])]
    if not blocks:
        raise ValueError('Kumo requires at least one feature')
    return sdm.TableTensor(columns=columns, **blocks)


def _prediction(out, task_type, classes):
    if task_type.is_regression:
        # Uniformly spaced quantiles approximate the predictive mean.
        prediction = out.numerical.float().mean(dim=-1)
    else:
        # SDM names probability columns by category value, not class position.
        probs = torch.stack(
            [out[str(int(c))].numerical.squeeze(-1) for c in classes], dim=-1
        )
        prediction = probs[:, 1] if task_type.is_binclass else probs
    return prediction.detach().float().cpu().numpy()


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
    if config.concat_val and any(
        p == 'val' or p.startswith('val_') for p in config.eval_parts
    ):
        raise ValueError('Cannot evaluate validation data included in training')
    for name in ('num_estimators', 'query_batch_size'):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if any(config.data.get(k) is not None for k in ('num_transform', 'cat_transform')):
        raise ValueError('Kumo requires num_transform=None and cat_transform=None')
    if {'task', 'device'} & config.model.keys():
        raise ValueError('Task and device are set by the runner')
    if config.model.get('pretrained') is False:
        raise ValueError('Kumo evaluation requires pretrained weights')

    weights_dir = Path(os.environ.get('KUMO_WEIGHTS_DIR', 'data/kumo-tabular'))
    if weights_dir.exists() or 'KUMO_WEIGHTS_DIR' in os.environ:
        # Full checksums are checked once before the queue. Each seed checks the
        # cache layout before enabling offline inference.
        verify_weights(weights_dir, check_hashes=False)
        os.environ['HF_HUB_CACHE'] = str((weights_dir / 'hub').resolve())
        os.environ['HF_HUB_OFFLINE'] = '1'

    sdm = importlib.import_module('sdm')

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
    labels = data['y']
    eval_parts = lib.data.evaluation_parts(labels, config.eval_parts)
    if config.concat_val:
        # Independent splits expand the requested parts to every holdout.
        # Validation is training data here, including overlapping protocols.
        eval_parts = [
            part for part in eval_parts if part != 'val' and not part.startswith('val_')
        ]
    if not eval_parts or any(not len(labels[p]) for p in eval_parts):
        raise ValueError('Evaluation parts must be nonempty')
    report['eval_parts'] = eval_parts
    train_parts = ['train', 'val'] if config.concat_val else ['train']
    y_train = np.concatenate([labels[p] for p in train_parts])
    classes = None
    if not task_type.is_regression:
        classes = np.unique(y_train)
        if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError('Classification requires contiguous train classes from 0')
        if task_type.is_binclass and len(classes) != 2:
            raise ValueError('Binary classification requires two train classes')
        if any(not np.isin(labels[p], classes).all() for p in eval_parts):
            raise ValueError('Evaluation contains classes absent from the context')
    prediction_type = (
        lib.metrics.PredictionType.LABELS
        if task_type.is_regression
        else lib.metrics.PredictionType.PROBS
    )
    report['prediction_type'] = prediction_type.value
    t0 = time.perf_counter()
    model = sdm.models.KumoTabular(
        task='regression' if task_type.is_regression else 'classification',
        device=device,
        **config.model,
    )
    model.eval()
    predictions = {}
    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.use_amp and use_cuda,
            ),
        ):
            x_train = _make_x(sdm, data, train_parts, device)
            # SDM scales targets internally and inverts scaling on predictions.
            target = sdm.TableTensor.from_tensor(
                torch.as_tensor(
                    y_train[:, None],
                    device=device,
                    dtype=torch.float32 if task_type.is_regression else torch.int64,
                ),
                columns=['target'],
            )
            _LOGGER.info('Fit: %s context rows', len(y_train))
            model.fit(x=x_train, y=target, num_estimators=config.num_estimators)
            del x_train, target
            for part in eval_parts:
                chunks = []
                _LOGGER.info('Predicting %s: %s rows', part, len(labels[part]))
                for start in range(0, len(labels[part]), config.query_batch_size):
                    x = _make_x(
                        sdm,
                        data,
                        [part],
                        device,
                        slice(start, start + config.query_batch_size),
                    )
                    out = model.predict(x)
                    chunks.append(_prediction(out, task_type, classes))
                    del x, out
                predictions[part] = np.concatenate(chunks)
    finally:
        model.clear()
    if use_cuda:
        torch.cuda.synchronize(device)
    report['time'] = time.perf_counter() - t0

    if task_type.is_regression:
        compute_metrics = lib.metrics.compute_regression_metrics
        metric_options = {}
    else:
        compute_metrics = (
            lib.metrics.compute_binclass_metrics
            if task_type.is_binclass
            else lib.metrics.compute_multiclass_metrics
        )
        metric_options = {'prediction_type': prediction_type}
    metrics = {}
    for part, prediction in predictions.items():
        metrics[part] = compute_metrics(
            y_true=labels[part], y_pred=prediction, **metric_options
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
