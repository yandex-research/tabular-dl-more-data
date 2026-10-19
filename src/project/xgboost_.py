"""Train and evaluate XGBoost."""

import copy
import dataclasses
import hashlib
import json
import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from xgboost import Booster, DMatrix, QuantileDMatrix, train

import lib.data
import lib.env
import lib.experiment
import lib.metrics
import project.ranking
import lib.preprocessing
import lib.utils.dataclasses
import lib.utils.misc
import lib.utils.time
from lib.types import KWArgs, PartKey

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int
    data: KWArgs
    model: KWArgs
    # Kwargs for `xgboost.train` not covered by `model` (currently only `verbose`).
    fit: KWArgs = dataclasses.field(default_factory=dict)


class _XGBoostData(TypedDict):
    x: dict[PartKey, np.ndarray]
    y: dict[PartKey, np.ndarray]
    feature_types: None | list[str]
    dmats: dict[PartKey, DMatrix]
    regression_label_transformer: None | lib.preprocessing.StandardScaler1d


# QuantileDMatrices are expensive to build, so they are cached across the
# (potentially many) tuning trials that share the same data within one process.
_DMAT_CACHE: dict[str, _XGBoostData] = {}


def _check_config(config: Config) -> None:
    assert 'random_state' not in config.model, (
        'XGBoost uses `seed`/`config.seed`, not `random_state`'
    )
    # XGBoost consumes categorical features as ordinal codes (`enable_categorical`),
    # so the data must not be one-hot encoded.
    assert config.data.get('cat_transform') is None, (
        'XGBoost expects ordinal-encoded categorical features (cat_transform=None)'
    )
    assert 'early_stopping_rounds' in config.model, (
        'XGBoost does not automatically use the best model, '
        'so early stopping must be used'
    )
    if (
        config.model.get('tree_method') == 'gpu_hist'
        or config.model.get('device') == 'cuda'
    ):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')
    else:
        assert not os.environ.get('CUDA_VISIBLE_DEVICES')


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


def _make_x(
    data: dict[str, dict[PartKey, np.ndarray]],
) -> tuple[dict[PartKey, np.ndarray], None | list[str]]:
    """Merge all feature groups into matrices accepted by XGBoost.

    `lib.data.build_data` folds binary features into `x_cat`, so only `x_num`
    and `x_cat` may be present here.
    """
    parts = list(data['y'].keys())
    x: dict[PartKey, list[np.ndarray]] = {part: [] for part in parts}
    feature_types: list[str] = []

    if 'x_num' in data:
        for part in x:
            x[part].append(data['x_num'][part])
        feature_types.extend(['float'] * data['x_num']['train'].shape[1])

    if 'x_cat' in data:
        for part in x:
            x[part].append(data['x_cat'][part].astype('int64'))
        feature_types.extend(['c'] * data['x_cat']['train'].shape[1])

    assert feature_types
    return (
        {
            part: cast(
                np.ndarray,
                np.column_stack(part_arrays).astype(np.float32, copy=False),
            )
            for part, part_arrays in x.items()
        },
        feature_types if 'x_cat' in data else None,
    )


def _make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_make_jsonable(x) for x in value]
    return value


def _make_cache_key(config: Config) -> str:
    # QuantileDMatrix depends on the data and on quantization parameters.
    # Other hyperparameters (depth, lr, etc.) can change without rebuilding it.
    data_config = copy.deepcopy(config.data)
    if 'dataset_dir' in data_config:
        data_config['dataset_dir'] = str(Path(data_config['dataset_dir']).resolve())
    key_payload = {
        'data': data_config,
        'max_bin': config.model.get('max_bin'),
    }
    payload = json.dumps(_make_jsonable(key_payload), sort_keys=True)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def _make_dmats(
    x: dict[PartKey, np.ndarray],
    y: dict[PartKey, np.ndarray],
    feature_types: None | list[str],
    *,
    max_bin: None | int,
    nthread: None | int,
) -> dict[PartKey, DMatrix]:
    kwargs = {
        'feature_types': feature_types,
        'enable_categorical': feature_types is not None,
        'max_bin': max_bin,
        'nthread': nthread,
    }
    dtrain = QuantileDMatrix(x['train'], label=y['train'], **kwargs)
    return {
        'train': dtrain,
        **{
            part: QuantileDMatrix(values, label=y[part], ref=dtrain, **kwargs)
            for part, values in x.items()
            if part != 'train'
        },
    }


def _get_xgboost_data(config: Config, task_type: lib.data.TaskType) -> _XGBoostData:
    cache_key = _make_cache_key(config)
    if cache_key in _DMAT_CACHE:
        _LOGGER.info(f'Using cached XGBoost DMatrices: {cache_key}')
        return _DMAT_CACHE[cache_key]

    _LOGGER.info(f'Preparing XGBoost DMatrices: {cache_key}')
    data = lib.data.build_data(**config.data)
    assert 'x_bin' not in data

    # The labels used for training are standardized for regression; metrics are
    # computed later against the original labels loaded separately in `main`.
    if task_type.is_regression:
        regression_label_transformer = lib.preprocessing.StandardScaler1d()
        y = lib.preprocessing.fit_transform_dict(
            regression_label_transformer, data['y']
        )
    else:
        regression_label_transformer = None
        y = data['y']

    x, feature_types = _make_x(data)
    dmats = _make_dmats(
        x,
        y,
        feature_types,
        max_bin=config.model.get('max_bin'),
        nthread=config.model.get('n_jobs'),
    )
    xgboost_data: _XGBoostData = {
        'x': x,
        'y': y,
        'feature_types': feature_types,
        'dmats': dmats,
        'regression_label_transformer': regression_label_transformer,
    }
    _DMAT_CACHE[cache_key] = xgboost_data
    return xgboost_data


def _average_precision_metric(prediction, dmatrix):
    import sklearn.metrics

    weights = dmatrix.get_weight()
    return 'average-precision', float(
        sklearn.metrics.average_precision_score(
            dmatrix.get_label(),
            prediction,
            sample_weight=weights if len(weights) else None,
        )
    )


def _make_train_kwargs(
    config: Config, task_type: lib.data.TaskType, task_score: lib.data.TaskScore
) -> tuple[KWArgs, int, int, Any]:
    params = copy.deepcopy(config.model)
    num_boost_round = int(params.pop('n_estimators'))
    early_stopping_rounds = int(params.pop('early_stopping_rounds'))

    params.pop('use_label_encoder', None)
    if 'n_jobs' in params:
        params['nthread'] = params.pop('n_jobs')
    params['seed'] = config.seed

    if task_type.is_regression:
        params.setdefault('objective', 'reg:squarederror')
        params.setdefault('eval_metric', 'rmse')
    elif task_type.is_multiclass:
        params.setdefault('objective', 'multi:softprob')
        params.setdefault('eval_metric', 'merror')
    else:
        params.setdefault('objective', 'binary:logistic')
        params.setdefault(
            'eval_metric',
            'auc' if task_score == lib.data.TaskScore.ROC_AUC else 'error',
        )

    fit_config = copy.deepcopy(config.fit)
    if task_score == lib.data.TaskScore.AVERAGE_PRECISION:
        params.pop('eval_metric', None)
        params['disable_default_eval_metric'] = True
    verbose_eval = fit_config.pop('verbose', True)
    if fit_config:
        raise ValueError(f'Unsupported XGBoost fit options: {sorted(fit_config)}')

    return params, num_boost_round, early_stopping_rounds, verbose_eval


def _get_best_iteration(model: Booster) -> None | int:
    try:
        return int(model.best_iteration)
    except AttributeError:
        return None


def _predict(
    model: Booster, dmats: dict[PartKey, DMatrix]
) -> dict[PartKey, np.ndarray]:
    best_iteration = _get_best_iteration(model)
    iteration_range = (0, best_iteration + 1) if best_iteration is not None else (0, 0)
    return {
        part: model.predict(dmat, iteration_range=iteration_range)
        for part, dmat in dmats.items()
    }


def _get_feature_importances(model: Booster, n_features: int) -> np.ndarray:
    result = np.zeros(n_features, dtype=np.float32)
    for name, value in model.get_score(importance_type='gain').items():
        if name.startswith('f') and name[1:].isdigit():
            idx = int(name[1:])
            if idx < n_features:
                result[idx] = value
    return result


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


def main(exp: str | Path, *, force: bool = False) -> lib.experiment.ReportDict | None:
    # >>> Start

    if not lib.experiment.try_start(exp, main, force=force):
        print(f'The experiment is already completed: {exp}')
        return None

    exp = lib.experiment.check(exp)
    config = lib.utils.dataclasses.from_dict(Config, lib.experiment.load_config(exp))
    report = lib.experiment.create_report(exp, main)
    print(f'{lib.experiment.format_config(config)}\n')

    # >>> Setup

    _check_config(config)
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))

    # >>> Data

    # The task labels are loaded separately from the rest of the data.
    # They are used for computing metrics, and thus must never be modified.
    task_labels, task_type, task_score, ranking_qids = _load_task(
        config.data['dataset_dir'],
        config.data.get('split_id', lib.data.DEFAULT_SPLIT_ID),
        **lib.data.split_options(config.data),
    )

    data = _get_xgboost_data(config, task_type)
    dmats = data['dmats']
    regression_label_transformer = data['regression_label_transformer']
    n_features = data['x']['train'].shape[1]

    prediction_type = (
        lib.metrics.PredictionType.LABELS
        if task_type.is_regression
        else lib.metrics.PredictionType.PROBS
    )
    report['prediction_type'] = prediction_type.value

    # >>> Training

    params, num_boost_round, early_stopping_rounds, verbose_eval = _make_train_kwargs(
        config, task_type, task_score
    )
    if task_type.is_multiclass:
        params.setdefault(
            'num_class',
            len(np.unique(np.concatenate(list(task_labels.values())))),
        )

    print('Training...')
    timer = lib.utils.time.Timer()
    timer.run()
    model = train(
        params,
        dmats['train'],
        num_boost_round=num_boost_round,
        evals=[
            (dmat, part) for part, dmat in dmats.items() if part not in ('train', 'val')
        ]
        + [(dmats['val'], 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        **(
            {'custom_metric': _average_precision_metric, 'maximize': True}
            if task_score == lib.data.TaskScore.AVERAGE_PRECISION
            else {}
        ),
    )
    timer.pause()
    report['time'] = timer.elapsed()
    report['best_iteration'] = _get_best_iteration(model)

    # >>> Evaluation

    model.save_model(str(exp / 'model.ubj'))
    np.save(
        exp / 'feature_importances.npy',
        _get_feature_importances(model, n_features),
    )

    predictions = _predict(model, dmats)
    if regression_label_transformer is not None:
        predictions = {
            part: regression_label_transformer.inverse_transform(prediction)
            for part, prediction in predictions.items()
        }

    compute_metrics = _make_compute_metrics(task_type, prediction_type=prediction_type)
    metrics = {}
    for part, y_pred in predictions.items():
        metrics[part] = compute_metrics(y_true=task_labels[part], y_pred=y_pred)
        if ranking_qids is not None:
            metrics[part].update(
                project.ranking.compute_metrics(
                    y_true=task_labels[part],
                    y_score=y_pred,
                    qid=ranking_qids[part],
                )
            )
        # Auxiliary ranking metrics never change the pointwise task score used for
        # early stopping and hyperparameter tuning.
        metrics[part]['score'] = (1 if task_score.higher_is_better else -1) * metrics[
            part
        ][task_score.value]
    report['metrics'] = metrics

    # >>> Finish

    print(f'Report\n{lib.utils.misc.add_frame(lib.experiment.format_report(report))}\n')
    lib.experiment.finish(exp, main, report=report)
    return report


if __name__ == '__main__':
    lib.env.initialize()

    import cyclopts

    cyclopts.run(main, result_action='return_value')
