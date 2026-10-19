"""Train and evaluate LightGBM with native categorical features."""

import dataclasses
from functools import partial
from pathlib import Path

import lightgbm
import numpy as np

import lib.data
import lib.env
import lib.experiment
import lib.metrics
import lib.preprocessing
import lib.utils.dataclasses
import lib.utils.time
import project.ranking
from lib.types import KWArgs, PartKey


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int
    data: KWArgs
    model: KWArgs
    fit: KWArgs = dataclasses.field(default_factory=dict)


def _check_config(config: Config) -> None:
    if config.data.get('cat_transform') is not None:
        raise ValueError('LightGBM requires cat_transform=None (ordinal categories)')
    if 'random_state' in config.model:
        raise ValueError('Use config.seed instead of model.random_state')
    if config.model.get('stopping_rounds', 0) <= 0:
        raise ValueError('Set positive model.stopping_rounds')
    if config.model.get('boosting_type', 'gbdt') == 'dart':
        raise ValueError('DART does not support early stopping')
    reserved = {
        'X',
        'y',
        'eval_set',
        'eval_X',
        'eval_y',
        'categorical_feature',
        'eval_metric',
    }
    if reserved & config.fit.keys():
        raise ValueError(f'Fit options managed by the runner: {sorted(reserved)}')


def _make_x(data) -> tuple[dict[PartKey, np.ndarray], list[int]]:
    # build_data folds binary features into x_cat. Handle x_bin as well so
    # feature assembly also works for data containing all three groups.
    keys = [key for key in ('x_num', 'x_bin', 'x_cat') if key in data]
    n_noncat = sum(data[key]['train'].shape[1] for key in keys if key != 'x_cat')
    n_features = sum(data[key]['train'].shape[1] for key in keys)
    if not n_features:
        raise ValueError('Dataset has no nonconstant features')
    return (
        {
            part: np.column_stack([data[key][part] for key in keys])
            for part in data['y']
        },
        list(range(n_noncat, n_features)),
    )


def _eval_metric(task_type, task_score):
    return {
        lib.data.TaskScore.RMSE: 'rmse',
        lib.data.TaskScore.R2: 'rmse',
        lib.data.TaskScore.MAE: 'mae',
        lib.data.TaskScore.ROC_AUC: 'auc',
        lib.data.TaskScore.AVERAGE_PRECISION: 'average_precision',
        lib.data.TaskScore.ACCURACY: (
            'multi_error' if task_type.is_multiclass else 'binary_error'
        ),
        lib.data.TaskScore.CROSS_ENTROPY: (
            'multi_logloss' if task_type.is_multiclass else 'binary_logloss'
        ),
    }[task_score]


def main(exp: str | Path, *, force: bool = False) -> lib.experiment.ReportDict | None:
    config = lib.utils.dataclasses.from_dict(Config, lib.experiment.load_config(exp))
    _check_config(config)
    if not lib.experiment.try_start(exp, main, force=force):
        print(f'The experiment is already completed: {exp}')
        return None
    exp = lib.experiment.check(exp)
    report = lib.experiment.create_report(exp, main)
    print(lib.experiment.format_config(config))

    info = lib.data.load_info(config.data['dataset_dir'])
    task_type = lib.data.TaskType(info['task']['type'])
    task_score = lib.data.TaskScore(info['task']['score'])
    data = lib.data.build_data(**config.data)
    task_labels = data['y']
    transformer = None
    if task_type.is_regression:
        transformer = lib.preprocessing.StandardScaler1d()
        labels = lib.preprocessing.fit_transform_dict(transformer, task_labels)
    else:
        labels = task_labels
    x, categorical_features = _make_x(data)
    ranking_qids = project.ranking.load_qids(
        config.data['dataset_dir'],
        config.data.get('split_id', lib.data.DEFAULT_SPLIT_ID),
        **lib.data.split_options(config.data),
    )

    params = dict(config.model)
    stopping_rounds = params.pop('stopping_rounds')
    # Disable the objective's default metric to select by the task metric only.
    params['metric'] = 'None'
    model = (
        lightgbm.LGBMRegressor if task_type.is_regression else lightgbm.LGBMClassifier
    )(**params, random_state=config.seed)
    fit_kwargs = dict(config.fit)
    fit_kwargs['callbacks'] = [
        *fit_kwargs.get('callbacks', []),
        lightgbm.early_stopping(stopping_rounds, first_metric_only=True),
    ]
    print('Training...')
    timer = lib.utils.time.Timer()
    timer.run()
    model.fit(
        x['train'],
        labels['train'],
        eval_X=x['val'],
        eval_y=labels['val'],
        categorical_feature=categorical_features,
        eval_metric=_eval_metric(task_type, task_score),
        **fit_kwargs,
    )
    timer.pause()
    report['time'] = timer.elapsed()
    report['best_iteration'] = model.best_iteration_
    np.save(exp / 'feature_importances.npy', model.feature_importances_)

    prediction_type = (
        lib.metrics.PredictionType.LABELS
        if task_type.is_regression
        else lib.metrics.PredictionType.PROBS
    )
    report['prediction_type'] = prediction_type.value
    compute_metrics = {
        lib.data.TaskType.REGRESSION: lib.metrics.compute_regression_metrics,
        lib.data.TaskType.BINCLASS: partial(
            lib.metrics.compute_binclass_metrics, prediction_type=prediction_type
        ),
        lib.data.TaskType.MULTICLASS: partial(
            lib.metrics.compute_multiclass_metrics, prediction_type=prediction_type
        ),
    }[task_type]
    predictions = {}
    metrics = {}
    for part, values in x.items():
        prediction = np.asarray(
            model.predict(values)
            if task_type.is_regression
            else model.predict_proba(values)
        )
        if task_type == lib.data.TaskType.BINCLASS:
            prediction = prediction[:, 1]
        if transformer is not None:
            prediction = transformer.inverse_transform(prediction)
        predictions[part] = prediction
        metrics[part] = compute_metrics(y_true=task_labels[part], y_pred=prediction)
        if ranking_qids is not None:
            metrics[part].update(
                project.ranking.compute_metrics(
                    y_true=task_labels[part], y_score=prediction, qid=ranking_qids[part]
                )
            )
        metrics[part]['score'] = (1 if task_score.higher_is_better else -1) * metrics[
            part
        ][task_score.value]
    report['metrics'] = metrics
    np.savez(exp / 'predictions.npz', allow_pickle=False, **predictions)
    print(lib.experiment.format_report(report))
    lib.experiment.finish(exp, main, report=report)
    return report


if __name__ == '__main__':
    lib.env.initialize()
    import cyclopts

    cyclopts.run(main, result_action='return_value')
