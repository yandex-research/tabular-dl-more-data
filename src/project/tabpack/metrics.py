import numpy as np
import sklearn.metrics

import project.ranking
from lib.data import TaskScore, TaskType
from lib.metrics import PredictionType

from .nn import BATCH_DIM, PACK_DIM

type Metrics = dict[str, np.ndarray]


def calculate_metrics_pack(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str | TaskType,
    prediction_type: str | PredictionType,
    score: TaskScore,
    ranking_qid: np.ndarray | None = None,
) -> Metrics:
    assert y_true.ndim == 1
    task_type = TaskType(task_type)
    prediction_type = PredictionType(prediction_type)

    ranking_y_true = y_true
    pack_size = y_pred.shape[PACK_DIM]
    y_true = np.repeat(np.expand_dims(y_true, PACK_DIM), pack_size, axis=PACK_DIM)

    if task_type == TaskType.REGRESSION:
        assert prediction_type == PredictionType.LABELS
        assert y_true.shape == y_pred.shape
        y_true_T = y_true.T
        y_pred_T = y_pred.T
        result = {
            'rmse': (
                sklearn.metrics.mean_squared_error(
                    y_true_T, y_pred_T, multioutput='raw_values'
                )
                ** 0.5
            ),
            'r2': sklearn.metrics.r2_score(
                y_true_T, y_pred_T, multioutput='raw_values'
            ),
        }

    elif task_type == TaskType.BINCLASS:
        assert prediction_type == PredictionType.PROBS
        assert y_true.shape == y_pred.shape
        # Always compute accuracy.
        result = {
            'accuracy': (
                (y_true == np.round(y_pred).astype(np.int64)).sum(BATCH_DIM)
                / y_true.shape[BATCH_DIM]
            ),
        }
        if score == TaskScore.AVERAGE_PRECISION:
            result['average-precision'] = np.array(
                [
                    sklearn.metrics.average_precision_score(ranking_y_true, prediction)
                    for prediction in y_pred
                ]
            )
        elif score == TaskScore.ROC_AUC:
            roc_auc_values = sklearn.metrics.roc_auc_score(
                y_true.T, y_pred.T, average=None
            )
            if isinstance(roc_auc_values, float):
                # This happens when the pack size equals 1.
                roc_auc_values = np.array([roc_auc_values])
            result['roc-auc'] = roc_auc_values
        elif score == TaskScore.CROSS_ENTROPY:
            result['cross-entropy'] = np.mean(
                -np.log(np.where(y_true, y_pred, 1 - y_pred)), axis=-1
            )
    else:
        assert task_type == TaskType.MULTICLASS
        assert prediction_type == PredictionType.PROBS
        assert y_true.shape == y_pred.shape[:2]
        # Always compute accuracy.
        result = {
            'accuracy': (
                (y_true == np.argmax(y_pred, axis=-1)).sum(BATCH_DIM)
                / y_true.shape[BATCH_DIM]
            )
        }
        if score == TaskScore.CROSS_ENTROPY:
            result['cross-entropy'] = np.mean(
                -np.log(np.take_along_axis(y_pred, y_true[..., None], axis=-1))[..., 0],
                axis=-1,
            )

    if ranking_qid is not None:
        if task_type != TaskType.REGRESSION:
            raise ValueError('Auxiliary ranking metrics require a regression task')
        ranking_metrics = [
            project.ranking.compute_metrics(
                y_true=ranking_y_true,
                y_score=model_prediction,
                qid=ranking_qid,
            )
            for model_prediction in y_pred
        ]
        for name in ranking_metrics[0]:
            result[name] = np.array([metrics[name] for metrics in ranking_metrics])

    result['score'] = (
        result[score.value] if score.higher_is_better else -result[score.value]
    )

    return result
