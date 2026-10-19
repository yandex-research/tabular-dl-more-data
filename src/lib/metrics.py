import enum
import warnings
from typing import Any

import numpy as np
import scipy.special
import sklearn.metrics


@enum.unique
class PredictionType(enum.Enum):
    LABELS = 'labels'
    PROBS = 'probs'
    LOGITS = 'logits'


type MetricDict = dict[str, Any]


def _compute_labels_and_probs(
    prediction: np.ndarray, prediction_type: PredictionType, *, multiclass: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute labels and probabilities from raw predictions."""
    if prediction_type == PredictionType.LABELS:
        return prediction, None
    elif prediction_type == PredictionType.PROBS:
        probs = prediction
    elif prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.softmax(prediction, axis=1)
            if multiclass
            else scipy.special.expit(prediction)
        )
    else:
        raise ValueError(f'Unknown prediction type: {prediction_type}')

    labels = probs.argmax(axis=-1) if multiclass else np.round(probs)
    return labels.astype(np.int64), probs


def _log_loss(*args, **kwargs):
    def impl():
        return sklearn.metrics.log_loss(*args, **kwargs)

    if 'y_pred' in kwargs:
        # In Scikit-learn 1.9, `y_pred` was renamed to `y_proba` in
        # `sklearn.metrics.log_loss`. This module continues using `y_pred` to remain
        # compatible with projects using older Scikit-learn versions.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            return impl()
    else:
        return impl()


def compute_binclass_metrics(
    *, y_true: np.ndarray, y_pred: np.ndarray, prediction_type: PredictionType
) -> MetricDict:
    y_pred_labels, y_pred_probs = _compute_labels_and_probs(
        y_pred, prediction_type, multiclass=False
    )
    del y_pred

    metrics = {
        'accuracy': sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred_labels),
        'f1-score': sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred_labels),
    }
    if y_pred_probs is not None:
        metrics['average-precision'] = sklearn.metrics.average_precision_score(
            y_true=y_true, y_score=y_pred_probs
        )
        metrics['roc-auc'] = sklearn.metrics.roc_auc_score(
            y_true=y_true, y_score=y_pred_probs
        )
        metrics['cross-entropy'] = _log_loss(
            y_true=y_true, y_pred=y_pred_probs, labels=np.arange(2)
        )
    return {k: float(v) for k, v in metrics.items()}


def compute_multiclass_metrics(
    *, y_true: np.ndarray, y_pred: np.ndarray, prediction_type: PredictionType
) -> MetricDict:
    y_pred_labels, y_pred_probs = _compute_labels_and_probs(
        y_pred, prediction_type, multiclass=True
    )
    del y_pred

    metrics = {
        'accuracy': sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred_labels)
    }
    if y_pred_probs is not None:
        metrics['cross-entropy'] = _log_loss(
            y_true=y_true, y_pred=y_pred_probs, labels=np.arange(y_pred_probs.shape[-1])
        )
    return {k: float(v) for k, v in metrics.items()}


def compute_regression_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> MetricDict:
    metrics = {
        'rmse': sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5,
        'mae': sklearn.metrics.mean_absolute_error(y_true, y_pred),
        'r2': sklearn.metrics.r2_score(y_true, y_pred),
    }
    return {k: float(v) for k, v in metrics.items()}
