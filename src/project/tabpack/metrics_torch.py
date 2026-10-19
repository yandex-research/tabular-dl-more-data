from typing import Literal

import torch
import torch.nn as nn

from lib.data import TaskScore, TaskType
from lib.metrics import PredictionType

from .nn import BATCH_DIM

type MetricsTorch = dict[str, torch.Tensor]


def roc_auc_score(*, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute ROC-AUC for binary classification.

    NOTE
    The output of this function is equal to that of `sklearn.metric.roc_auc_score`
    up to the 1e-6 precision.

    **Shape**

    Assuming `N` is the number of objects and `*` is an arbitrary number of batch
    dimensions:

    * `y_true`: `(N,)`
    * `y_true`: `(*, N,)`
    * Output: `(*,)`
    """
    assert y_true.ndim == 1
    assert y_true.dtype == torch.int64
    # NOTE
    # The code actually supports any number of dimensions for y_pred,
    # but it was tested only for y_pred.ndim in (1, 2), hence the assert.
    assert y_pred.ndim in (1, 2), 'y_pred must have one or two dimensions'

    sorted_indices = torch.argsort(y_pred, dim=-1, descending=True)
    y_true_sorted = y_true[sorted_indices]

    n_positives = y_true.sum()
    n_negatives = len(y_true) - n_positives

    tpr = torch.cumsum(y_true_sorted, dim=-1) / n_positives
    fpr = torch.cumsum(1 - y_true_sorted, dim=-1) / n_negatives

    tpr = torch.cat(
        [torch.zeros((*tpr.shape[:-1], 1), dtype=tpr.dtype, device=tpr.device), tpr],
        dim=-1,
    )
    fpr = torch.cat(
        [torch.zeros((*fpr.shape[:-1], 1), dtype=fpr.dtype, device=fpr.device), fpr],
        dim=-1,
    )

    return (
        torch.trapz(tpr, fpr)
        if y_pred.ndim == 1
        else torch.stack(
            [torch.trapz(*x) for x in zip(tpr.flatten(0, -2), fpr.flatten(0, -2))]
        ).unflatten(0, y_pred.shape[:-1])
        if y_pred.shape[0] > 0
        else torch.tensor([], dtype=y_pred.dtype, device=y_pred.device)
    )


def average_precision_score(
    *, y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """Non-interpolated AP, grouping tied scores as sklearn does."""
    assert y_true.ndim == 1 and y_pred.ndim in (1, 2)
    assert y_pred.shape[-1] == len(y_true) and len(y_true) > 0
    scores, order = y_pred.sort(dim=-1, descending=True)
    positives = y_true[order]
    cumulative = positives.cumsum(dim=-1)
    n = len(y_true)
    positions = torch.arange(n, device=y_pred.device).expand_as(order)
    group_end = torch.cat(
        [
            scores[..., :-1] != scores[..., 1:],
            torch.ones_like(scores[..., -1:], dtype=torch.bool),
        ],
        dim=-1,
    )
    # For each row find the last index with the same score. Precision is
    # evaluated there, never within a tie group.
    ends = torch.where(group_end, positions, n)
    ends = ends.flip(-1).cummin(dim=-1).values.flip(-1)
    precision = cumulative.gather(-1, ends).to(torch.float64) / (ends + 1)
    return (precision * positives).sum(dim=-1) / y_true.sum().clamp_min(1)


def multiclass_cross_entropy(
    *,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Literal['none'],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Like `torch.nn.functional.binary_cross_entropy`, but for multiclass tasks."""
    assert reduction == 'none', 'For now, only reduction="none" is supported'
    assert y_true.shape == y_pred.shape[:-1]
    assert y_true.ndim + 1 == y_pred.ndim, (
        'For now, only class labels are supported as `y_true`'
    )
    assert y_true.dtype == torch.int64, f'y_true must have the {torch.int64} dtype'
    correct_class_probs = y_pred.gather(-1, y_true[..., None]).squeeze(-1)
    return -torch.log(correct_class_probs + eps)


def calculate_metrics_pack(
    *,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    task_type: str | TaskType,
    prediction_type: str | PredictionType,
    score: TaskScore,
) -> MetricsTorch:
    assert y_true.ndim == 1
    task_type = TaskType(task_type)
    prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type == PredictionType.LABELS
        mse = (y_true - y_pred).square_().mean(1)
        r2_denom = (y_true - y_true.mean()).square_().mean()
        result = {'rmse': mse.sqrt(), 'r2': 1 - mse / r2_denom}

    elif task_type == TaskType.BINCLASS:
        assert prediction_type == PredictionType.PROBS
        if score == TaskScore.ACCURACY:
            result = {
                'accuracy': ((y_true == y_pred.round()).float().mean(BATCH_DIM)),
            }
        elif score == TaskScore.AVERAGE_PRECISION:
            result = {
                'average-precision': average_precision_score(
                    y_true=y_true, y_pred=y_pred
                )
            }
        elif score == TaskScore.ROC_AUC:
            result = {'roc-auc': roc_auc_score(y_true=y_true, y_pred=y_pred)}
        elif score == TaskScore.CROSS_ENTROPY:
            result = {
                'cross-entropy': nn.functional.binary_cross_entropy(
                    input=y_pred,
                    target=y_true.to(y_pred.dtype).expand_as(y_pred),
                    reduction='none',
                ).mean(dim=-1)
            }
        elif score == TaskScore.RMSE:
            result = {
                'rmse': nn.functional.mse_loss(
                    input=y_pred,
                    target=y_true.to(y_pred.dtype).expand_as(y_pred),
                    reduction='none',
                ).mean(dim=-1)
            }
        else:
            raise ValueError(f'{score=} is not supported for the {task_type=}')

    else:
        assert task_type == TaskType.MULTICLASS
        assert prediction_type == PredictionType.PROBS
        if score == TaskScore.ACCURACY:
            result = {
                'accuracy': ((y_true == y_pred.argmax(-1)).float().mean(BATCH_DIM))
            }
        elif score == TaskScore.CROSS_ENTROPY:
            result = {
                'cross-entropy': multiclass_cross_entropy(
                    y_true=y_true[None].expand(len(y_pred), -1),
                    y_pred=y_pred,
                    reduction='none',
                ).mean(dim=-1)
            }
        elif score == TaskScore.RMSE:
            result = {
                'rmse': nn.functional.mse_loss(
                    input=y_pred,
                    target=(
                        nn.functional.one_hot(y_true).to(y_pred.dtype).expand_as(y_pred)
                    ),
                    reduction='none',
                ).mean(dim=(-2, -1))
            }
        else:
            raise ValueError(f'{score=} is not supported for the {task_type=}')

    result['score'] = (
        result[score.value] if score.higher_is_better else -result[score.value]
    )

    return result
