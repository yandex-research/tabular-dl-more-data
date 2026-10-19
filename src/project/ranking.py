"""Utilities for evaluating pointwise models on query-grouped datasets."""

from pathlib import Path

import numpy as np

import lib.data
from lib.types import PartKey


def load_qids(
    dataset_dir: str | Path, split_id: lib.data.SplitIDLike, **split_options
) -> dict[PartKey, np.ndarray] | None:
    """Load split query IDs declared by ``info.json["ranking"]["qid"]``."""
    ranking = lib.data.load_info(dataset_dir).get('ranking')
    if ranking is None:
        return None

    try:
        qid_spec = ranking['qid']
        path = Path(str(dataset_dir).removeprefix(':')) / qid_spec['array']
        column = int(qid_spec['column'])
    except (KeyError, TypeError, ValueError) as err:
        raise RuntimeError('Invalid ranking QID metadata in info.json') from err

    if not path.is_file() or path.suffix != '.npy':
        raise RuntimeError(f'Ranking QID array does not exist: {path}')
    array = np.load(path, mmap_mode='r')
    if array.ndim != 2 or not 0 <= column < array.shape[1]:
        raise RuntimeError(
            f'Invalid ranking QID column {column} for array with shape {array.shape}'
        )

    qids = lib.data.apply_split(
        array[:, column], lib.data.load_split(dataset_dir, split_id, **split_options)
    )
    assert isinstance(qids, dict)
    return qids


def group_sizes_from_qid(qid: np.ndarray) -> np.ndarray:
    qid = np.asarray(qid)
    if qid.ndim != 1:
        raise ValueError(f'qid must be one-dimensional, got shape={qid.shape}')
    if len(qid) == 0:
        return np.empty(0, dtype=np.int64)

    starts = np.r_[0, np.flatnonzero(qid[1:] != qid[:-1]) + 1]
    block_qids = qid[starts]
    if len(np.unique(block_qids)) != len(block_qids):
        raise ValueError('A qid occurs in more than one non-contiguous block')
    return np.diff(np.r_[starts, len(qid)])


def compute_metrics(
    *, y_true: np.ndarray, y_score: np.ndarray, qid: np.ndarray
) -> dict[str, float]:
    """Compute ranking metrics, averaging all queries equally."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    qid = np.asarray(qid)
    if y_true.ndim != 1 or y_score.ndim != 1 or qid.ndim != 1:
        raise ValueError('y_true, y_score and qid must be one-dimensional')
    if not (len(y_true) == len(y_score) == len(qid)):
        raise ValueError('y_true, y_score and qid must have equal lengths')
    if len(y_true) == 0:
        raise ValueError('Ranking metrics require at least one row')
    if not np.isin(np.unique(y_true), np.arange(5)).all():
        raise ValueError('Ranking labels must be integers in {0, 1, 2, 3, 4}')
    if not np.isfinite(y_score).all():
        raise ValueError('Ranking scores must be finite')

    groups = group_sizes_from_qid(qid)
    starts = np.r_[0, np.cumsum(groups[:-1])]
    sums = {
        name: 0.0 for name in ('p@1', 'p@5', 'p@10', 'ndcg@10', 'ndcg@20', 'mrr', 'map')
    }

    for start, size in zip(starts, groups):
        labels = y_true[start : start + size]
        scores = y_score[start : start + size]
        ranked = labels[np.argsort(-scores, kind='stable')]
        relevant = ranked >= 1

        for k in (1, 5, 10):
            sums[f'p@{k}'] += float(relevant[:k].sum()) / k

        relevant_ranks = np.flatnonzero(relevant)
        if len(relevant_ranks):
            sums['mrr'] += 1.0 / (int(relevant_ranks[0]) + 1)
            precision = np.cumsum(relevant) / np.arange(1, size + 1)
            sums['map'] += float(precision[relevant].mean())

        for k in (10, 20):
            n = min(k, size)
            discount = 1.0 / np.log2(np.arange(2, n + 2))
            gain = np.exp2(ranked[:n].astype(np.float64)) - 1.0
            ideal = np.exp2(np.sort(labels)[::-1][:n].astype(np.float64)) - 1.0
            idcg = float(ideal @ discount)
            sums[f'ndcg@{k}'] += 1.0 if idcg == 0.0 else float(gain @ discount) / idcg

    return {name: value / len(groups) for name, value in sums.items()}
