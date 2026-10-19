"""Istella with core query holdouts and row-IID evaluation on a shared train pool."""

import argparse
import gc
import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from preprocessing.util import (
    DATA_DIR,
    PROJECT_DIR,
    TRAIN_SIZE_LABELS,
    TrainSampler,
    make_iid_split,
    resolve_train_size,
    save_dataset_variants,
)
from project.ranking import group_sizes_from_qid

logger = logging.getLogger(__name__)


N_SOURCE_FEATURES = 220
SOURCE_PARTS = ('train', 'valid', 'test')
EXPECTED_ROWS = {'train': 8_349_810, 'valid': 2_062_424, 'test': 1_501_704}
EXPECTED_QUERIES = {'train': 44_249, 'valid': 11_005, 'test': 2_198}
EXPECTED_SENTINELS = {
    'train': {1: 3_130_413, 194: 1_503_361},
    'valid': {1: 766_614, 194: 369_318},
    'test': {1: 624_633, 194: 226_685},
}
# Exact full-split EDA found these columns constant in train, validation and test.
CONSTANT_FEATURE_VALUES = {
    **{index: 0.0 for index in range(82, 96)},
    93: 16.0,
    198: 0.0,
    203: 0.0,
    205: 0.0,
    206: 0.0,
    213: 0.0,
    219: 0.0,
}
BINARY_FEATURES = (
    21,
    24,
    35,
    38,
    49,
    52,
    63,
    66,
    77,
    80,
    105,
    108,
    119,
    122,
    161,
    164,
    175,
    178,
    197,
    204,
    214,
    215,
    217,
    218,
)
NUM_FEATURES = tuple(
    index
    for index in range(1, N_SOURCE_FEATURES + 1)
    if index not in CONSTANT_FEATURE_VALUES and index not in BINARY_FEATURES
)
assert len(CONSTANT_FEATURE_VALUES) == 20
assert len(BINARY_FEATURES) == 24
assert len(NUM_FEATURES) == 176


def extract_source(archive_path):
    """Extract only the three SVM tables, skipping docs.jsonl."""
    directory = PROJECT_DIR / 'preprocessing/tmp/istella22'
    directory.mkdir(parents=True, exist_ok=True)
    paths = {part: directory / f'{part}.svm' for part in SOURCE_PARTS}
    with zipfile.ZipFile(Path(archive_path).expanduser()) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        for part, destination in paths.items():
            if destination.exists():
                continue
            logger.info('Extracting %s', part)
            temporary = destination.with_suffix('.svm.partial')
            with (
                archive.open(members[f'{part}.svm']) as source,
                temporary.open('wb') as target,
            ):
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            temporary.replace(destination)
    return paths


def feature_expression(index):
    value = pl.col(f'f{index}').str.strip_prefix(f'{index}:').cast(pl.Float64)
    return (
        pl.when(value.abs() > float(np.finfo(np.float32).max))
        .then(None)
        .otherwise(value)
        .cast(pl.Float32)
        .alias(f'f{index}')
    )


def parse_part(
    path, part, *, expected_rows=None, expected_queries=None, expected_sentinels=None
):
    """Read dense SVM-Rank rows, clean features, and retain QIDs as metadata."""
    logger.info('Reading %s from %s', part, path)
    features = [f'f{index}' for index in range(1, N_SOURCE_FEATURES + 1)]
    columns = ['label', 'qid', *features, 'trailing']
    # The release is dense with a trailing space. Polars avoids a slow sparse parser.
    raw = pl.read_csv(
        path,
        separator=' ',
        has_header=False,
        new_columns=columns,
        quote_char=None,
        schema_overrides={name: pl.String for name in columns},
    )
    assert raw.width == len(columns)
    assert expected_rows is None or raw.height == expected_rows
    assert raw['trailing'].null_count() == raw.height
    assert (
        raw.select(
            [
                pl.col(name).str.starts_with(f'{index}:').fill_null(False).all()
                for index, name in enumerate(features, start=1)
            ]
        ).row(0)
        == (True,) * N_SOURCE_FEATURES
    )
    assert raw['qid'].str.contains(r'^qid:[0-9]+$').fill_null(False).all()

    data = raw.select(
        pl.col('label').cast(pl.Int8),
        pl.col('qid').str.strip_prefix('qid:').cast(pl.Int64),
        *[feature_expression(index) for index in range(1, N_SOURCE_FEATURES + 1)],
    )
    del raw
    gc.collect()

    assert data['label'].null_count() == data['qid'].null_count() == 0
    sentinels = {
        index: data[f'f{index}'].null_count()
        for index in range(1, N_SOURCE_FEATURES + 1)
        if data[f'f{index}'].null_count()
    }
    assert set(sentinels) <= {1, 194}
    assert expected_sentinels is None or sentinels == expected_sentinels
    for index, value in CONSTANT_FEATURE_VALUES.items():
        assert data[f'f{index}'].min() == data[f'f{index}'].max() == value

    labels = data['label'].to_numpy()
    assert np.isin(labels, np.arange(5)).all()
    qid = data['qid'].to_numpy()
    groups = group_sizes_from_qid(qid)
    assert expected_queries is None or len(groups) == expected_queries
    x_num = data.select([f'f{index}' for index in NUM_FEATURES])
    x_cat = data.select([f'f{index}' for index in BINARY_FEATURES])
    assert all(dtype == pl.Float32 for dtype in x_num.dtypes)
    assert x_cat.select(pl.all().is_in([0, 1]).all()).row(0) == (True,) * len(
        BINARY_FEATURES
    )

    stats = {
        'rows': len(labels),
        'queries': len(groups),
        'label_counts': np.bincount(labels.astype(np.int64), minlength=5).tolist(),
        'sentinel_counts': {str(k): v for k, v in sentinels.items()},
        'min_query_size': int(groups.min()),
        'max_query_size': int(groups.max()),
        'mean_query_size': float(groups.mean()),
    }
    return {
        'X_num': x_num,
        'X_cat': x_cat.cast(pl.Int64),
        'X_meta': data.select('qid'),
        'Y': data.select(pl.col('label').cast(pl.Float32)),
    }, stats


def sample_query_groups(groups, eligible, *, target_rows, seed):
    """Select whole queries close to, but not above, a fixed holdout budget."""
    assert target_rows > 0
    remaining = target_rows
    selected = []
    for query in np.random.default_rng(seed).permutation(eligible):
        if groups[query] <= remaining:
            selected.append(query)
            remaining -= int(groups[query])
    assert selected, 'The holdout budget must fit at least one query'
    return np.sort(selected)


def query_rows(groups, queries):
    boundaries = np.r_[0, np.cumsum(groups)]
    return np.concatenate(
        [np.arange(boundaries[q], boundaries[q + 1], dtype=np.int64) for q in queries]
    )


def make_splits(qids, *, holdout_size=600_000, seed=0):
    """Reserve core queries from test, then IID rows from pooled train/valid."""
    query_ids = {part: np.unique(qid) for part, qid in qids.items()}
    for i, left in enumerate(SOURCE_PARTS):
        for right in SOURCE_PARTS[i + 1 :]:
            assert not np.intersect1d(query_ids[left], query_ids[right]).size, (
                'Source partitions must have distinct query IDs'
            )

    pool_size = len(qids['train']) + len(qids['valid'])
    groups = group_sizes_from_qid(qids['test'])
    eligible = np.arange(len(groups))
    val_queries = sample_query_groups(
        groups, eligible, target_rows=holdout_size, seed=seed
    )
    test_queries = sample_query_groups(
        groups,
        np.setdiff1d(eligible, val_queries),
        target_rows=holdout_size,
        seed=seed + 1,
    )
    val_idx = pool_size + query_rows(groups, val_queries)
    test_idx = pool_size + query_rows(groups, test_queries)
    iid = make_iid_split(
        np.arange(pool_size),
        val_size=len(val_idx),
        test_size=len(test_idx),
        seed=seed + 2,
    )
    return {
        'core': {
            'train_idx': iid['train_idx'],
            'val_idx': val_idx,
            'test_idx': test_idx,
        },
        'iid': iid,
    }


class QueryTrainSampler:
    """Nested query prefixes, including only documents left in the train pool."""

    def __init__(self, train_idx, qid, *, seed=0):
        self.train_idx = np.asarray(train_idx)
        assert np.all(self.train_idx[1:] > self.train_idx[:-1])
        groups = group_sizes_from_qid(qid[self.train_idx])
        self.boundaries = np.r_[0, np.cumsum(groups)]
        self.order = np.random.default_rng(seed).permutation(len(groups))
        self.cumulative = np.cumsum(groups[self.order])

    def sample(self, target_rows):
        if target_rows is None or target_rows >= len(self.train_idx):
            return self.train_idx
        n = np.searchsorted(self.cumulative, target_rows, side='right')
        assert n > 0, 'The training budget must fit the first query in the permutation'
        queries = np.sort(self.order[:n])
        return np.concatenate(
            [
                self.train_idx[self.boundaries[q] : self.boundaries[q + 1]]
                for q in queries
            ]
        )


def make_train_variants(train_idx, qid, train_sizes, *, train_sampling='both', seed=0):
    assert train_sampling in ('random', 'query', 'both')
    policies = ('random', 'query') if train_sampling == 'both' else (train_sampling,)
    row_sampler = TrainSampler(train_idx, seed=seed)
    query_sampler = QueryTrainSampler(train_idx, qid, seed=seed)
    variants = {policy: {} for policy in policies}
    for value in train_sizes:
        _, count = resolve_train_size(value)
        label = 'full' if count is None else str(count)
        for policy in policies:
            variants[policy][label] = (
                row_sampler.sample(count, sampling='random')
                if policy == 'random'
                else query_sampler.sample(count)
            )
    return variants


def load_data(paths):
    """Keep cleaned partitions as Polars tables in their original source order."""
    parts, qids, stats = [], {}, {}
    for source in SOURCE_PARTS:
        part, stats[source] = parse_part(
            paths[source],
            source,
            expected_rows=EXPECTED_ROWS[source],
            expected_queries=EXPECTED_QUERIES[source],
            expected_sentinels=EXPECTED_SENTINELS[source],
        )
        qids[source] = part['X_meta']['qid'].to_numpy()
        parts.append(part)
    data = {
        key: pl.concat([part[key] for part in parts], rechunk=False) for key in parts[0]
    }
    return data, qids, stats


def main(
    raw_data_dir,
    *,
    output_dir=None,
    train_sizes=None,
    train_sampling='both',
):
    if train_sizes is None:
        train_sizes = TRAIN_SIZE_LABELS

    logger.info('Preprocessing Istella with core and IID evaluation')
    paths = extract_source(raw_data_dir)
    output_root = (
        DATA_DIR if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    data, qids, stats = load_data(paths)
    splits = make_splits(qids)
    train_variants = make_train_variants(
        splits['core']['train_idx'],
        data['X_meta']['qid'].to_numpy(),
        train_sizes,
        train_sampling=train_sampling,
    )

    logger.info('Writing data to disk')
    save_dataset_variants(
        name='istella',
        task_type='regression',
        data=data,
        splits=splits,
        default_protocol='core',
        train_sizes=train_sizes,
        train_variants=train_variants,
        output_dir=output_root,
        metadata={
            'recipe': 'istella-core-iid-v1',
            'source': {'kind': 'zip', 'input_name': Path(raw_data_dir).name},
            'parts': stats,
            'features': {
                'numeric_source_indices': list(NUM_FEATURES),
                'categorical_source_indices': list(BINARY_FEATURES),
            },
            'evaluation_sampling': {
                'core': {
                    'source': 'test',
                    'unit': 'query',
                    'target_rows': 600_000,
                    'val_seed': 0,
                    'test_seed': 1,
                },
                'iid': {
                    'source': ['train', 'valid'],
                    'unit': 'row',
                    'seed': 2,
                    'sizes': 'match realized core holdouts',
                },
            },
            'ranking': {
                'labels': [0, 1, 2, 3, 4],
                'qid': {
                    'array': 'arrays/x_meta.npy',
                    'column': 0,
                    'source_width': 6,
                },
                'queries_are_contiguous': True,
                'gain': '2**label - 1',
                'binary_relevance': 'label >= 1',
                'iid_candidate_lists': 'partial',
                'query_training_groups': 'remaining documents after IID removal',
            },
        },
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_data_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--train-size', nargs='+', default=list(TRAIN_SIZE_LABELS))
    parser.add_argument(
        '--train-sampling', choices=['random', 'query', 'both'], default='both'
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'
    )
    main(
        args.raw_data_dir,
        output_dir=args.output_dir,
        train_sizes=args.train_size,
        train_sampling=args.train_sampling,
    )
