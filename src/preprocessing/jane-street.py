# >>> Jane Street Real-Time Market Data Forecasting
import logging
from typing import cast

import polars as pl
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from preprocessing.util import (
    default_train_sizes,
    get_reference_split_idx,
    make_iid_split,
    parse_args,
    prepare_data_dir,
    save_dataset_variants,
)

logger = logging.getLogger(__name__)


# symbol_id and time_id are ids, not magnitudes.
CAT_COLS = ['symbol_id', 'time_id']
TARGET_COL = 'responder_6'
RESPONDER_COLS = [f'responder_{i}' for i in range(9)]
# feature_61 only changes with date_id (roughly every ~20 dates), so its raw value
# is dropped in favor of a periodic encoding (see below).
DROPPED_FEATURE_COL = 'feature_61'
TIME_ID_PERIODS = [967, 483]
FEATURE_61_PERIOD = 20
# date_id/time_id are ordinally sorted trading dates/intervals with no calendar
# timestamp, so val/test cutoffs are chosen as the last N trading dates rather than
# fixed calendar dates. 100 dates is roughly the same order of magnitude as the ~6
# month forecasting horizon described by the competition.
VAL_TEST_N_DATES = 100


def main(
    raw_data_dir,
    *,
    output_dir=None,
    train_sizes=None,
    train_sampling='both',
    reference_data_dir=None,
):
    if train_sizes is None:
        train_sizes = default_train_sizes('jane-street')
    train_sizes = cast(list[str | int | None], train_sizes)
    # ================================================================
    # >> Load and split into num, cat
    # ================================================================

    logger.info('Preprocessing Jane Street market data forecasting dataset')
    tmp_data_path = prepare_data_dir(
        'jane-street',
        raw_data_dir,
        requirements=['train.parquet'],
        archive_names=['jane-street-real-time-market-data-forecasting.zip'],
    )
    train_parquet_path = tmp_data_path / 'train.parquet'

    # Use a fixed train pool so IID holdouts and nested sizes do not depend
    # on which train sizes were requested in this invocation.
    date_counts = cast(
        pl.DataFrame,
        (
            pl.scan_parquet(train_parquet_path, hive_partitioning=True)
            .group_by('date_id')
            .agg(pl.len().alias('n_rows'))
            .sort('date_id')
            .collect()
        ),
    )
    date_ids = date_counts['date_id'].to_list()

    test_start_date = date_ids[-VAL_TEST_N_DATES]
    val_start_date = date_ids[-2 * VAL_TEST_N_DATES]

    pre_val_dates = date_ids[: -2 * VAL_TEST_N_DATES]
    train_pool_start_date = pre_val_dates[0]
    logger.info(
        f'Loading all dates from {train_pool_start_date} for a fixed train pool'
    )
    # train.parquet ships as a directory hive-partitioned by partition_id; date_id
    # already encodes time, so the partition_id column itself is redundant.
    data = cast(
        pl.DataFrame,
        (
            pl.scan_parquet(train_parquet_path, hive_partitioning=True)
            .filter(pl.col('date_id').ge(train_pool_start_date))
            .collect()
        ),
    )
    if 'partition_id' in data.columns:
        data = data.drop('partition_id')
    # index_in_full is only used within this run (eval subsample + val/test
    # membership checks), so it's fine for it to be relative to this already
    # date-filtered table rather than the full unfiltered file.
    data = data.with_row_index(name='index_in_full')
    # Sort once up front (this is the only full-frame sort in the script) and reuse
    # it both for the lag group-by below and for the final row order.
    data = data.sort('date_id', 'time_id', 'symbol_id')

    # Lag-1 responder features: the last (max time_id) record of the previous
    # date_id, per symbol_id. Computed before the leaky same-date responder
    # columns are dropped below. Relies on `data` already being sorted by
    # ('date_id', 'time_id', 'symbol_id'), so each group's rows are time-ordered.
    last_per_date_symbol = (
        data.group_by(['date_id', 'symbol_id'])
        .agg([pl.col(c).last().alias(f'{c}_lag_1') for c in RESPONDER_COLS])
        .with_columns((pl.col('date_id') + 1).alias('date_id'))
    )
    data = data.join(last_per_date_symbol, on=['date_id', 'symbol_id'], how='left')
    lag_cols = [f'{c}_lag_1' for c in RESPONDER_COLS]

    # responder_0..8 (other than the target) are computed from the same time window
    # as the target and are not available as plain features at inference time
    # (only their lag-1 versions, computed above, are), so they would leak the label.
    leaky_cols = [c for c in RESPONDER_COLS if c != TARGET_COL]
    # The official metric is a sample-weighted R^2; this project's canonical format
    # uses plain regression/RMSE here, so the weight column is dropped rather
    # than misused as a plain feature.
    data = data.drop([*leaky_cols, 'weight'])

    # Captured before adding any engineered columns below, so the two new
    # feature_61_* periodic columns (which also start with "feature_") aren't
    # accidentally picked up twice.
    feature_cols = [
        c for c in data.columns if c.startswith('feature_') and c != DROPPED_FEATURE_COL
    ]

    # feature_61 only moves with date_id and isn't a useful continuous signal on its
    # own, but it cycles roughly every 20 dates, so its cycle is encoded instead of
    # the raw value. time_id has its own strong intraday periodicity.
    # polars' sin()/cos() upcast to Float64; cast back down to Float32 right away
    # so these columns don't quietly double their memory footprint at this scale.
    two_pi = 2 * np.pi
    periodic_cols = []
    for period in TIME_ID_PERIODS:
        sin_col, cos_col = f'time_id_sin_{period}', f'time_id_cos_{period}'
        data = data.with_columns(
            (pl.col('time_id') * (two_pi / period))
            .sin()
            .cast(pl.Float32)
            .alias(sin_col),
            (pl.col('time_id') * (two_pi / period))
            .cos()
            .cast(pl.Float32)
            .alias(cos_col),
        )
        periodic_cols += [sin_col, cos_col]
    data = data.with_columns(
        (pl.col(DROPPED_FEATURE_COL) * (two_pi / FEATURE_61_PERIOD))
        .sin()
        .cast(pl.Float32)
        .alias('feature_61_sin_20'),
        (pl.col(DROPPED_FEATURE_COL) * (two_pi / FEATURE_61_PERIOD))
        .cos()
        .cast(pl.Float32)
        .alias('feature_61_cos_20'),
    )
    periodic_cols += ['feature_61_sin_20', 'feature_61_cos_20']
    data = data.drop(DROPPED_FEATURE_COL)

    cat_cols = CAT_COLS
    num_cols = feature_cols + periodic_cols + lag_cols

    # `data` is already sorted by ('date_id', 'time_id', 'symbol_id') from the top
    # of the function; the left join above preserves that row order.
    #
    # No min_frequency here (unlike the other datasets' business-category cat_cols):
    # symbol_id/time_id are small fixed dictionaries where every value is meaningful,
    # not long-tail id-like categories with a handful of common values and a rare
    # tail. time_id in particular has ~968 nearly-uniformly-distributed values, each
    # appearing in ~0.1% of rows -- well under a 1% min_frequency threshold, which
    # would merge every single time_id value into one "infrequent" bucket.
    data_cat = data.select(cat_cols)
    data = data.drop(cat_cols)
    data = pl.concat(
        [
            data,
            pl.DataFrame(
                OrdinalEncoder().fit_transform(data_cat.to_numpy()),
                schema={c: pl.Int64 for c in data_cat.columns},
            ),
        ],
        how='horizontal',
    )

    Y_data = data.select(pl.col(TARGET_COL))
    X_meta_data = data.select('date_id', 'index_in_full')
    X_cat_data = data.select(cat_cols)
    X_num_data = data.select(num_cols)

    # ==================================================================================
    # >>> Default task split <<<
    # ==================================================================================

    # val_start_date/test_start_date were already computed from the cheap date_id
    # pre-scan above, before any row filtering.

    # Preserve the existing seed-0 draw from the fixed evaluation date window.
    # Only training partitioning changes when adding the IID protocol.
    eval_index_in_full = (
        data.filter(pl.col('date_id').ge(val_start_date))
        .sample(fraction=0.025, seed=0)['index_in_full']
        .to_list()
    )

    default_train_idx = (
        data.with_row_index()
        .filter(pl.col('date_id').lt(val_start_date))['index']
        .to_numpy()
    )

    default_val_idx = get_reference_split_idx(
        data,
        'jane-street',
        'val',
        reference_data_dir=reference_data_dir,
        index_column=1,
    )
    if default_val_idx is None:
        default_val_idx = (
            data.with_row_index()
            .filter(
                pl.col('date_id').ge(val_start_date)
                & pl.col('date_id').lt(test_start_date)
                & pl.col('index_in_full').is_in(eval_index_in_full)
            )['index']
            .to_numpy()
        )

    default_test_idx = get_reference_split_idx(
        data,
        'jane-street',
        'test',
        reference_data_dir=reference_data_dir,
        index_column=1,
    )
    if default_test_idx is None:
        default_test_idx = (
            data.with_row_index()
            .filter(
                pl.col('date_id').ge(test_start_date)
                & pl.col('index_in_full').is_in(eval_index_in_full)
            )['index']
            .to_numpy()
        )

    iid_split = make_iid_split(
        default_train_idx,
        val_size=len(default_val_idx),
        test_size=len(default_test_idx),
    )
    default_split = {
        'train_idx': iid_split['train_idx'],
        'val_idx': default_val_idx,
        'test_idx': default_test_idx,
    }

    data_parts = {
        n.rsplit('_', maxsplit=1)[0]: v
        for n, v in locals().items()
        if n.endswith('_data')
    }
    all_splits = {
        'time': default_split,
        'iid': iid_split,
    }

    logger.info('Writing data to disk')
    save_dataset_variants(
        name='jane-street',
        task_type='regression',
        data=data_parts,
        splits=all_splits,
        default_protocol='time',
        train_sizes=train_sizes,
        train_sampling=train_sampling,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    args = parse_args('jane-street')
    main(
        args.raw_data_dir,
        output_dir=args.output_dir,
        train_sizes=args.train_size,
        train_sampling=args.train_sampling,
        reference_data_dir=args.reference_data_dir,
    )
