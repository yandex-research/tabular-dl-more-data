# Weather Forecasting
import logging
from datetime import datetime
from typing import cast

import polars as pl

from preprocessing.util import (
    default_train_sizes,
    make_time_iid_splits,
    parse_args,
    prepare_data_dir,
    save_dataset_variants,
)

logger = logging.getLogger(__name__)


def main(
    raw_data_dir,
    *,
    output_dir=None,
    train_sizes=None,
    train_sampling='both',
    reference_data_dir=None,
):
    if train_sizes is None:
        train_sizes = default_train_sizes('weather')
    train_sizes = cast(list[str | int | None], train_sizes)
    # ================================================================
    # >> Load and split into num, bin, cat
    # ================================================================

    logger.info('Preprocessing Weather data')
    tmp_data_path = prepare_data_dir(
        'weather',
        raw_data_dir,
        requirements=['weather.parquet'],
        archive_names=['tabred-weather.zip'],
    )

    data = pl.read_parquet(tmp_data_path / 'weather.parquet').with_row_index(
        name='index_in_full'
    )
    data = data.sort('fact_time')

    # Generate time features
    data = data.with_columns((pl.col('fact_time') * 10**6).cast(pl.Datetime))
    data = data.with_columns(
        pl.col('fact_time').dt.weekday().alias('day_of_week'),
        pl.col('fact_time').dt.day().alias('day_of_month'),
        pl.col('fact_time')
        .dt.time()
        .cast(pl.Duration)
        .dt.total_minutes()
        .alias('minute_of_day'),
        pl.col('fact_time')
        .dt.time()
        .cast(pl.Duration)
        .dt.total_hours()
        .alias('hour_of_day'),
        pl.col('fact_time').dt.month().alias('month'),
    )

    target_cols = ['fact_temperature']
    meta_cols = [
        'fact_time',
        'apply_time_rl',
        'fact_latitude',
        'fact_longitude',
        'fact_station_id',
        'index_in_full',
    ]

    bin_cols = [
        c for c in data.columns if 'available' in c and c not in target_cols + meta_cols
    ]
    num_cols = [
        c
        for c in data.columns
        if 'available' not in c and c not in target_cols + meta_cols
    ]

    Y_data = data.select(target_cols).cast(pl.Float32)
    X_meta_data = data.select(meta_cols)
    X_bin_data = data.select(bin_cols).cast(pl.Float32)
    X_num_data = data.select(num_cols).cast(pl.Float32)

    all_splits = make_time_iid_splits(
        data,
        dataset_name='weather',
        time_column='fact_time',
        val_start=datetime(2023, 6, 1),
        test_start=datetime(2023, 7, 1),
        reference_data_dir=reference_data_dir,
        reference_index_column=5,
    )

    data_parts = {
        'X_num': X_num_data,
        'X_meta': X_meta_data,
        'Y': Y_data,
        'X_bin': X_bin_data,
    }

    logger.info('Writing data to disk')
    save_dataset_variants(
        name='weather',
        task_type='regression',
        data=data_parts,
        splits=all_splits,
        default_protocol='time',
        train_sizes=train_sizes,
        train_sampling=train_sampling,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    args = parse_args('weather')
    main(
        args.raw_data_dir,
        output_dir=args.output_dir,
        train_sizes=args.train_size,
        train_sampling=args.train_sampling,
        reference_data_dir=args.reference_data_dir,
    )
