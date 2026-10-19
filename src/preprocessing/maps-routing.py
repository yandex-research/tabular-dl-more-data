# > Maps Routing task and dataset
import logging
from datetime import datetime
from typing import cast

import polars as pl
from sklearn.preprocessing import OrdinalEncoder

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
        train_sizes = default_train_sizes('maps-routing')
    train_sizes = cast(list[str | int | None], train_sizes)
    # ================================================================
    # >> Load and split into num, bin, cat
    # ================================================================

    logger.info('Preprocessing maps routing dataset')
    tmp_data_path = prepare_data_dir(
        'maps-routing',
        raw_data_dir,
        requirements=['maps_routing.parquet'],
        archive_names=['maps-routing.zip'],
    )
    # Store full_index to help with potential future experiments on large data
    data = pl.read_parquet(tmp_data_path / 'maps_routing.parquet').with_row_index(
        name='index_in_full'
    )

    cat_cols = [c for c in data.columns if c.startswith('cat')]
    num_cols = [
        c
        for c in data.columns
        if c.startswith('num') or c in ['day_of_week', 'minute_of_day', 'hour_of_day']
    ]

    data_cat = data.select(cat_cols)
    data = data.drop(cat_cols)
    data = pl.concat(
        [
            data,
            pl.DataFrame(
                OrdinalEncoder(min_frequency=1 / 100).fit_transform(
                    data_cat.to_numpy()
                ),
                schema={c: pl.Int64 for c in data_cat.columns},
            ),
        ],
        how='horizontal',
    )

    data = data.with_columns(pl.col('timestamp').cast(pl.Datetime))
    data = data.sort('timestamp')
    data = data.with_columns(
        pl.col('timestamp').dt.weekday().alias('day_of_week'),
        pl.col('timestamp')
        .dt.time()
        .cast(pl.Duration)
        .dt.total_minutes()
        .alias('minute_of_day'),
        pl.col('timestamp')
        .dt.time()
        .cast(pl.Duration)
        .dt.total_hours()
        .alias('hour_of_day'),
    )

    Y_data = data.select(pl.col('target_log_spkm'))
    X_meta_data = data.select('timestamp', 'index_in_full', 'track_length')
    X_cat_data = data.select(cat_cols)
    X_num_data = data.select(num_cols)

    all_splits = make_time_iid_splits(
        data,
        dataset_name='maps-routing',
        time_column='timestamp',
        val_start=datetime(2023, 11, 20),
        test_start=datetime(2023, 11, 27),
        test_end=datetime(2023, 12, 4),
        reference_data_dir=reference_data_dir,
        reference_index_column=1,
    )

    data_parts = {
        'X_num': X_num_data,
        'X_meta': X_meta_data,
        'Y': Y_data,
        'X_cat': X_cat_data,
    }

    logger.info('Writing data to disk')
    save_dataset_variants(
        name='maps-routing',
        task_type='regression',
        data=data_parts,
        splits=all_splits,
        default_protocol='time',
        train_sizes=train_sizes,
        train_sampling=train_sampling,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    args = parse_args('maps-routing')
    main(
        args.raw_data_dir,
        output_dir=args.output_dir,
        train_sizes=args.train_size,
        train_sampling=args.train_sampling,
        reference_data_dir=args.reference_data_dir,
    )
