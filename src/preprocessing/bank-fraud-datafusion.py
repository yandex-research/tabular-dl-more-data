"""Bank account fraud dataset prep, from ods.ai"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import OrdinalEncoder

from preprocessing.util import (
    DATA_DIR,
    TRAIN_SIZE_LABELS,
    save_dataset_variants,
)

logger = logging.getLogger(__name__)
NAME = 'bank-fraud-datafusion'
BOUNDARIES = {
    'train': ['2024-10-01', '2025-05-04'],
    'val': ['2025-05-04', '2025-05-18'],
    'test': ['2025-05-18', '2025-06-01'],
}
TRAIN_SIZES = list(TRAIN_SIZE_LABELS)
PARTS = (1, 2, 3)
NEGATIVE_FRACTION = 0.15
SECONDS_PER_DAY = 86400


def in_period(period):
    start, end = BOUNDARIES[period]
    return pl.col('date').is_between(
        pl.lit(start).str.to_date(), pl.lit(end).str.to_date(), closed='left'
    )


CATEGORIES = [
    'event_type_nm',
    'event_desc',
    'channel_indicator_type',
    'channel_indicator_sub_type',
    'currency_iso_cd',
    'mcc_code',
    'pos_cd',
    'timezone',
    'operating_system_type',
    'phone_voip_call_state',
    'web_rdp_connection',
    'developer_tools',
    'compromised',
    'accept_language',
    'browser_language',
]


def read_metadata(raw):
    """Read split inputs; source_row indexes the concatenated raw training files."""
    labels = pl.read_parquet(raw / 'train_labels.parquet')
    data = (
        pl.scan_parquet([raw / f'train_part_{i}.parquet' for i in PARTS])
        .select('customer_id', 'event_id', 'event_dttm')
        .with_row_index('source_row')
        .with_columns(pl.col('event_dttm').str.to_datetime().alias('timestamp'))
        .drop('event_dttm')
        .collect()
    )
    keys = ['customer_id', 'event_id']
    return data.join(  # ty: ignore
        labels.select(*keys, 'target'), on=keys, how='left', maintain_order='left'
    ).with_columns(
        pl.col('target').fill_null(-1), pl.col('timestamp').dt.date().alias('date')
    )


def select_evaluation_days(days, seed):
    """All fraud days, otherwise one random day per customer; keep whole days."""
    fraud = days.filter(pl.col('positive') > 0)
    other = days.filter(
        ~pl.col('customer_id').is_in(fraud['customer_id'].unique().implode())
    )
    other = other.sort('customer_id', 'date')
    other = other[np.random.default_rng(seed).permutation(len(other))]
    other = other.unique('customer_id', keep='first')
    return (
        pl.concat([fraud, other])
        .select('customer_id', 'date')
        .sort('customer_id', 'date')
    )


def stratified_holdout(train, reference, seed):
    """Sample without replacement, matching reference row counts in each class."""
    selected = []
    for positive in (False, True):
        rows = train.filter((pl.col('target') == 1) == positive)
        count = reference.select(((pl.col('target') == 1) == positive).sum()).item()
        if count:
            indices = np.random.default_rng(seed).choice(
                len(rows),
                size=count,
                replace=False,
            )
            selected.append(rows[indices])
    held_out = pl.concat(selected).sort('source_row')
    remaining = train.join(held_out.select('event_id'), on='event_id', how='anti')
    return remaining, held_out


def make_splits(data, *, seed=0):
    """Temporal customer-day evaluation and size/prevalence-matched IID rows.

    Both protocols share training rows. IID evaluation is reserved before retaining
    all training positives and a fraction of the other rows.
    """
    days = (
        data.group_by('customer_id', 'date')
        .agg((pl.col('target') == 1).sum().alias('positive'))
        .sort('customer_id', 'date')
    )
    train = data.filter(in_period('train')).sort('source_row')
    evaluation = {}
    for offset, part in enumerate(('val', 'test')):
        chosen = select_evaluation_days(days.filter(in_period(part)), seed + offset)
        temporal = data.join(chosen, on=['customer_id', 'date'], how='semi')
        train, iid = stratified_holdout(train, temporal, seed + offset)
        evaluation[f'time_{part}'] = temporal
        evaluation[f'iid_{part}'] = iid

    positive = train.filter(pl.col('target') == 1)
    negative = train.filter(pl.col('target') != 1)
    indices = np.random.default_rng(seed).choice(
        len(negative), int(NEGATIVE_FRACTION * len(negative)), replace=False
    )
    populations = {'train': pl.concat([positive, negative[indices]]), **evaluation}
    candidates = pl.concat(
        [
            rows.with_columns(pl.lit(name).alias('split'))
            for name, rows in populations.items()
        ]
    ).sort('timestamp', 'event_id')
    membership = candidates['split'].to_numpy()
    train_idx = np.flatnonzero(membership == 'train')
    splits = {
        protocol: {
            'train_idx': train_idx,
            'val_idx': np.flatnonzero(membership == f'{protocol}_val'),
            'test_idx': np.flatnonzero(membership == f'{protocol}_test'),
        }
        for protocol in (
            'time',
            'iid',
        )
    }
    return (
        candidates,
        splits,
        {name: split_counts(rows) for name, rows in populations.items()},
    )


def split_counts(rows):
    positive = rows.filter(pl.col('target') == 1)
    return {
        'rows': len(rows),
        'positives': len(positive),
        'prevalence': len(positive) / len(rows),
        'customers': rows['customer_id'].n_unique(),
        'positive_customers': positive['customer_id'].n_unique(),
        'fraud_days': positive.select('customer_id', 'date').unique().height,
    }


def current_features(frame):
    """Current transaction only; no history or labels."""
    frame = frame.with_columns(
        pl.col('datetime').dt.weekday().sub(1).alias('weekday'),
        pl.col('datetime').dt.month().alias('month'),
        (
            pl.col('battery')
            .str.extract(r'(\d{1,3})', 1)
            .cast(pl.Float64, strict=False)
            .alias('battery_pct')
        ),
        (
            pl.col('device_system_version')
            .str.extract(r'^(\d+)', 1)
            .cast(pl.Float64, strict=False)
            .alias('os_version_major')
        ),
        (
            pl.col('screen_size')
            .str.extract(r'^(\d+)', 1)
            .cast(pl.Float64, strict=False)
            .alias('screen_width')
        ),
        (
            pl.col('screen_size')
            .str.extract(r'[xX](\d+)$', 1)
            .cast(pl.Float64, strict=False)
            .alias('screen_height')
        ),
    )
    amount = pl.col('amount')
    seconds_today = pl.col('timestamp') % SECONDS_PER_DAY
    width, height = pl.col('screen_width'), pl.col('screen_height')
    missing = {
        'currency_missing': 'currency_iso_cd',
        'mcc_missing': 'mcc_code',
        'pos_missing': 'pos_cd',
        'accept_language_missing': 'accept_language',
        'browser_language_missing': 'browser_language',
        'timezone_missing': 'timezone',
        'session_id_missing': 'session_id',
        'operating_system_missing': 'operating_system_type',
        'battery_missing': 'battery',
        'device_system_version_missing': 'device_system_version',
        'screen_size_missing': 'screen_size',
        'developer_tools_missing': 'developer_tools',
        'phone_voip_missing': 'phone_voip_call_state',
        'web_rdp_missing': 'web_rdp_connection',
        'compromised_missing': 'compromised',
    }
    return frame.select(
        amount,
        amount.abs().alias('amount_abs'),
        amount.abs().log1p().alias('amount_log_abs'),
        amount.is_null().alias('amount_is_missing'),
        (amount == 0).fill_null(False).alias('amount_is_zero'),
        (amount < 0).fill_null(False).alias('amount_is_negative'),
        (4 * amount.abs().log1p()).floor().clip(0, 63).alias('amount_bucket'),
        pl.col('datetime').dt.hour().alias('hour'),
        'weekday',
        'month',
        pl.col('datetime').dt.day().alias('day_of_month'),
        (pl.col('weekday') >= 5).alias('is_weekend'),
        (pl.col('datetime').dt.hour() < 6).alias('is_night'),
        (2 * np.pi * seconds_today / SECONDS_PER_DAY).sin().alias('hour_sin'),
        (2 * np.pi * seconds_today / SECONDS_PER_DAY).cos().alias('hour_cos'),
        (2 * np.pi * pl.col('weekday') / 7).sin().alias('weekday_sin'),
        (2 * np.pi * pl.col('weekday') / 7).cos().alias('weekday_cos'),
        (2 * np.pi * (pl.col('month') - 1) / 12).sin().alias('month_sin'),
        (2 * np.pi * (pl.col('month') - 1) / 12).cos().alias('month_cos'),
        'battery_pct',
        'os_version_major',
        'screen_width',
        'screen_height',
        (width * height).alias('screen_pixels'),
        pl.when((width > 0) & (height > 0)).then(width / height).alias('screen_ratio'),
        *[pl.col(source).is_null().alias(name) for name, source in missing.items()],
    )


def ratio(numerator, denominator):
    return pl.when(
        numerator.is_finite() & denominator.is_finite() & (denominator != 0)
    ).then(numerator / denominator)


def history_features(df):
    df = df.with_row_index('_row').with_columns(
        (pl.col('timestamp') // SECONDS_PER_DAY).alias('_day'),
        (pl.col('timestamp') // 3600).alias('_hour'),
        pl.col('amount').fill_nan(None),
        pl.col('session_id').cast(pl.Int64).fill_null(-1),
    )
    lifetime = int(df['timestamp'].max() - df['timestamp'].min()) + 1

    def rolling(keys, seconds, expressions):
        join_keys = [*keys, 'timestamp']

        ordered = df.select([*keys, 'timestamp', 'amount', '_row']).sort(
            [*keys, 'timestamp', '_row']
        )

        result = ordered.rolling(
            'timestamp', group_by=keys, period=f'{seconds}i', closed='left'
        ).agg(expressions)

        return (
            ordered.select([*join_keys, '_row'])
            .join(
                result.unique(subset=join_keys),
                on=join_keys,
                how='left',
                nulls_equal=True,
                validate='m:1',
                maintain_order='left',
            )
            .sort('_row')
            .drop([*join_keys, '_row'])
        )

    result = rolling(
        ['customer_id'],
        lifetime,
        [
            pl.len().cast(pl.Float64).alias('history_event_count_all'),
            pl.col('timestamp').first().alias('_first'),
            pl.col('timestamp').last().alias('_previous'),
            pl.col('amount').last().alias('previous_amount'),
            pl.col('amount').mean().alias('history_amount_mean_all'),
            pl.col('amount').std(ddof=0).alias('history_amount_std_all'),
            pl.col('amount').abs().max().alias('history_max_abs_amount_all'),
        ],
    ).with_columns(df['timestamp'], df['amount'])
    result = result.with_columns(
        ((pl.col('timestamp') - pl.col('_first')) / SECONDS_PER_DAY).alias(
            'history_age_days'
        ),
        (pl.col('timestamp') - pl.col('_previous')).alias(
            'seconds_since_previous_event'
        ),
        (pl.col('amount') - pl.col('previous_amount')).alias('amount_minus_previous'),
        ratio(pl.col('amount').abs(), pl.col('history_max_abs_amount_all')).alias(
            'amount_abs_to_personal_max'
        ),
    ).drop('_first', '_previous')

    days = (
        df.group_by('customer_id', '_day')
        .len()
        .sort('customer_id', '_day')
        .with_columns(
            pl.int_range(pl.len()).over('customer_id').alias('active_days_before'),
            pl.col('len')
            .cum_sum()
            .shift(1, fill_value=0)
            .over('customer_id')
            .alias('_before_today'),
        )
    )
    day_features = (
        df.select('customer_id', '_day')
        .join(days, on=['customer_id', '_day'], how='left', maintain_order='left')
        .select('active_days_before', '_before_today')
    )
    result = (
        result.hstack(day_features)
        .with_columns(
            ratio(pl.col('_before_today'), pl.col('active_days_before')).alias(
                'average_events_per_active_day'
            ),
            (pl.col('history_event_count_all') - pl.col('_before_today')).alias(
                'events_today_before'
            ),
        )
        .drop('_before_today')
    )
    result = result.hstack(
        rolling(
            ['customer_id', '_hour'],
            lifetime,
            [
                pl.len().alias('events_this_hour_before'),
            ],
        )
    )

    for window, seconds in {
        '1h': 3600,
        '6h': 21600,
        '24h': SECONDS_PER_DAY,
        '30d': 30 * SECONDS_PER_DAY,
        '365d': 365 * SECONDS_PER_DAY,
    }.items():
        expressions = [
            pl.len().alias(f'history_event_count_{window}'),
            pl.col('amount').sum().alias(f'history_amount_sum_{window}'),
            pl.col('amount').mean().alias(f'history_amount_mean_{window}'),
            pl.col('amount').std(ddof=0).alias(f'history_amount_std_{window}'),
        ]
        if window == '24h':
            expressions.append(
                pl.col('amount').abs().max().alias('history_max_abs_amount_24h')
            )
        result = result.hstack(rolling(['customer_id'], seconds, expressions))

    for name, field in {
        'event_type': 'event_type_nm',
        'event_desc': 'event_desc',
        'channel': 'channel_indicator_type',
        'subchannel': 'channel_indicator_sub_type',
        'currency': 'currency_iso_cd',
        'mcc': 'mcc_code',
    }.items():
        group = rolling(
            ['customer_id', field],
            lifetime,
            [
                pl.len().cast(pl.Float64).alias('_count'),
                pl.col('timestamp').last().alias('_last'),
            ],
        ).with_columns(df[field], df['timestamp'])
        result = result.hstack(
            group.select(
                pl.when(pl.col(field).is_not_null())
                .then(pl.col('_count'))
                .alias(f'same_{name}_count_all'),
                pl.when(pl.col(field).is_not_null())
                .then(pl.col('timestamp') - pl.col('_last'))
                .alias(f'seconds_since_same_{name}'),
                pl.when(pl.col(field).is_not_null())
                .then((pl.col('_count') == 0).cast(pl.Float64))
                .alias(f'{name}_is_novel'),
            )
        )

    session = rolling(
        ['customer_id', 'session_id'],
        lifetime,
        [
            pl.len().cast(pl.Float64).alias('_count'),
            pl.col('timestamp').first().alias('_first'),
            pl.col('amount').sum().alias('_sum'),
        ],
    ).with_columns(df['session_id'], df['timestamp'])
    result = result.hstack(
        session.select(
            pl.when(pl.col('session_id') != -1)
            .then(pl.col('_count'))
            .alias('session_event_count_before'),
            pl.when(pl.col('session_id') != -1)
            .then((pl.col('_count') == 0).cast(pl.Float64))
            .alias('session_is_first_event'),
            pl.when(pl.col('session_id') != -1)
            .then((pl.col('timestamp') - pl.col('_first')).fill_null(0))
            .alias('seconds_since_session_start'),
            pl.when(pl.col('session_id') != -1)
            .then(pl.col('_sum'))
            .alias('session_amount_sum_before'),
        )
    )
    for short, long in [('1h', '24h'), ('30d', '365d')]:
        result = result.with_columns(
            ratio(
                pl.col(f'history_event_count_{short}'),
                pl.col(f'history_event_count_{long}'),
            ).alias(f'event_count_ratio_{short}_{long}')
        )
    for window in ['24h', '30d', '365d']:
        result = result.with_columns(
            ratio(pl.col('amount'), pl.col(f'history_amount_mean_{window}')).alias(
                f'amount_to_mean_{window}'
            )
        )
    for window in ['1h', '24h']:
        result = result.with_columns(
            ratio(pl.col('amount'), pl.col(f'history_amount_sum_{window}')).alias(
                f'amount_to_sum_{window}'
            )
        )
    return (
        result.with_columns(
            ratio(
                pl.col('amount') - pl.col('history_amount_mean_30d'),
                pl.col('history_amount_std_30d'),
            ).alias('amount_zscore_30d')
        )
        .drop('timestamp', 'amount')
        .cast(pl.Float64)
    )


def part_features(raw, part, candidates):
    """Each matching pretrain/train pair contains complete, disjoint customers."""
    frame = (
        pl.concat(
            [
                pl.read_parquet(raw / f'{period}_part_{part}.parquet')
                for period in ['pretrain', 'train']
            ],
            how='diagonal_relaxed',
        )
        .with_columns(pl.col('event_dttm').str.to_datetime().alias('datetime'))
        .with_columns(
            pl.col('datetime').dt.epoch('s').alias('timestamp'),
            pl.when(pl.col('operaton_amt').cast(pl.Float64).is_finite())
            .then(pl.col('operaton_amt').cast(pl.Float64))
            .alias('amount'),
            *[pl.col(name).cast(pl.String) for name in CATEGORIES],
        )
        .sort('customer_id', 'timestamp', 'event_id')
    )

    selected = (
        frame.select('customer_id', 'event_id')
        .with_row_index('part_row')
        .join(
            candidates.with_row_index('dataset_row').select(
                'customer_id', 'event_id', 'dataset_row'
            ),
            on=['customer_id', 'event_id'],
            how='inner',
            maintain_order='left',
        )
    )
    rows = selected['part_row']
    current = current_features(frame[rows])
    history = history_features(frame)[rows]
    return (
        current.hstack(history)
        .cast(pl.Float32)
        .hstack(frame[rows].select(CATEGORIES))
        .hstack(selected.select('dataset_row'))
    )


def encode_categories(features):
    """Encode selected rows; retain source nulls as -1, without rare-value merging."""
    values = features.select(CATEGORIES).to_numpy().astype(object)
    values[values == None] = np.nan  # noqa: E711
    encoder = OrdinalEncoder(encoded_missing_value=-1, dtype=np.int64)
    categorical = pl.DataFrame(encoder.fit_transform(values), schema=CATEGORIES)
    dictionaries = {
        name: [value for value in categories if isinstance(value, str)]
        for name, categories in zip(CATEGORIES, encoder.categories_)
    }
    return categorical, dictionaries


def main(
    raw_data_dir, *, output_dir=None, train_sizes=None, train_sampling='both', seed=0
):
    raw = Path(raw_data_dir).expanduser().resolve()
    output = DATA_DIR if output_dir is None else Path(output_dir).expanduser().resolve()
    target = output / NAME
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    started = time.monotonic()
    sizes = TRAIN_SIZES if train_sizes is None else train_sizes
    candidates, splits, counts = make_splits(read_metadata(raw), seed=seed)
    logger.info('Split counts: %s', counts)

    features = (
        pl.concat([part_features(raw, part, candidates) for part in PARTS])
        .sort('dataset_row')
        .drop('dataset_row')
    )

    categorical, dictionaries = encode_categories(features)
    data = {
        'X_num': features.drop(CATEGORIES),
        'X_cat': categorical,
        'X_meta': candidates.select(
            pl.col('customer_id').cast(pl.Int64),
            pl.col('event_id').cast(pl.Int64),
            pl.col('timestamp').dt.epoch('s'),
            pl.col('source_row').cast(pl.Int64),
        ),
        'Y': candidates.select((pl.col('target') == 1).cast(pl.Int64).alias('target')),
    }
    save_dataset_variants(
        NAME,
        'binclass',
        data,
        splits,
        output_dir=output,
        train_sizes=sizes,
        train_sampling=train_sampling,
        seed=seed,
        default_protocol='time',
        metadata={
            'feature_names': {
                'numerical': data['X_num'].columns,
                'categorical': CATEGORIES,
            },
            'meta_columns': data['X_meta'].columns,
            'boundaries': BOUNDARIES,
            'population_counts': counts,
            'label_history': False,
        },
    )
    (target / 'categories.json').write_text(json.dumps(dictionaries, indent=2))
    files = [
        raw / f'{period}_part_{part}.parquet'
        for period in ('pretrain', 'train')
        for part in PARTS
    ]
    provenance = {
        'raw_directory': str(raw),
        'raw_files': {
            path.name: path.stat().st_size
            for path in [*files, raw / 'train_labels.parquet']
        },
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy': np.__version__,
        'polars': pl.__version__,
        'seconds': time.monotonic() - started,
        'sampling': {
            'seed': seed,
            'negative_fraction': NEGATIVE_FRACTION,
            'iid_reference': 'temporal evaluation class counts',
            'evaluation_seeds': {'val': seed, 'test': seed + 1},
        },
    }
    (target / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    logger.info('Finished %s in %.1fs', target, provenance['seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_data_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-size', nargs='+', default=TRAIN_SIZES)
    parser.add_argument(
        '--train-sampling', choices=['random', 'latest', 'both'], default='both'
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main(
        args.reference_data_dir,
        output_dir=args.output_dir,
        train_sizes=args.train_size,
        train_sampling=args.train_sampling,
        seed=args.seed,
    )
