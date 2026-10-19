"""Prepare a 150-feature Aeolus departure-delay classification task.

The target is ``DEP_DELAY > 15`` minutes. The features reproduce and extend the
Aeolus tabular benchmark recipe with domain schedule, calendar, route, geospatial,
weather-interaction, climatology, and causal operational-history features. History
windows use only the shared train pool after removing IID holdouts, with a
conservative full-calendar-day buffer. Actual-operation columns from
the predicted flight are excluded because they are only known after the prediction
target starts to materialize.

The Kaggle archive is large and contains graph/chain modalities that are irrelevant
to this tabular task. When given the ZIP directly, this script extracts only the nine
``Flight_Tab`` CSV files into the ignored preprocessing cache.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from preprocessing.util import (
    PROJECT_DIR,
    TrainSamplingSelection,
    default_train_sizes,
    make_iid_split,
    parse_args,
    save_dataset_variants,
)


logger = logging.getLogger(__name__)

DATASET_NAME = 'aeolus-flight-delay'
YEARS = tuple(range(2016, 2025))
TRAIN_END_YEAR = 2022
VALIDATION_YEAR = 2023
TEST_YEAR = 2024
DEFAULT_EVAL_SIZE = 100_000
DEFAULT_MAX_TRAIN_SIZE = 10_000_000
DEFAULT_MAX_CATEGORIES = 512  # Includes one unknown code (-1).
SOURCE_ID_STRIDE = 10_000_000

RAW_COLUMNS = [
    'FL_DATE',
    'OP_CARRIER',
    'OP_CARRIER_FL_NUM',
    'ORIGIN',
    'DEST',
    'CRS_DEP_TIME',
    'DEP_DELAY',
    'CRS_ARR_TIME',
    'CRS_ELAPSED_TIME',
    'MONTH',
    'DAY_OF_MONTH',
    'DAY_OF_WEEK',
    'ORIGIN_INDEX',
    'DEST_INDEX',
    'O_TEMP',
    'O_PRCP',
    'O_WSPD',
    'D_TEMP',
    'D_PRCP',
    'D_WSPD',
    'O_LATITUDE',
    'O_LONGITUDE',
    'D_LATITUDE',
    'D_LONGITUDE',
]

BASE_CAT_FEATURES = [
    'operating_carrier',
    'flight_number',
    'origin_airport',
    'destination_airport',
    'route',
    'flight_year',
    'month',
    'day_of_month',
    'day_of_week',
]
BASE_NUM_FEATURES = [
    'scheduled_departure_minutes',
    'scheduled_departure_sin',
    'scheduled_departure_cos',
    'scheduled_arrival_minutes',
    'scheduled_arrival_sin',
    'scheduled_arrival_cos',
    'day_of_year_sin',
    'day_of_year_cos',
    'scheduled_elapsed_minutes',
    'origin_temperature',
    'origin_precipitation',
    'origin_wind_speed',
    'destination_temperature',
    'destination_precipitation',
    'destination_wind_speed',
    'origin_latitude',
    'origin_longitude',
    'destination_latitude',
    'destination_longitude',
    'great_circle_distance_km',
]

ENGINEERED_CAT_FEATURES = [
    'carrier_flight',
    'carrier_origin',
    'carrier_destination',
    'carrier_route',
    'origin_departure_hour',
    'destination_arrival_hour',
    'route_day_of_week',
    'route_month',
    'distance_band',
]
CALENDAR_FEATURES = [
    'scheduled_departure_hour',
    'scheduled_arrival_hour',
    'is_weekend',
    'quarter',
    'week_of_year_sin',
    'week_of_year_cos',
    'month_sin',
    'month_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'scheduled_timezone_offset_hours',
]
GEOMETRY_FEATURES = [
    'log_great_circle_distance_km',
    'latitude_delta',
    'longitude_delta_wrapped',
    'absolute_latitude_delta',
    'absolute_longitude_delta',
    'midpoint_latitude',
    'midpoint_longitude',
    'bearing_sin',
    'bearing_cos',
    'scheduled_speed_kmh',
    'scheduled_minutes_per_1000km',
    'is_long_haul',
]
WEATHER_INTERACTION_FEATURES = [
    'mean_temperature',
    'temperature_delta',
    'absolute_temperature_delta',
    'minimum_temperature',
    'maximum_temperature',
    'total_precipitation',
    'maximum_precipitation',
    'minimum_precipitation',
    'precipitation_delta',
    'has_precipitation',
    'both_endpoints_have_precipitation',
    'has_heavy_precipitation',
    'mean_wind_speed',
    'maximum_wind_speed',
    'minimum_wind_speed',
    'wind_speed_delta',
    'absolute_wind_speed_delta',
    'has_high_wind',
    'origin_freezing_precipitation',
    'destination_freezing_precipitation',
]
WEATHER_CLIMATOLOGY_FEATURES = [
    'origin_temperature_anomaly',
    'destination_temperature_anomaly',
    'origin_wind_speed_anomaly',
    'destination_wind_speed_anomaly',
    'origin_log_precipitation_anomaly',
    'destination_log_precipitation_anomaly',
]

HISTORY_WINDOWS = {'7d': '7d', '28d': '28d', '90d': '90d'}
HISTORY_GROUPS = {
    'origin': ['origin_airport'],
    'destination': ['destination_airport'],
    'carrier': ['operating_carrier'],
    'carrier_origin': ['operating_carrier', 'origin_airport'],
    'carrier_destination': ['operating_carrier', 'destination_airport'],
}
HISTORY_FEATURES = [
    feature
    for group in HISTORY_GROUPS
    for window in HISTORY_WINDOWS
    for feature in (
        f'{group}_flights_previous_{window}',
        f'{group}_delayed_flights_previous_{window}',
        f'{group}_delay_rate_previous_{window}',
        f'{group}_mean_departure_delay_previous_{window}',
    )
] + [
    'days_since_origin_previous_active_day',
    'days_since_destination_previous_active_day',
    'days_since_carrier_previous_active_day',
]

CAT_FEATURES = BASE_CAT_FEATURES + ENGINEERED_CAT_FEATURES
STRING_CAT_FEATURES = [
    'operating_carrier',
    'carrier_flight',
    'carrier_origin',
    'carrier_destination',
    'carrier_route',
]
NUM_FEATURES = (
    BASE_NUM_FEATURES
    + CALENDAR_FEATURES
    + GEOMETRY_FEATURES
    + WEATHER_INTERACTION_FEATURES
    + WEATHER_CLIMATOLOGY_FEATURES
    + HISTORY_FEATURES
)
assert len(CAT_FEATURES) == 18
assert len(NUM_FEATURES) == 132
assert len(CAT_FEATURES) + len(NUM_FEATURES) == 150
META_FEATURES = ['scheduled_departure_unix_us', 'source_row_id']

EXCLUDED_LEAKAGE_COLUMNS = [
    'DEP_TIME',
    'DEP_DELAY',
    'TAXI_OUT',
    'WHEELS_OFF',
    'WHEELS_ON',
    'TAXI_IN',
    'ARR_TIME',
    'ARR_DELAY',
    'ACTUAL_ELAPSED_TIME',
    'AIR_TIME',
]
WEATHER_COLUMNS = [
    'O_TEMP',
    'O_PRCP',
    'O_WSPD',
    'D_TEMP',
    'D_PRCP',
    'D_WSPD',
]


def _csv_filename(year: int) -> str:
    return f'flight_with_weather_{year}.csv'


def _find_extracted_files(raw_path: Path) -> dict[int, Path] | None:
    if not raw_path.is_dir():
        return None
    files: dict[int, Path] = {}
    for year in YEARS:
        matches = list(raw_path.rglob(_csv_filename(year)))
        if len(matches) != 1:
            return None
        files[year] = matches[0]
    return files


def prepare_flight_tab_files(raw_data_path: str | Path) -> dict[int, Path]:
    """Locate or selectively extract the nine tabular CSV files."""

    raw_path = Path(raw_data_path).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f'Raw data path does not exist: {raw_path}')

    extracted = _find_extracted_files(raw_path)
    if extracted is not None:
        return extracted

    archives: list[Path]
    if raw_path.is_file() and raw_path.suffix.lower() == '.zip':
        archives = [raw_path]
    elif raw_path.is_dir():
        archives = sorted(raw_path.rglob('*.zip'))
    else:
        archives = []

    cache_dir = PROJECT_DIR / 'preprocessing' / 'tmp' / DATASET_NAME / 'raw'
    cache_dir.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            members = {
                year: next(
                    (
                        info
                        for info in zf.infolist()
                        if Path(info.filename).name == _csv_filename(year)
                    ),
                    None,
                )
                for year in YEARS
            }
            if any(info is None for info in members.values()):
                continue

            output: dict[int, Path] = {}
            for year, maybe_info in members.items():
                assert maybe_info is not None
                destination = cache_dir / _csv_filename(year)
                if (
                    not destination.exists()
                    or destination.stat().st_size != maybe_info.file_size
                ):
                    logger.info('Extracting %s from %s', maybe_info.filename, archive)
                    temporary = destination.with_suffix('.csv.partial')
                    with zf.open(maybe_info) as source, temporary.open('wb') as target:
                        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                    temporary.replace(destination)
                output[year] = destination

            marker = cache_dir.parent / 'source.json'
            marker.write_text(
                json.dumps(
                    {
                        'archive': str(archive),
                        'size': archive.stat().st_size,
                        'mtime_ns': archive.stat().st_mtime_ns,
                    },
                    indent=4,
                )
                + '\n'
            )
            return output

    raise FileNotFoundError(
        'Could not find all nine Aeolus Flight_Tab CSV files in '
        f'{raw_path} or in a ZIP below it.'
    )


def _read_year(path: Path, year: int) -> pl.DataFrame:
    logger.info('Reading Aeolus %d from %s', year, path)
    data = pl.read_csv(
        path,
        columns=RAW_COLUMNS,
        try_parse_dates=True,
        infer_schema_length=10_000,
        low_memory=True,
    ).with_row_index('_row_in_year')
    return data.with_columns(
        (
            pl.lit(year, dtype=pl.Int64) * SOURCE_ID_STRIDE
            + pl.col('_row_in_year').cast(pl.Int64)
        ).alias('_source_id')
    )


def audit_raw_file(path: Path, year: int) -> dict[str, object]:
    """Collect source-level checks without materializing the full CSV."""

    logger.info('Auditing raw Aeolus %d', year)
    data = pl.scan_csv(
        path,
        try_parse_dates=True,
        infer_schema_length=10_000,
        low_memory=True,
    )
    flight_key = [
        'OP_CARRIER',
        'OP_CARRIER_FL_NUM',
        'ORIGIN',
        'DEST',
        'CRS_DEP_TIME',
    ]
    collected = cast(
        pl.DataFrame,
        data.select(
            pl.len().alias('rows'),
            pl.col('DEP_DELAY').null_count().alias('target_null_rows'),
            (pl.col('DEP_DELAY') > 15.0).sum().alias('positive_rows'),
            pl.col('CRS_DEP_TIME').min().alias('scheduled_time_min'),
            pl.col('CRS_DEP_TIME').max().alias('scheduled_time_max'),
            pl.struct(flight_key).n_unique().alias('unique_flight_keys'),
            *(
                pl.col(column).null_count().alias(f'{column}_nulls')
                for column in RAW_COLUMNS
            ),
            *(
                pl.col(column).is_nan().sum().alias(f'{column}_nans')
                for column in WEATHER_COLUMNS
            ),
        ).collect(engine='streaming'),
    )
    result = collected.to_dicts()[0]

    labeled_rows = result['rows'] - result['target_null_rows']
    result['labeled_rows'] = labeled_rows
    result['positive_rate'] = result['positive_rows'] / labeled_rows
    result['duplicate_flight_keys'] = result['rows'] - result['unique_flight_keys']
    for key in ('scheduled_time_min', 'scheduled_time_max'):
        result[key] = result[key].isoformat()
    return {'year': year, **result}


def _great_circle_distance_expr() -> pl.Expr:
    """Haversine distance using only static airport coordinates."""

    lat1 = pl.col('O_LATITUDE').radians()
    lon1 = pl.col('O_LONGITUDE').radians()
    lat2 = pl.col('D_LATITUDE').radians()
    lon2 = pl.col('D_LONGITUDE').radians()
    haversine = ((lat2 - lat1) / 2).sin().pow(2) + lat1.cos() * lat2.cos() * (
        (lon2 - lon1) / 2
    ).sin().pow(2)
    return (2.0 * 6371.0088 * haversine.clip(0.0, 1.0).sqrt().arcsin()).alias(
        'great_circle_distance_km'
    )


def make_model_table(
    data: pl.DataFrame, *, include_history_delay: bool = False
) -> pl.DataFrame:
    """Create the Aeolus tabular features and the binary target."""

    required = [
        'CRS_DEP_TIME',
        'CRS_ARR_TIME',
        'DEP_DELAY',
        'ORIGIN_INDEX',
        'DEST_INDEX',
    ]
    data = data.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in required))

    departure_minute = pl.col('CRS_DEP_TIME').dt.hour().cast(pl.Int64) * 60 + pl.col(
        'CRS_DEP_TIME'
    ).dt.minute().cast(pl.Int64)
    arrival_minute = pl.col('CRS_ARR_TIME').dt.hour().cast(pl.Int64) * 60 + pl.col(
        'CRS_ARR_TIME'
    ).dt.minute().cast(pl.Int64)
    day_of_year = pl.col('CRS_DEP_TIME').dt.ordinal_day()

    return data.select(
        pl.col('CRS_DEP_TIME').alias('_scheduled_departure'),
        pl.col('_source_id'),
        *(
            [pl.col('DEP_DELAY').alias('_history_departure_delay')]
            if include_history_delay
            else []
        ),
        pl.col('OP_CARRIER').fill_null('__MISSING__').alias('operating_carrier'),
        pl.col('OP_CARRIER_FL_NUM').fill_null(-1).cast(pl.Int64).alias('flight_number'),
        pl.col('ORIGIN_INDEX').cast(pl.Int64).alias('origin_airport'),
        pl.col('DEST_INDEX').cast(pl.Int64).alias('destination_airport'),
        (
            pl.col('ORIGIN_INDEX').cast(pl.Int64) * 1024
            + pl.col('DEST_INDEX').cast(pl.Int64)
        ).alias('route'),
        pl.col('CRS_DEP_TIME').dt.year().cast(pl.Int64).alias('flight_year'),
        pl.col('CRS_DEP_TIME').dt.month().cast(pl.Int64).alias('month'),
        pl.col('CRS_DEP_TIME').dt.day().cast(pl.Int64).alias('day_of_month'),
        pl.col('CRS_DEP_TIME').dt.weekday().cast(pl.Int64).alias('day_of_week'),
        departure_minute.cast(pl.Float64).alias('scheduled_departure_minutes'),
        (departure_minute * (2.0 * math.pi / 1440.0))
        .sin()
        .alias('scheduled_departure_sin'),
        (departure_minute * (2.0 * math.pi / 1440.0))
        .cos()
        .alias('scheduled_departure_cos'),
        arrival_minute.cast(pl.Float64).alias('scheduled_arrival_minutes'),
        (arrival_minute * (2.0 * math.pi / 1440.0))
        .sin()
        .alias('scheduled_arrival_sin'),
        (arrival_minute * (2.0 * math.pi / 1440.0))
        .cos()
        .alias('scheduled_arrival_cos'),
        (day_of_year * (2.0 * math.pi / 365.25)).sin().alias('day_of_year_sin'),
        (day_of_year * (2.0 * math.pi / 365.25)).cos().alias('day_of_year_cos'),
        pl.col('CRS_ELAPSED_TIME').alias('scheduled_elapsed_minutes'),
        pl.col('O_TEMP').alias('origin_temperature'),
        pl.col('O_PRCP').alias('origin_precipitation'),
        pl.col('O_WSPD').alias('origin_wind_speed'),
        pl.col('D_TEMP').alias('destination_temperature'),
        pl.col('D_PRCP').alias('destination_precipitation'),
        pl.col('D_WSPD').alias('destination_wind_speed'),
        pl.col('O_LATITUDE').alias('origin_latitude'),
        pl.col('O_LONGITUDE').alias('origin_longitude'),
        pl.col('D_LATITUDE').alias('destination_latitude'),
        pl.col('D_LONGITUDE').alias('destination_longitude'),
        _great_circle_distance_expr(),
        (pl.col('DEP_DELAY') > 15.0).cast(pl.Int64).alias('_target'),
    )


def add_row_level_features(data: pl.DataFrame) -> pl.DataFrame:
    """Add schedule, route, and weather interactions available for one flight."""

    departure_minutes = pl.col('scheduled_departure_minutes')
    arrival_minutes = pl.col('scheduled_arrival_minutes')
    elapsed_minutes = pl.col('scheduled_elapsed_minutes').clip(1.0, None)
    distance = pl.col('great_circle_distance_km').clip(1.0, None)

    latitude_delta = pl.col('destination_latitude') - pl.col('origin_latitude')
    raw_longitude_delta = pl.col('destination_longitude') - pl.col('origin_longitude')
    longitude_delta = ((raw_longitude_delta + 180.0) % 360.0) - 180.0
    midpoint_longitude = (
        (pl.col('origin_longitude') + longitude_delta / 2.0 + 180.0) % 360.0
    ) - 180.0

    lat1 = pl.col('origin_latitude').radians()
    lat2 = pl.col('destination_latitude').radians()
    delta_lon_radians = longitude_delta.radians()
    bearing_y = delta_lon_radians.sin() * lat2.cos()
    bearing_x = lat1.cos() * lat2.sin() - (
        lat1.sin() * lat2.cos() * delta_lon_radians.cos()
    )
    bearing_norm = (bearing_x.pow(2) + bearing_y.pow(2)).sqrt().clip(1e-12, None)

    origin_temperature = pl.col('origin_temperature')
    destination_temperature = pl.col('destination_temperature')
    origin_precipitation = pl.col('origin_precipitation')
    destination_precipitation = pl.col('destination_precipitation')
    origin_wind = pl.col('origin_wind_speed')
    destination_wind = pl.col('destination_wind_speed')

    route = pl.col('route').cast(pl.String)
    carrier = pl.col('operating_carrier').cast(pl.String)
    flight_number = pl.col('flight_number').cast(pl.String)
    origin = pl.col('origin_airport').cast(pl.String)
    destination = pl.col('destination_airport').cast(pl.String)
    departure_hour = (departure_minutes // 60).cast(pl.Int64)
    arrival_hour = (arrival_minutes // 60).cast(pl.Int64)

    timezone_offset = (
        (
            arrival_minutes
            - departure_minutes
            - pl.col('scheduled_elapsed_minutes')
            + 720.0
        )
        % 1440.0
        - 720.0
    ) / 60.0

    result = data.with_columns(
        pl.concat_str([carrier, flight_number], separator='|').alias('carrier_flight'),
        pl.concat_str([carrier, origin], separator='|').alias('carrier_origin'),
        pl.concat_str([carrier, destination], separator='|').alias(
            'carrier_destination'
        ),
        pl.concat_str([carrier, route], separator='|').alias('carrier_route'),
        (pl.col('origin_airport').cast(pl.Int64) * 24 + departure_hour).alias(
            'origin_departure_hour'
        ),
        (pl.col('destination_airport').cast(pl.Int64) * 24 + arrival_hour).alias(
            'destination_arrival_hour'
        ),
        (pl.col('route').cast(pl.Int64) * 8 + pl.col('day_of_week')).alias(
            'route_day_of_week'
        ),
        (pl.col('route').cast(pl.Int64) * 16 + pl.col('month')).alias('route_month'),
        pl.when(distance < 500.0)
        .then(0)
        .when(distance < 1000.0)
        .then(1)
        .when(distance < 1500.0)
        .then(2)
        .when(distance < 2500.0)
        .then(3)
        .when(distance < 4000.0)
        .then(4)
        .otherwise(5)
        .cast(pl.Int64)
        .alias('distance_band'),
        departure_hour.cast(pl.Float64).alias('scheduled_departure_hour'),
        arrival_hour.cast(pl.Float64).alias('scheduled_arrival_hour'),
        (pl.col('day_of_week') >= 6).cast(pl.Float64).alias('is_weekend'),
        (((pl.col('month') - 1) // 3) + 1).cast(pl.Float64).alias('quarter'),
        (pl.col('_scheduled_departure').dt.week() * (2.0 * math.pi / 52.1775))
        .sin()
        .alias('week_of_year_sin'),
        (pl.col('_scheduled_departure').dt.week() * (2.0 * math.pi / 52.1775))
        .cos()
        .alias('week_of_year_cos'),
        (pl.col('month') * (2.0 * math.pi / 12.0)).sin().alias('month_sin'),
        (pl.col('month') * (2.0 * math.pi / 12.0)).cos().alias('month_cos'),
        (pl.col('day_of_week') * (2.0 * math.pi / 7.0)).sin().alias('day_of_week_sin'),
        (pl.col('day_of_week') * (2.0 * math.pi / 7.0)).cos().alias('day_of_week_cos'),
        timezone_offset.alias('scheduled_timezone_offset_hours'),
        distance.log1p().alias('log_great_circle_distance_km'),
        latitude_delta.alias('latitude_delta'),
        longitude_delta.alias('longitude_delta_wrapped'),
        latitude_delta.abs().alias('absolute_latitude_delta'),
        longitude_delta.abs().alias('absolute_longitude_delta'),
        ((pl.col('origin_latitude') + pl.col('destination_latitude')) / 2.0).alias(
            'midpoint_latitude'
        ),
        midpoint_longitude.alias('midpoint_longitude'),
        (bearing_y / bearing_norm).alias('bearing_sin'),
        (bearing_x / bearing_norm).alias('bearing_cos'),
        (distance * 60.0 / elapsed_minutes).alias('scheduled_speed_kmh'),
        (elapsed_minutes * 1000.0 / distance).alias('scheduled_minutes_per_1000km'),
        (distance >= 3000.0).cast(pl.Float64).alias('is_long_haul'),
        ((origin_temperature + destination_temperature) / 2.0).alias(
            'mean_temperature'
        ),
        (destination_temperature - origin_temperature).alias('temperature_delta'),
        (destination_temperature - origin_temperature)
        .abs()
        .alias('absolute_temperature_delta'),
        pl.min_horizontal(origin_temperature, destination_temperature).alias(
            'minimum_temperature'
        ),
        pl.max_horizontal(origin_temperature, destination_temperature).alias(
            'maximum_temperature'
        ),
        (origin_precipitation + destination_precipitation).alias('total_precipitation'),
        pl.max_horizontal(origin_precipitation, destination_precipitation).alias(
            'maximum_precipitation'
        ),
        pl.min_horizontal(origin_precipitation, destination_precipitation).alias(
            'minimum_precipitation'
        ),
        (destination_precipitation - origin_precipitation).alias('precipitation_delta'),
        ((origin_precipitation > 0.0) | (destination_precipitation > 0.0))
        .cast(pl.Float64)
        .alias('has_precipitation'),
        ((origin_precipitation > 0.0) & (destination_precipitation > 0.0))
        .cast(pl.Float64)
        .alias('both_endpoints_have_precipitation'),
        ((origin_precipitation >= 2.5) | (destination_precipitation >= 2.5))
        .cast(pl.Float64)
        .alias('has_heavy_precipitation'),
        ((origin_wind + destination_wind) / 2.0).alias('mean_wind_speed'),
        pl.max_horizontal(origin_wind, destination_wind).alias('maximum_wind_speed'),
        pl.min_horizontal(origin_wind, destination_wind).alias('minimum_wind_speed'),
        (destination_wind - origin_wind).alias('wind_speed_delta'),
        (destination_wind - origin_wind).abs().alias('absolute_wind_speed_delta'),
        ((origin_wind >= 30.0) | (destination_wind >= 30.0))
        .cast(pl.Float64)
        .alias('has_high_wind'),
        ((origin_temperature <= 0.0) & (origin_precipitation > 0.0))
        .cast(pl.Float64)
        .alias('origin_freezing_precipitation'),
        ((destination_temperature <= 0.0) & (destination_precipitation > 0.0))
        .cast(pl.Float64)
        .alias('destination_freezing_precipitation'),
    )
    return result


def impute_weather_features(data: pl.DataFrame, train_rows: int) -> pl.DataFrame:
    """Impute rare source NaNs from train-only airport/month climatology."""

    pairs = [
        ('origin_airport', source)
        for source in (
            'origin_temperature',
            'origin_precipitation',
            'origin_wind_speed',
        )
    ] + [
        ('destination_airport', source)
        for source in (
            'destination_temperature',
            'destination_precipitation',
            'destination_wind_speed',
        )
    ]
    train = data.head(train_rows)
    result = data
    for airport, source in pairs:
        baseline = f'_imputation_{source}'
        finite_value = pl.col(source).filter(pl.col(source).is_finite())
        statistics = train.group_by(airport, 'month').agg(
            finite_value.mean().alias(baseline)
        )
        global_mean = train.select(finite_value.mean()).item()
        result = (
            result.join(
                statistics, on=[airport, 'month'], how='left', maintain_order='left'
            )
            .with_columns(
                pl.when(pl.col(source).is_finite())
                .then(pl.col(source))
                .otherwise(pl.col(baseline).fill_null(global_mean))
                .alias(source)
            )
            .drop(baseline)
        )
    return result


def add_weather_climatology_features(
    data: pl.DataFrame, train_rows: int
) -> pl.DataFrame:
    """Add train-fitted airport/month weather anomalies without eval leakage."""

    specifications = [
        (
            'origin_airport',
            'origin_temperature',
            'origin_temperature_anomaly',
            False,
        ),
        (
            'destination_airport',
            'destination_temperature',
            'destination_temperature_anomaly',
            False,
        ),
        (
            'origin_airport',
            'origin_wind_speed',
            'origin_wind_speed_anomaly',
            False,
        ),
        (
            'destination_airport',
            'destination_wind_speed',
            'destination_wind_speed_anomaly',
            False,
        ),
        (
            'origin_airport',
            'origin_precipitation',
            'origin_log_precipitation_anomaly',
            True,
        ),
        (
            'destination_airport',
            'destination_precipitation',
            'destination_log_precipitation_anomaly',
            True,
        ),
    ]
    train = data.head(train_rows)
    result = data
    for airport, source, output, use_log in specifications:
        value = pl.col(source).log1p() if use_log else pl.col(source)
        baseline = f'_baseline_{output}'
        statistics = train.group_by(airport, 'month').agg(value.mean().alias(baseline))
        global_mean = train.select(value.mean()).item()
        result = (
            result.join(
                statistics, on=[airport, 'month'], how='left', maintain_order='left'
            )
            .with_columns(
                (value - pl.col(baseline).fill_null(global_mean)).alias(output)
            )
            .drop(baseline)
        )
    return result


def daily_history_aggregates(data: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Aggregate one raw annual table for causal operational history features."""

    context = (
        data.filter(
            pl.all_horizontal(
                pl.col(column).is_not_null()
                for column in (
                    'CRS_DEP_TIME',
                    'DEP_DELAY',
                    'OP_CARRIER',
                    'ORIGIN_INDEX',
                    'DEST_INDEX',
                )
            )
        )
        .select(
            pl.col('CRS_DEP_TIME').dt.date().alias('_history_day'),
            pl.col('OP_CARRIER').alias('operating_carrier'),
            pl.col('ORIGIN_INDEX').cast(pl.Int64).alias('origin_airport'),
            pl.col('DEST_INDEX').cast(pl.Int64).alias('destination_airport'),
            pl.col('DEP_DELAY').cast(pl.Float64).alias('_departure_delay'),
        )
        .with_columns(
            (pl.col('_departure_delay') > 15.0).cast(pl.Int64).alias('_delayed')
        )
    )
    return {
        group: context.group_by(*keys, '_history_day').agg(
            pl.len().cast(pl.Int64).alias('_flights'),
            pl.col('_delayed').sum().cast(pl.Int64).alias('_delayed_flights'),
            pl.col('_departure_delay').sum().alias('_departure_delay_sum'),
        )
        for group, keys in HISTORY_GROUPS.items()
    }


def add_historical_features(
    data: pl.DataFrame,
    daily_parts: dict[str, list[pl.DataFrame]] | dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Join histories ending before a conservative 24-hour calendar cutoff.

    For a flight on day D, the lookup is evaluated at D-1 and excludes that lookup
    day. Thus no event on D or D-1 contributes, irrespective of departure time.
    """

    result = data.with_columns(
        (pl.col('_scheduled_departure').dt.date() - pl.duration(days=1)).alias(
            '_history_lookup_day'
        )
    )
    recency_groups = {'origin', 'destination', 'carrier'}
    for group, keys in HISTORY_GROUPS.items():
        parts = daily_parts[group]
        actual = (
            pl.concat(parts, rechunk=True)
            if isinstance(parts, list)
            else parts.rechunk()
        )
        queries = (
            result.select(
                *keys,
                pl.col('_history_lookup_day').alias('_history_day'),
            )
            .unique()
            .with_columns(
                pl.lit(0, dtype=pl.Int64).alias('_flights'),
                pl.lit(0, dtype=pl.Int64).alias('_delayed_flights'),
                pl.lit(0.0).alias('_departure_delay_sum'),
                pl.lit(True).alias('_is_query'),
            )
        )
        combined = (
            pl.concat(
                [
                    actual.with_columns(pl.lit(False).alias('_is_query')),
                    queries,
                ],
                how='vertical_relaxed',
            )
            .group_by(*keys, '_history_day')
            .agg(
                pl.col('_flights').sum(),
                pl.col('_delayed_flights').sum(),
                pl.col('_departure_delay_sum').sum(),
                pl.col('_is_query').max(),
            )
            .sort(*keys, '_history_day')
        )

        expressions: list[pl.Expr] = []
        for label, window in HISTORY_WINDOWS.items():
            flights = (
                pl.col('_flights')
                .rolling_sum_by('_history_day', window_size=window, closed='left')
                .over(keys)
                .fill_null(0)
            )
            delayed = (
                pl.col('_delayed_flights')
                .rolling_sum_by('_history_day', window_size=window, closed='left')
                .over(keys)
                .fill_null(0)
            )
            delay_sum = (
                pl.col('_departure_delay_sum')
                .rolling_sum_by('_history_day', window_size=window, closed='left')
                .over(keys)
                .fill_null(0.0)
            )
            expressions.extend(
                [
                    flights.alias(f'{group}_flights_previous_{label}'),
                    delayed.alias(f'{group}_delayed_flights_previous_{label}'),
                    ((delayed + 10.0) / (flights + 50.0)).alias(
                        f'{group}_delay_rate_previous_{label}'
                    ),
                    pl.when(flights > 0)
                    .then(delay_sum / flights)
                    .otherwise(0.0)
                    .alias(f'{group}_mean_departure_delay_previous_{label}'),
                ]
            )

        if group in recency_groups:
            active_day = (
                pl.when(pl.col('_flights') > 0)
                .then(pl.col('_history_day').cast(pl.Int32))
                .otherwise(None)
                .forward_fill()
                .shift(1)
                .over(keys)
            )
            expressions.append(
                (pl.col('_history_day').cast(pl.Int32) - active_day)
                .fill_null(3650)
                .clip(0, 3650)
                .alias(f'days_since_{group}_previous_active_day')
            )

        feature_names = [
            feature
            for label in HISTORY_WINDOWS
            for feature in (
                f'{group}_flights_previous_{label}',
                f'{group}_delayed_flights_previous_{label}',
                f'{group}_delay_rate_previous_{label}',
                f'{group}_mean_departure_delay_previous_{label}',
            )
        ]
        if group in recency_groups:
            feature_names.append(f'days_since_{group}_previous_active_day')
        lookup = (
            combined.with_columns(expressions)
            .filter('_is_query')
            .select(
                *keys,
                pl.col('_history_day').alias('_history_lookup_day'),
                *feature_names,
            )
        )
        result = result.join(
            lookup,
            on=[*keys, '_history_lookup_day'],
            how='left',
            maintain_order='left',
            validate='m:1',
        )

    return result.drop('_history_lookup_day').with_columns(
        pl.col(HISTORY_FEATURES).fill_null(0.0)
    )


def _splitmix64(values: np.ndarray, seed: int = 0) -> np.ndarray:
    """Stable 64-bit mixing used for label-independent eval subsampling."""

    x = values.astype(np.uint64, copy=True)
    x += np.uint64(0x9E3779B97F4A7C15 + seed)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def deterministic_subsample(
    data: pl.DataFrame, size: int, *, seed: int, sort_result: bool = True
) -> pl.DataFrame:
    """Choose rows by a stable hash of source identity, never by the target."""

    if size >= data.height:
        return data.sort('_scheduled_departure', '_source_id') if sort_result else data
    scores = _splitmix64(data['_source_id'].to_numpy(), seed)
    selected = np.argpartition(scores, size - 1)[:size]
    result = data[selected]
    return result.sort('_scheduled_departure', '_source_id') if sort_result else result


def update_random_reservoir(
    reservoir: pl.DataFrame | None,
    new_rows: pl.DataFrame,
    size: int,
    *,
    seed: int,
) -> pl.DataFrame:
    """Keep the globally smallest stable hashes seen so far.

    This is equivalent to hashing and sampling all rows in one batch, while bounding
    the in-memory train candidate pool at ``size + len(new_rows)``.
    """

    candidates = (
        new_rows
        if reservoir is None
        else pl.concat([reservoir, new_rows], rechunk=False)
    )
    return deterministic_subsample(
        candidates, size, seed=seed, sort_result=False
    ).rechunk()


def _encode_carriers(
    data: pl.DataFrame, train_rows: int
) -> tuple[pl.DataFrame, dict[str, int]]:
    carriers = sorted(
        cast(
            list[str],
            data['operating_carrier'][:train_rows].unique().to_list(),
        )
    )
    mapping = dict(zip(carriers, range(len(carriers)), strict=True))
    return (
        data.with_columns(
            pl.col('operating_carrier')
            .replace_strict(mapping, default=-1, return_dtype=pl.Int64)
            .alias('operating_carrier')
        ),
        mapping,
    )


def _encode_string_categories(
    data: pl.DataFrame, train_rows: int
) -> tuple[pl.DataFrame, dict[str, dict[str, int]]]:
    """Fit string-category vocabularies on train and map eval-only values to -1."""

    mappings: dict[str, dict[str, int]] = {}
    result = data
    for column in STRING_CAT_FEATURES:
        values = sorted(cast(list[str], result[column][:train_rows].unique().to_list()))
        mapping = dict(zip(values, range(len(values)), strict=True))
        mappings[column] = mapping
        result = result.with_columns(
            pl.col(column)
            .replace_strict(mapping, default=-1, return_dtype=pl.Int64)
            .alias(column)
        )
    return result, mappings


def encode_bounded_categories(
    data: pl.DataFrame, train_rows: int, *, max_categories: int = DEFAULT_MAX_CATEGORIES
) -> tuple[pl.DataFrame, dict[str, dict[str, int]], dict[str, object]]:
    """Bound one-hot width without using labels or evaluation vocabularies.

    Keep small vocabularies exact. Hash larger train vocabularies with a stable,
    feature-specific BLAKE2b hash; eval-only/null values map to -1. Zero restores
    the original unbounded recipe. Fit once on the shared base training pool.
    """
    if not 0 < train_rows <= data.height:
        raise ValueError('train_rows must select a nonempty prefix of data')
    if max_categories != 0 and max_categories < 2:
        raise ValueError('max_categories must be 0 (legacy) or at least 2')
    if max_categories == 0:
        encoded, mappings = _encode_string_categories(data, train_rows)
        return encoded, mappings, {'method': 'legacy-unbounded', 'max_categories': 0}

    mappings: dict[str, dict[str, int]] = {}
    stats: dict[str, dict[str, int | bool]] = {}
    expressions = []
    for column in CAT_FEATURES:
        # Canonical strings make metadata mappings JSON-safe for numeric codes too.
        values = sorted(data[column][:train_rows].cast(pl.String).drop_nulls().unique())
        hashed = len(values) >= max_categories
        if hashed:
            mapping = {
                value: int.from_bytes(
                    hashlib.blake2b(
                        json.dumps([column, value], ensure_ascii=False).encode('utf-8'),
                        digest_size=8,
                        person=b'aeolus-cat-v1',
                    ).digest(),
                    'little',
                )
                % (max_categories - 1)
                for value in values
            }
        else:
            mapping = dict(zip(values, range(len(values)), strict=True))
        mappings[column] = mapping
        used = len(set(mapping.values()))
        stats[column] = {
            'train_cardinality': len(values),
            'hashed': hashed,
            'occupied_codes': used,
            'one_hot_upper_bound': used + 1,
            'collisions': len(values) - used,
        }
        expressions.append(
            pl.col(column)
            .cast(pl.String)
            .replace_strict(mapping, default=-1, return_dtype=pl.Int64)
            .alias(column)
        )
    return (
        data.with_columns(expressions),
        mappings,
        {
            'method': 'train-vocabulary-blake2b-v1',
            'max_categories': max_categories,
            'unknown_code': -1,
            'fit_scope': 'shared base train pool only; no labels or eval values',
            'features': stats,
            'categorical_one_hot_upper_bound': sum(
                entry['one_hot_upper_bound'] for entry in stats.values()
            ),
        },
    )


def _part_statistics(data: pl.DataFrame) -> dict[str, object]:
    stats = data.select(
        pl.len().alias('rows'),
        pl.col('_target').sum().alias('positive_rows'),
        pl.col('_target').mean().alias('positive_rate'),
        pl.col('_scheduled_departure').min().alias('time_min'),
        pl.col('_scheduled_departure').max().alias('time_max'),
        pl.col('_source_id').n_unique().alias('unique_source_rows'),
    ).to_dicts()[0]
    for key in ('time_min', 'time_max'):
        stats[key] = stats[key].isoformat()
    return stats


def build_split_features(
    data: pl.DataFrame,
    train_idx: np.ndarray,
    *,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> tuple[pl.DataFrame, dict[str, dict[str, int]], dict[str, object]]:
    """Fit all cross-row features on the shared train pool, preserving row order."""
    train_idx = np.asarray(train_idx)
    if (
        train_idx.ndim != 1
        or not np.issubdtype(train_idx.dtype, np.integer)
        or not len(train_idx)
        or len(np.unique(train_idx)) != len(train_idx)
        or train_idx.min() < 0
        or train_idx.max() >= data.height
    ):
        raise ValueError('Expected nonempty unique in-range train indices')
    # Existing fit helpers use a train prefix. Restore the original table order
    # afterwards, so every split and source-row identity stays unchanged.
    held_out = np.ones(data.height, dtype=bool)
    held_out[train_idx] = False
    order = np.concatenate([train_idx, np.flatnonzero(held_out)])
    result = data[order]
    train_rows = len(train_idx)
    context = result.head(train_rows).select(
        pl.col('_scheduled_departure').alias('CRS_DEP_TIME'),
        pl.col('_history_departure_delay').alias('DEP_DELAY'),
        pl.col('operating_carrier').alias('OP_CARRIER'),
        pl.col('origin_airport').alias('ORIGIN_INDEX'),
        pl.col('destination_airport').alias('DEST_INDEX'),
    )
    histories = daily_history_aggregates(context)
    result = result.drop('_history_departure_delay')
    result = impute_weather_features(result, train_rows)
    result = add_row_level_features(result)
    result = add_weather_climatology_features(result, train_rows)
    result = add_historical_features(result, histories)
    result, mappings, encoding = encode_bounded_categories(
        result, train_rows, max_categories=max_categories
    )
    return result[np.argsort(order)], mappings, encoding


def main(
    raw_data_dir,
    *,
    output_dir=None,
    train_sizes=None,
    train_sampling: TrainSamplingSelection = 'both',
    reference_data_dir=None,
    eval_size: int = DEFAULT_EVAL_SIZE,
    max_train_size: int = DEFAULT_MAX_TRAIN_SIZE,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
):
    if max_train_size <= 0:
        raise ValueError('max_train_size must be positive')
    if max_categories != 0 and max_categories < 2:
        raise ValueError('max_categories must be 0 (legacy) or at least 2')
    if reference_data_dir is not None:
        raise ValueError('Aeolus has no external reference split to restore')
    if train_sizes is None:
        train_sizes = default_train_sizes(DATASET_NAME)

    files = prepare_flight_tab_files(raw_data_dir)
    raw_audit = [audit_raw_file(files[year], year) for year in YEARS]

    def prepare_year(year: int) -> pl.DataFrame:
        return make_model_table(
            _read_year(files[year], year), include_history_delay=True
        )

    # Keep the complete pre-validation pool while preparing features. IID
    # holdouts are removed from the shared model-train pool before imposing the
    # training-size budget, and latest must span the real history.
    train = pl.concat(
        [prepare_year(year) for year in range(YEARS[0], TRAIN_END_YEAR + 1)],
        rechunk=True,
    ).sort('_scheduled_departure', '_source_id')

    validation = deterministic_subsample(
        prepare_year(VALIDATION_YEAR),
        eval_size,
        seed=1,
    )
    test = deterministic_subsample(
        prepare_year(TEST_YEAR),
        eval_size,
        seed=2,
    )

    train_time_min = cast(datetime, train['_scheduled_departure'].min())
    train_time_max = cast(datetime, train['_scheduled_departure'].max())
    validation_time_min = cast(datetime, validation['_scheduled_departure'].min())
    test_time_min = cast(datetime, test['_scheduled_departure'].min())
    if not (train_time_max < validation_time_min < test_time_min):
        raise RuntimeError('Aeolus temporal split boundaries overlap')

    parts = {'train': train, 'val': validation, 'test': test}
    part_statistics = {name: _part_statistics(part) for name, part in parts.items()}
    data = pl.concat(list(parts.values()), rechunk=True)
    train_end = train.height
    validation_end = train_end + validation.height
    iid_split = make_iid_split(
        np.arange(train_end, dtype=np.int32),
        val_size=validation.height,
        test_size=test.height,
    )
    split = {
        'train_idx': iid_split['train_idx'],
        'val_idx': np.arange(train_end, validation_end, dtype=np.int32),
        'test_idx': np.arange(validation_end, data.height, dtype=np.int32),
    }
    logger.info('Building features after removing IID holdouts from the fit pool')
    data, categorical_mappings, categorical_encoding = build_split_features(
        data, iid_split['train_idx'], max_categories=max_categories
    )
    missing_features = (set(CAT_FEATURES) | set(NUM_FEATURES)) - set(data.columns)
    if missing_features:
        raise RuntimeError(f'Missing engineered features: {sorted(missing_features)}')

    data_parts = {
        'X_cat': data.select(CAT_FEATURES),
        'X_num': data.select(NUM_FEATURES).cast(pl.Float32),
        'X_meta': data.select(
            pl.col('_scheduled_departure')
            .dt.epoch('us')
            .alias('scheduled_departure_unix_us'),
            pl.col('_source_id').alias('source_row_id'),
        ),
        'Y': data.select(pl.col('_target').alias('departure_delay_over_15m')),
    }

    metadata = {
        'source': {
            'name': 'Aeolus: A Multi-structural Flight Delay Dataset',
            'kaggle_url': (
                'https://www.kaggle.com/datasets/flnny123/'
                'mfddmulti-modal-flight-delay-dataset'
            ),
            'raw_years': list(YEARS),
            'used_modalities': ['Flight_Tab'],
            'raw_audit': raw_audit,
        },
        'prediction_context': (
            'schedule plus Aeolus temporally aligned historical weather; before '
            'actual flight-operation outcomes'
        ),
        'target': {
            'source_column': 'DEP_DELAY',
            'positive_definition': 'DEP_DELAY > 15 minutes',
            'threshold_minutes': 15,
            'strict_inequality': True,
        },
        'feature_names': {
            'categorical': CAT_FEATURES,
            'numeric': NUM_FEATURES,
            'metadata': META_FEATURES,
        },
        'categorical_mappings': categorical_mappings,
        'categorical_encoding': categorical_encoding,
        'carrier_mapping': categorical_mappings['operating_carrier'],
        'feature_recipe': {
            'name': (
                'domain-engineered-150-bounded-categories-train-only-v2'
                if max_categories
                else 'domain-engineered-150-train-only-v2'
            ),
            'total_model_features': len(CAT_FEATURES) + len(NUM_FEATURES),
            'categorical_features': len(CAT_FEATURES),
            'numeric_features': len(NUM_FEATURES),
            'blocks': {
                'base_categorical': len(BASE_CAT_FEATURES),
                'engineered_categorical': len(ENGINEERED_CAT_FEATURES),
                'base_numeric': len(BASE_NUM_FEATURES),
                'calendar': len(CALENDAR_FEATURES),
                'geometry': len(GEOMETRY_FEATURES),
                'weather_interactions': len(WEATHER_INTERACTION_FEATURES),
                'weather_climatology': len(WEATHER_CLIMATOLOGY_FEATURES),
                'causal_history': len(HISTORY_FEATURES),
            },
        },
        'leakage_controls': {
            'excluded_post_cutoff_columns': EXCLUDED_LEAKAGE_COLUMNS,
            'included_aeolus_weather_columns': WEATHER_COLUMNS,
            'weather_scope_note': (
                'Aeolus bundles historical observations, not operational weather '
                'forecasts; use this task as the released benchmark rather than as '
                'a deployable pre-departure forecast simulation.'
            ),
            'target_or_label_derived_features': [
                name
                for name in HISTORY_FEATURES
                if 'delayed_flights' in name
                or 'delay_rate' in name
                or 'mean_departure_delay' in name
            ],
            'history_feature_context': (
                'Only the shared 2016-2022 train pool after removing IID holdouts '
                'provides history. No IID or temporal holdout outcomes are used. '
                'For a flight on day D, histories exclude D and D-1. This fixed '
                'context is shared by all train-size variants and is additional '
                'to their supervised training-row budget. Histories are not '
                'updated during temporal evaluation.'
            ),
            'history_windows': list(HISTORY_WINDOWS),
            'history_groups': HISTORY_GROUPS,
            'history_rate_smoothing': 'Beta prior: 10 delayed / 50 flights',
            'weather_climatology_fit': 'shared train pool after removing IID holdouts',
            'feature_fit_scope': 'shared train pool after removing IID holdouts',
            'eval_subsampling_key': 'stable hash(source_row_id), independent of y',
            'train_pool_subsetting': (
                'all valid 2016-2022 rows; IID holdouts removed before size sampling'
            ),
            'size_variant_subsetting': (
                'nested random/latest from one train pool shared by protocols'
            ),
        },
        'temporal_split': {
            'train_source_years': [
                train_time_min.year,
                TRAIN_END_YEAR,
            ],
            'validation_source_year': VALIDATION_YEAR,
            'test_source_year': TEST_YEAR,
            'part_statistics': part_statistics,
        },
    }

    logger.info('Writing Aeolus arrays and split variants')
    save_dataset_variants(
        name=DATASET_NAME,
        task_type='binclass',
        data=data_parts,
        splits={
            'time': split,
            'iid': iid_split,
        },
        default_protocol='time',
        train_sizes=train_sizes,
        train_sampling=train_sampling,
        output_dir=output_dir,
        metadata=metadata,
    )


if __name__ == '__main__':
    args = parse_args(DATASET_NAME)
    main(
        args.raw_data_dir,
        output_dir=args.output_dir,
        train_sizes=args.train_size,
        train_sampling=args.train_sampling,
        reference_data_dir=args.reference_data_dir,
        max_categories=args.max_categories,
    )
