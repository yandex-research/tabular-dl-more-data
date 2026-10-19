import enum
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import lib.metrics
import lib.utils.misc
from lib.types import DataKey, PartKey

# >>> Data types


X_NUM_DTYPE = np.float32
X_BIN_DTYPE = np.float32
X_CAT_INT_DTYPE = np.int64
X_CAT_STR_DTYPE = np.str_
Y_REG_DTYPE = np.float32
Y_CLF_DTYPE = np.int64
SPLIT_DTYPE = np.int32


# >>> Utilities


def _check_dataset_dir(dataset_dir: str | Path) -> Path:
    if isinstance(dataset_dir, str):
        dataset_dir = dataset_dir.removeprefix(':')
        dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise RuntimeError(f'The dataset does not exist: {dataset_dir}')
    return dataset_dir


def _is_npy_path(path: Path) -> bool:
    return path.suffix == '.npy' and not path.is_dir()


# >>> Split


# For a given dataset, splits are stored in the `splits/` directory. This directory
# has a tree-like layout representing different split parts, with some splits
# potentially sharing some of the parts. Consider the following example:
#
# ```
# splits/
#     a/
#         test.npy
#         0/
#             train.npy
#             val.npy
#         1/
#             train.npy
#             val.npy
#     b/
#         test.npy
#         train.npy
#         val.npy
# ```
#
# The above layout represents _three_ splits, where:
#
# - Two splits are stored in the a/ subdirectory and have the same test indices,
#   but different test and validation indices.
# - The third split is stored in the b/ directory and does not share any of its parts
#   with other splits.
#
# Thus, a split is fully identified by a sequence of nested directories that must be
# traversed to collect all of its parts. Formally, in the above example,
# the three splits have the following IDs:
#
# 1. ("a", "0")
# 2. ("a", "1")
# 3. ("b",)
type SplitDict = dict[PartKey, np.ndarray]
type SplitID = tuple[str, ...]

# Some parts of the API also accept `list[str]` as a split ID to make it possible
# to read a split ID from a text file (e.g. in a JSON format, where tuples
# cannot be represented) and pass it as-is.
type SplitIDLike = list[str] | SplitID


def make_split_id(split_id: SplitIDLike) -> SplitID:
    if not isinstance(split_id, tuple | list):
        raise TypeError('split_id must be either a tuple or a list')
    return tuple(split_id)


def split_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """Options shared by feature, label and query-ID loading."""
    return {
        key: config[key] for key in ('train', 'val_key', 'test_key') if key in config
    }


def evaluation_parts(
    labels: Mapping[str, Any], requested: Any = ('val', 'test')
) -> list[str]:
    """Evaluate every available holdout for the independent split format."""
    if any(key.startswith(('val_', 'test_')) for key in labels):
        return [key for key in labels if key != 'train' or 'train' in requested]
    return list(requested)


def load_split(
    dataset_dir: str | Path,
    split_id: SplitIDLike = ('default',),
    *,
    train: Mapping[str, Any] | None = None,
    val_key: str | None = None,
    test_key: str | None = None,
) -> SplitDict:
    dataset_dir = _check_dataset_dir(dataset_dir)
    if (dataset_dir / 'splits/train').is_dir():
        if train is None or val_key is None or test_key is None:
            raise ValueError('Independent splits require train, val_key and test_key')
        if tuple(split_id) != ('default',):
            raise ValueError(
                'Use train/evaluation options instead of split_id '
                'for independent splits'
            )
        if set(train) != {'sampling', 'size'} or train['sampling'] not in (
            'random',
            'latest',
            'query',
        ):
            raise ValueError(
                'train must contain sampling (random/latest/query) and size'
            )
        size = str(train['size'])
        if size != 'full' and (not size.isdecimal() or int(size) <= 0):
            raise ValueError('train.size must be a positive integer or full')
        eval_root = dataset_dir / 'splits/eval'
        protocols = sorted(p.name for p in eval_root.iterdir() if p.is_dir())
        for selected in (val_key, test_key):
            if selected not in protocols:
                raise ValueError(
                    f'Unknown evaluation protocol: {selected!r}; available: {protocols}'
                )
        paths = {
            'train': dataset_dir / 'splits/train' / train['sampling'] / f'{size}.npy'
        }
        for part, selected in (('val', val_key), ('test', test_key)):
            for protocol in protocols:
                key = part if protocol == selected else f'{part}_{protocol}'
                paths[key] = eval_root / protocol / f'{part}.npy'
        result = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
        if any(
            value.dtype != SPLIT_DTYPE or value.ndim != 1 for value in result.values()
        ):
            raise ValueError('Split indices must be one-dimensional int32 arrays')
        return result
    if train is not None or val_key is not None or test_key is not None:
        raise ValueError(
            'Independent train/evaluation options require the new dataset layout'
        )
    if not split_id:
        raise ValueError('split_id must be non-empty')

    split_id = make_split_id(split_id)
    split = {}
    directory = dataset_dir / 'splits'
    for dirname in split_id:
        directory = directory / dirname
        if not directory.is_dir():
            raise RuntimeError(
                'split_id must be a sequence of existing nested directory names'
                ' relative to the dataset directory. However, this is not the case for'
                f' "{dirname}"'
            )
        for path in directory.iterdir():
            if _is_npy_path(path):
                part = path.stem
                if part in split:
                    raise RuntimeError(
                        f'The part "{part}" is presented more than once'
                        f' in the split {split_id}'
                    )
                value = np.load(path)
                if value.dtype != SPLIT_DTYPE:
                    raise RuntimeError(
                        f'Split data type must be {SPLIT_DTYPE} not {value.dtype}'
                    )
                split[part] = value
    return split


def apply_split(data: Any, split: SplitDict) -> Any:
    """Split a NumPy array or a container holding NumPy arrays.

    * If `data` is `np.ndarray`, the function returns `dict[PartKey, np.ndarray]`
      based on the provided `split`.
    * If `data` is a (nested) mapping, then the function traverses `data` and replaces
      all `np.ndarray`s with `dict[PartKey, np.ndarray]` using the provided `split`.
    """
    if isinstance(data, np.ndarray):
        split_data = {}
        for part_name, idx in split.items():
            # Fancy-indexing a large random row subset from a multi-GB array is a
            # cache-hostile operation (tens of seconds on the 40M-row Avazu x4
            # datasets with random reczoo splits). Two exact optimizations:
            #   * contiguous index ranges are converted to slices, which yields a
            #     zero-copy view (and keeps the row order);
            #   * non-contiguous ranges are sorted before gathering, which turns
            #     the random access into a near-sequential read. Training and
            #     evaluation are agnostic to the row order, and every part is
            #     gathered with the same sorted indices, so train/val/test stay
            #     mutually aligned.
            if (
                idx.size > 0
                and (idx[-1] - idx[0] + 1) == idx.size
                and (np.diff(idx) == 1).all()
            ):
                index = slice(int(idx[0]), int(idx[-1]) + 1)
                split_data[part_name] = data[index]
            else:
                if idx.size > 1 and not (np.diff(idx) >= 0).all():
                    idx = np.sort(idx)
                split_data[part_name] = data[idx]
        return split_data
    elif isinstance(data, Mapping):
        return type(data)(
            (k, apply_split(v, split))
            for k, v in data.items()  # pyright: ignore[reportCallIssue]
        )
    else:
        raise TypeError(f'Unsupported type of `data`: {type(data)}')


# >>> Info


@enum.unique
class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'

    @property
    def is_binclass(self) -> bool:
        return self == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self == TaskType.REGRESSION


@enum.unique
class TaskScore(enum.Enum):
    ACCURACY = 'accuracy'
    AVERAGE_PRECISION = 'average-precision'
    CROSS_ENTROPY = 'cross-entropy'
    MAE = 'mae'
    R2 = 'r2'
    RMSE = 'rmse'
    ROC_AUC = 'roc-auc'

    @property
    def higher_is_better(self) -> bool:
        return {
            TaskScore.ACCURACY: True,
            TaskScore.AVERAGE_PRECISION: True,
            TaskScore.CROSS_ENTROPY: False,
            TaskScore.MAE: False,
            TaskScore.R2: True,
            TaskScore.RMSE: False,
            TaskScore.ROC_AUC: True,
        }[self]


def load_info(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_dir = _check_dataset_dir(dataset_dir)
    return json.loads(dataset_dir.joinpath('info.json').read_text())


# >>> Data


def load_data(
    dataset_dir: str | Path,
    split_id: SplitIDLike = ('default',),
    *,
    keys: list[DataKey] | None = None,
    train: Mapping[str, Any] | None = None,
    val_key: str | None = None,
    test_key: str | None = None,
) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    # >>> Load the data.
    dataset_dir = _check_dataset_dir(dataset_dir)
    data = {
        x.stem: np.load(x, mmap_mode='r')
        for x in (
            dataset_dir / 'arrays' if (dataset_dir / 'arrays').is_dir() else dataset_dir
        ).iterdir()
        if _is_npy_path(x) and (keys is None or x.stem in keys)
    }
    split = load_split(
        dataset_dir,
        split_id,
        train=train,
        val_key=val_key,
        test_key=test_key,
    )
    data = apply_split(data, split)

    # >>> Validate the data.
    task_type = TaskType(load_info(dataset_dir)['task']['type'])
    for key, expected_dtypes in [
        ('x_num', (X_NUM_DTYPE,)),
        ('x_bin', (X_BIN_DTYPE,)),
        ('x_cat', (X_CAT_INT_DTYPE, X_CAT_STR_DTYPE)),
        ('y', (Y_REG_DTYPE if task_type.is_regression else Y_CLF_DTYPE,)),
    ]:
        subdata = data.get(key)
        if subdata is not None and not any(
            all(
                (
                    isinstance(x.dtype, np.dtypes.StrDType)
                    if expected_dtype is np.str_
                    else x.dtype == expected_dtype
                )
                for x in subdata.values()
            )
            for expected_dtype in expected_dtypes
        ):
            raise RuntimeError(
                f'Invalid data type of "{key}".'
                f' Expected one of: {expected_dtypes}.'
                f' Actual: {next(iter(subdata.values())).dtype}'
            )

    return data


# >>> High-level API


DEFAULT_SPLIT_ID: SplitID = ('default',)


@enum.unique
class NumTransform(enum.Enum):
    STANDARD = 'standard'
    NOISY_QUANTILE = 'noisy-quantile'


@enum.unique
class CatTransform(enum.Enum):
    ONE_HOT = 'one-hot'


def _extract_binary_features(
    x: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray] | None, dict[PartKey, np.ndarray] | None]:
    n_features = x['train'].shape[1]
    binary_idx, high_values = [], []

    for j in range(n_features):
        # Reject obvious non-binary features before scanning all splits.
        sample = np.unique(x['train'][:16_384, j])
        if len(sample) > 2 or np.isnan(sample).any():
            continue

        columns = [v[:, j] for v in x.values() if len(v)]
        if not columns:
            continue
        lo = np.array([col.min() for col in columns]).min()
        hi = np.array([col.max() for col in columns]).max()
        if np.isnan(lo) or np.isnan(hi) or lo == hi:
            continue

        # Exactly two distinct values across all splits, with no NaNs.
        if all(((col == lo) | (col == hi)).all() for col in columns):
            binary_idx.append(j)
            high_values.append(hi)

    if not binary_idx:
        return None, x

    remaining_mask = np.ones(n_features, dtype=bool)
    remaining_mask[binary_idx] = False
    x_binary = {}
    for part, values in x.items():
        encoded = np.empty((len(values), len(binary_idx)), dtype=X_BIN_DTYPE)
        for k, (j, hi) in enumerate(zip(binary_idx, high_values)):
            encoded[:, k] = values[:, j] == hi
        x_binary[part] = encoded

    x_remaining = (
        {part: values[:, remaining_mask] for part, values in x.items()}
        if remaining_mask.any()
        else None
    )

    return x_binary, x_remaining


type DatasetFingerprint = tuple[tuple[str, int, int], ...]


def _get_dataset_fingerprint(
    dataset_dir: str | Path,
) -> tuple[Path, DatasetFingerprint]:
    """Fingerprint the files that affect ``build_data`` without reading their data."""
    dataset_dir = _check_dataset_dir(dataset_dir).resolve()
    paths = sorted(
        [dataset_dir / 'info.json', *dataset_dir.rglob('*.npy')],
        key=lambda x: str(x.relative_to(dataset_dir)),
    )
    fingerprint = tuple(
        (
            str(path.relative_to(dataset_dir)),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in paths
        if path.exists()
    )
    return dataset_dir, fingerprint


@lib.utils.misc.cache_on_disk
def _build_data_cached(
    dataset_dir: str | Path,
    split_id: SplitIDLike = DEFAULT_SPLIT_ID,
    *,
    num_transform: str | NumTransform | None = None,
    cat_transform: str | CatTransform | None = None,
    seed: int = 0,
    dataset_fingerprint: DatasetFingerprint,
    train: Mapping[str, Any] | None = None,
    val_key: str | None = None,
    test_key: str | None = None,
) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    # This value is deliberately part of the cache key. The actual loader still
    # reads through `dataset_dir`.
    del dataset_fingerprint
    import sklearn.impute
    import sklearn.preprocessing

    import lib.preprocessing

    import time

    stage_time = time.perf_counter()

    def _log_stage(stage: str) -> None:
        nonlocal stage_time
        now = time.perf_counter()
        print(f'[data] {stage} in {now - stage_time:.1f}s', flush=True)
        stage_time = now

    data = load_data(
        dataset_dir,
        split_id,
        train=train,
        val_key=val_key,
        test_key=test_key,
    )
    _log_stage('loaded data')

    # The code below always accesses the `data` content through indexing
    # without creating variables like `x_num = data['x_num']`,
    # because the data content changes multiple times throughout the pipeline.

    if 'x_num' in data:
        # Extract binary features from numeric features.
        x_bin_extracted, x_num_remaining = _extract_binary_features(data['x_num'])
        _log_stage('extracted binary features')
        if x_bin_extracted is not None:
            if x_num_remaining is None:
                del data['x_num']
            else:
                data['x_num'] = x_num_remaining

            if 'x_bin' in data:
                assert x_bin_extracted.keys() == data['x_bin'].keys()
                data['x_bin'] = {
                    k: np.concatenate([x_bin_extracted[k], data['x_bin'][k]], axis=-1)
                    for k in x_bin_extracted
                }
            else:
                data['x_bin'] = x_bin_extracted
        del x_bin_extracted, x_num_remaining

    # Numeric features may be gone after the binary feature extraction,
    # so their presence is checked again.
    if 'x_num' in data:
        if num_transform is not None:
            # Transform.
            num_transform = NumTransform(num_transform)
            if num_transform == NumTransform.STANDARD:
                num_transformer = sklearn.preprocessing.StandardScaler(copy=False)
            elif num_transform == NumTransform.NOISY_QUANTILE:
                num_transformer = lib.preprocessing.NoisyQuantileTransformer(
                    n_quantiles=max(
                        min(data['x_num']['train'].shape[0] // 30, 1000), 10
                    ),
                    output_distribution='normal',
                    subsample=1_000_000_000,
                    random_state=seed,
                    copy=False,
                )
            else:
                raise ValueError(f'Unknown num_transform: {num_transform}')
            data['x_num'] = lib.preprocessing.fit_transform_dict(
                num_transformer, data['x_num']
            )

        # Fill missing values.
        data['x_num'] = lib.preprocessing.fit_transform_dict(
            sklearn.impute.SimpleImputer(strategy='constant', fill_value=0.0),
            data['x_num'],
        )
        _log_stage('transformed/imputed x_num')

    else:
        if num_transform is not None:
            raise ValueError(
                'num_transform must be None,'
                ' because the data is missing numeric features'
            )

    if 'x_bin' in data:
        # Fill missing values.
        data['x_bin'] = lib.preprocessing.fit_transform_dict(
            sklearn.impute.SimpleImputer(strategy='constant', fill_value=2.0),
            data['x_bin'],
        )

        # Convert to categorical features.
        if 'x_cat' in data:
            data['x_bin'] = {
                k: v.astype(data['x_cat']['train'].dtype)
                for k, v in data['x_bin'].items()
            }
            data['x_cat'] = {
                part: np.column_stack([data['x_cat'][part], data['x_bin'][part]])
                for part in data['x_cat']
            }
        else:
            data['x_cat'] = {
                k: v.astype(X_CAT_INT_DTYPE) for k, v in data['x_bin'].items()
            }
        del data['x_bin']

    if 'x_cat' in data:
        # Encode.
        data['x_cat'] = lib.preprocessing.fit_transform_dict(
            lib.preprocessing.OrdinalEncoderWithAdaptiveUnknowns(dtype=X_CAT_INT_DTYPE),
            data['x_cat'],
        )
        _log_stage('encoded x_cat')

        # Transform.
        if cat_transform is not None:
            cat_transform = CatTransform(cat_transform)
            if cat_transform == CatTransform.ONE_HOT:
                cat_transformer = sklearn.preprocessing.OneHotEncoder(
                    handle_unknown='ignore', sparse_output=False, dtype=X_NUM_DTYPE
                )
            else:
                raise ValueError(f'Unknown cat_transform: {cat_transform}')
            data['x_cat'] = lib.preprocessing.fit_transform_dict(
                cat_transformer, data['x_cat']
            )
    else:
        if cat_transform is not None:
            raise ValueError(
                'cat_transform must be None,'
                ' because the data is missing categorical features'
            )

    for key in ['x_num', 'x_bin', 'x_cat']:
        if key in data:
            data[key] = lib.preprocessing.fit_transform_dict(
                lib.preprocessing.DropConstantFeatures(), data[key]
            )

    return data


def build_data(
    dataset_dir: str | Path,
    split_id: SplitIDLike = DEFAULT_SPLIT_ID,
    *,
    num_transform: str | NumTransform | None = None,
    cat_transform: str | CatTransform | None = None,
    seed: int = 0,
    train: Mapping[str, Any] | None = None,
    val_key: str | None = None,
    test_key: str | None = None,
) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    """Build model inputs, invalidating the disk cache when dataset files change."""
    dataset_dir, dataset_fingerprint = _get_dataset_fingerprint(dataset_dir)
    return _build_data_cached(
        dataset_dir,
        split_id,
        num_transform=num_transform,
        cat_transform=cat_transform,
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
        train=train,
        val_key=val_key,
        test_key=test_key,
    )
