# Preprocessing - common functions
import argparse
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import polars as pl

from lib import env
from lib.data import TaskType


TIME_IID_DATASETS = frozenset(
    {
        'cooking-time',
        'delivery-eta',
        'maps-routing',
        'weather',
        'jane-street',
        'aeolus-flight-delay',
    }
)


PROJECT_DIR = env.get_project_dir()
DATA_DIR = env.get_data_dir()
PARTS = ('train', 'val', 'test')
TRAIN_SIZE_GRID = (
    10_000,
    20_000,
    40_000,
    80_000,
    160_000,
    320_000,
    640_000,
    1_280_000,
    2_560_000,
    5_120_000,
    None,
)
TRAIN_SIZE_LABELS = tuple(
    'full' if size is None else str(size) for size in TRAIN_SIZE_GRID
)
TRAIN_SAMPLING_ALGORITHM = 'nested-permutation-v1'
TRAIN_SIZE_PRESETS = {
    'small': 100_000,
    'medium': 1_000_000,
    'big': 10_000_000,
    'full': None,
}
TrainSampling = Literal['random', 'latest']
TrainSamplingSelection = Literal['random', 'latest', 'both']
Split = dict[str, np.ndarray]


def clean_independent_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Keep descriptive metadata; split files describe their own sizes/layout."""
    redundant = {
        'id',
        'task_type',
        'score',
        'train_size',
        'val_size',
        'test_size',
        'split_variants',
        'split_protocols',
        'default_protocol',
        'train_variants',
        'evaluation_protocols',
        'full_train_available',
    }
    return {key: value for key, value in info.items() if key not in redundant}


def default_train_size_grid(dataset_name: str) -> tuple[int | None, ...]:
    return TRAIN_SIZE_GRID[:-1] if dataset_name == 'maps-routing' else TRAIN_SIZE_GRID


def default_train_sizes(dataset_name: str) -> list[str]:
    if dataset_name not in TIME_IID_DATASETS:
        return ['small']
    return [
        'full' if size is None else str(size)
        for size in default_train_size_grid(dataset_name)
    ]


def make_iid_split(
    train_pool: np.ndarray, *, val_size: int, test_size: int, seed: int = 0
) -> Split:
    """Hold out IID evaluation rows from a pool, before train-size sampling.

    Only row identity is used. The remaining pool and held-out parts retain their
    input order, so temporal train subsampling remains well-defined.
    """
    train_pool = np.asarray(train_pool)
    if train_pool.ndim != 1 or not np.issubdtype(train_pool.dtype, np.integer):
        raise ValueError('The IID train pool must be a one-dimensional integer array')
    if len(np.unique(train_pool)) != len(train_pool):
        raise ValueError('The IID train pool contains duplicate row indices')
    if val_size <= 0 or test_size <= 0:
        raise ValueError('IID validation and test sizes must be positive')
    if val_size + test_size >= len(train_pool):
        raise ValueError('IID evaluation must leave at least one training row')
    positions = np.random.default_rng(seed).choice(
        len(train_pool), val_size + test_size, replace=False
    )
    keep = np.ones(len(train_pool), dtype=bool)
    keep[positions] = False
    return {
        'train_idx': train_pool[keep],
        'val_idx': train_pool[np.sort(positions[:val_size])],
        'test_idx': train_pool[np.sort(positions[val_size:])],
    }


def make_time_iid_splits(
    data: pl.DataFrame,
    *,
    dataset_name: str,
    time_column: str,
    val_start: datetime,
    test_start: datetime,
    test_end: datetime | None = None,
    reference_data_dir: str | Path | None = None,
    reference_index_column: int = 1,
    eval_fraction: float = 0.025,
    seed: int = 0,
) -> dict[str, Split]:
    """Build temporal and IID protocols on the same chronologically ordered table.

    Temporal evaluation preserves TabReD's source-order subsample (or reference
    membership and order). IID evaluation is sampled from the pre-validation
    rows with the same evaluation sizes. The sampled IID evaluation rows are
    removed from the single train pool shared by both protocols.
    """
    if not val_start < test_start or (test_end is not None and test_end <= test_start):
        raise ValueError('Temporal boundaries must be strictly increasing')
    if not 0 < eval_fraction <= 1:
        raise ValueError('eval_fraction must be in (0, 1]')
    timestamps = data[time_column]
    if timestamps.null_count() or not timestamps.is_sorted():
        raise ValueError(
            'The model table must be sorted by time without null timestamps'
        )

    rows = data.select(time_column, 'index_in_full').with_row_index('_row')
    time = pl.col(time_column)
    temporal = {'train_idx': rows.filter(time < val_start)['_row'].to_numpy()}
    # Restoring source order on IDs alone avoids sorting/copying feature columns.
    # This reproduces sampling before the recipes' chronological sort and after
    # their target filtering, even when source IDs have gaps.
    evaluation_ids = (
        rows.select('index_in_full')
        .sort('index_in_full')
        .sample(fraction=eval_fraction, seed=seed)['index_in_full']
    )
    periods = {
        'val': (time >= val_start) & (time < test_start),
        'test': (time >= test_start)
        & (pl.lit(True) if test_end is None else time < test_end),
    }
    for part, period in periods.items():
        indices = get_reference_split_idx(
            data,
            dataset_name,
            part,
            reference_data_dir=reference_data_dir,
            index_column=reference_index_column,
        )
        if indices is None:
            indices = rows.filter(
                period & pl.col('index_in_full').is_in(evaluation_ids.implode())
            )['_row'].to_numpy()
        elif not rows[indices].select(period.all()).item():
            raise ValueError(f'Reference {part} rows are outside the temporal interval')
        temporal[f'{part}_idx'] = indices

    iid = make_iid_split(
        temporal['train_idx'],
        val_size=len(temporal['val_idx']),
        test_size=len(temporal['test_idx']),
        seed=seed,
    )
    temporal['train_idx'] = iid['train_idx']
    return {'time': temporal, 'iid': iid}


def parse_args(dataset_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f'Preprocess {dataset_name} from local raw files. The input can be a '
            'dataset directory, a root directory containing the dataset directory, or '
            'a Kaggle zip archive already downloaded locally.'
        )
    )
    parser.add_argument(
        'raw_data_dir',
        type=Path,
        help='Local directory/archive with raw dataset files. Nothing is downloaded.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DATA_DIR,
        help='Where to write the processed dataset directory (default: ./data).',
    )
    parser.add_argument(
        '--train-size',
        nargs='+',
        default=default_train_sizes(dataset_name),
        metavar='SIZE',
        help=(
            'Dataset version(s) to write: small=100k, medium=1m, big=10m, '
            'full=all available train rows, or an integer. '
            'TabReD defaults to doubling sizes from 10k to 5.12m plus full; '
            'other datasets default to small.'
        ),
    )
    parser.add_argument(
        '--train-sampling',
        choices=['random', 'latest', 'both'],
        default='both',
        help=('Which train subsampling variants to write. Defaults to both.'),
    )
    parser.add_argument(
        '--reference-data-dir',
        type=Path,
        default=None,
        help=(
            'Optional directory with reference processed datasets. If provided, '
            'default val/test splits are reproduced from '
            '<reference-data-dir>/<dataset>/splits/default/{val,test}.npy '
            'and x_meta.npy.'
        ),
    )
    if dataset_name == 'aeolus-flight-delay':
        parser.add_argument(
            '--max-categories',
            type=int,
            default=512,
            help='Aeolus category limit, including unknown; 0 keeps legacy IDs.',
        )
    return parser.parse_args()


def _candidate_roots(raw_path: Path, dataset_name: str) -> list[Path]:
    names = {
        dataset_name,
        dataset_name.replace('-', '_'),
        dataset_name.replace('_', '-'),
    }
    roots = [raw_path]
    if raw_path.is_dir():
        roots += [raw_path / name for name in sorted(names)]
    return [x.resolve() for x in roots if x.exists()]


def _requirement_exists(base: Path, requirement: str) -> bool:
    return (
        bool(list(base.glob(requirement)))
        if any(c in requirement for c in '*?[')
        else (base / requirement).exists()
    )


def _has_requirements(base: Path, requirements: list[str]) -> bool:
    return all(_requirement_exists(base, requirement) for requirement in requirements)


def _find_requirement(roots: list[Path], requirement: str) -> Path | None:
    if any(c in requirement for c in '*?['):
        for root in roots:
            matches = list(root.glob(requirement))
            if matches:
                return matches[0]
        return None

    name = Path(requirement).name
    for root in roots:
        direct = root / requirement
        if direct.exists():
            return direct
        if root.is_dir():
            matches = list(root.rglob(name))
            if matches:
                return matches[0]
    return None


def _find_archive(
    raw_path: Path, roots: list[Path], archive_names: list[str]
) -> Path | None:
    if raw_path.is_file() and raw_path.suffix == '.zip':
        return raw_path

    for root in roots:
        for archive_name in archive_names:
            direct = root / archive_name
            if direct.exists():
                return direct
            if root.is_dir():
                matches = list(root.rglob(archive_name))
                if matches:
                    return matches[0]
    return None


def prepare_data_dir(
    dataset_name: str,
    raw_data_dir: str | Path,
    *,
    requirements: list[str],
    archive_names: list[str] | None = None,
) -> Path:
    """Return a local directory with raw files, extracting a local zip if needed.

    This function never downloads anything. It accepts either already extracted raw
    files or a local zip archive. If an archive is extracted, the extracted files are
    cached under preprocessing/tmp/<dataset_name>.
    """

    raw_path = Path(raw_data_dir).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f'Raw data path does not exist: {raw_path}')

    roots = _candidate_roots(raw_path, dataset_name)
    for root in roots:
        if root.is_dir() and _has_requirements(root, requirements):
            return root

    # If a root directory contains an arbitrary nested dataset directory, infer
    # the dataset root from the first required file and verify the rest there.
    if requirements and not any(c in requirements[0] for c in '*?['):
        n_parts = len(Path(requirements[0]).parts)
        for root in roots:
            found = _find_requirement([root], requirements[0])
            if found is not None:
                base = found
                for _ in range(n_parts):
                    base = base.parent
                if _has_requirements(base, requirements):
                    return base

    archive = _find_archive(raw_path, roots, archive_names or [])
    if archive is not None:
        tmp_path = PROJECT_DIR / 'preprocessing' / 'tmp' / dataset_name
        tmp_path.mkdir(parents=True, exist_ok=True)
        marker = tmp_path / f'.extracted-{archive.name}'
        if not marker.exists() or marker.read_text() != str(archive):
            with zipfile.ZipFile(archive, 'r') as z:
                z.extractall(tmp_path)
            marker.write_text(str(archive))

        # Some datasets need an extra local file next to the archive (for example,
        # Sberbank's BAD_ADDRESS_FIX.xlsx). Copy such files into the extracted dir.
        for requirement in requirements:
            if not _requirement_exists(tmp_path, requirement):
                found = _find_requirement(roots, requirement)
                if (
                    found is not None
                    and found.is_file()
                    and not any(c in requirement for c in '*?[')
                ):
                    destination = tmp_path / requirement
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(found, destination)

        if _has_requirements(tmp_path, requirements):
            return tmp_path

    tried = ', '.join(str(x) for x in roots)
    missing = ', '.join(requirements)
    raise FileNotFoundError(
        f'Could not find raw files for {dataset_name}. '
        f'Missing requirements: {missing}. '
        f'Tried: {tried}'
    )


def _xlsx_column_to_index(cell_reference: str) -> int:
    letters = re.match(r'[A-Z]+', cell_reference)
    if letters is None:
        raise ValueError(f'Invalid XLSX cell reference: {cell_reference}')
    index = 0
    for char in letters.group(0):
        index = index * 26 + ord(char) - ord('A') + 1
    return index - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    namespace = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    cell_type = cell.attrib.get('t')

    if cell_type == 'inlineStr':
        text = cell.findtext('x:is/x:t', default='', namespaces=namespace)
        return text

    raw_value = cell.findtext('x:v', default=None, namespaces=namespace)
    if raw_value is None:
        return None
    if cell_type == 's':
        return shared_strings[int(raw_value)]
    if cell_type == 'b':
        return bool(int(raw_value))

    try:
        value = float(raw_value)
    except ValueError:
        return raw_value
    return int(value) if value.is_integer() else value


def read_xlsx(
    path: str | Path, *, null_values: tuple[str, ...] = ('NA', '')
) -> pl.DataFrame:
    """Read a simple first-sheet XLSX file without optional Excel dependencies.

    This is intentionally small and is used for Sberbank's BAD_ADDRESS_FIX.xlsx.
    It supports plain values and shared strings, which is enough for this dataset.
    """

    path = Path(path)
    namespace = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if 'xl/sharedStrings.xml' in workbook.namelist():
            root = ET.fromstring(workbook.read('xl/sharedStrings.xml'))
            for item in root.findall('x:si', namespace):
                parts = [node.text or '' for node in item.findall('.//x:t', namespace)]
                shared_strings.append(''.join(parts))

        sheet_name = 'xl/worksheets/sheet1.xml'
        root = ET.fromstring(workbook.read(sheet_name))

    rows: list[list[Any]] = []
    for row in root.findall('.//x:sheetData/x:row', namespace):
        values: list[Any] = []
        for cell in row.findall('x:c', namespace):
            index = _xlsx_column_to_index(cell.attrib['r'])
            while len(values) <= index:
                values.append(None)
            value = _xlsx_cell_value(cell, shared_strings)
            if isinstance(value, str) and value in null_values:
                value = None
            values[index] = value
        rows.append(values)

    if not rows:
        return pl.DataFrame()
    headers = [str(x) for x in rows[0]]
    records = [dict(zip(headers, row, strict=False)) for row in rows[1:]]
    return pl.DataFrame(records)


def resolve_train_size(value: str | int | None) -> tuple[str, int | None]:
    if value is None:
        raise ValueError('Train size cannot be None')
    if isinstance(value, int):
        if value <= 0:
            raise ValueError('Train size must be positive')
        return str(value), value

    value = str(value).replace('_', '').lower()
    if value in TRAIN_SIZE_PRESETS:
        return value, TRAIN_SIZE_PRESETS[value]
    try:
        size = int(value)
    except ValueError as err:
        raise ValueError(
            f'Unknown train size {value!r}. Use small, medium, big, full, '
            'or an integer.'
        ) from err
    if size <= 0:
        raise ValueError('Train size must be positive')
    return str(size), size


class TrainSampler:
    """Nested subsets from one size-independent random permutation of a pool."""

    def __init__(self, train_idx: np.ndarray, *, seed: int = 0):
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.seed = seed
        self._random_order: np.ndarray | None = None
        self._time_order: np.ndarray | None = None

    def sample(self, train_size: int | None, *, sampling: TrainSampling) -> np.ndarray:
        if sampling not in ('random', 'latest'):
            raise ValueError(f'Unknown train sampling policy: {sampling!r}')
        if train_size is not None and train_size <= 0:
            raise ValueError('Train size must be positive')
        if train_size is None or train_size >= len(self.train_idx):
            return self.train_idx
        if sampling == 'random':
            if self._random_order is None:
                self._random_order = np.random.default_rng(self.seed).permutation(
                    self.train_idx
                )
            # Sort only after taking the prefix: output order stays chronological,
            # while membership is nested and independent of other requested sizes.
            return np.sort(self._random_order[:train_size])
        if self._time_order is None:
            self._time_order = np.sort(self.train_idx)
        return self._time_order[-train_size:]


def _sample_train_idx(
    train_idx: np.ndarray,
    train_size: int | None,
    *,
    seed: int,
    sampling: TrainSampling,
) -> np.ndarray:
    return TrainSampler(train_idx, seed=seed).sample(train_size, sampling=sampling)


def train_size_aliases(
    label: str,
    size: int | None,
    *,
    requested_label: str,
    presets: Mapping[str, int | None] | None = None,
) -> list[str]:
    if size is None:
        return []
    effective_presets = {**TRAIN_SIZE_PRESETS, **(presets or {})}
    return sorted(
        (
            {str(size), requested_label}
            | {name for name, count in effective_presets.items() if count == size}
        )
        - {label}
    )


def _copy_split_alias(source: Path, destination: Path) -> None:
    if source == destination:
        return
    for sampling in ('random', 'latest'):
        stale = destination / sampling
        if stale.exists() and not (source / sampling).exists():
            shutil.rmtree(stale)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def get_reference_split_idx(
    data: pl.DataFrame,
    dataset_name: str,
    part: str,
    *,
    reference_data_dir: str | Path | None,
    index_column: int,
) -> np.ndarray | None:
    """Map a reference split to row indices in the current processed table.

    The reference dataset is expected to use the common processed format:
    x_meta.npy + splits/default/{train,val,test}.npy. The split is matched by
    the original row id stored in x_meta (``index_in_full``).
    """

    if reference_data_dir is None:
        return None

    reference_root = Path(reference_data_dir).expanduser().resolve()
    reference_path = reference_root / dataset_name
    # A direct dataset path is useful for legacy size-specific datasets such as
    # data/jane-street-big, where no common parent contains a jane-street child.
    if not reference_path.exists() and (
        (reference_root / 'x_meta.npy').exists()
        or (reference_root / 'arrays/x_meta.npy').exists()
    ):
        reference_path = reference_root
    x_meta_path = reference_path / 'x_meta.npy'
    split_path = reference_path / 'splits' / 'default' / f'{part}.npy'
    if (reference_path / 'arrays').is_dir():
        x_meta_path = reference_path / 'arrays/x_meta.npy'
        split_path = reference_path / 'splits/eval/time' / f'{part}.npy'
    if not x_meta_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            f'Cannot find reference split files: {x_meta_path} and/or {split_path}'
        )

    reference_x_meta = np.load(x_meta_path, mmap_mode='r')
    reference_split = np.load(split_path)
    reference_index_in_full = reference_x_meta[reference_split, index_column].astype(
        np.int64,
        copy=False,
    )

    current = data.with_row_index().select('index', 'index_in_full')
    current_index = current['index'].to_numpy()
    current_index_in_full = current['index_in_full'].to_numpy()
    index_map = dict(
        zip(current_index_in_full.tolist(), current_index.tolist(), strict=True)
    )

    try:
        return np.array(
            [index_map[int(x)] for x in reference_index_in_full], dtype=np.int64
        )
    except KeyError as err:
        raise KeyError(
            f'The reference split {dataset_name}/{part} contains '
            f'index_in_full={err.args[0]}, which is missing in the current data.'
        ) from err


def _to_numpy(name: str, frame: pl.DataFrame, task_type: TaskType) -> np.ndarray:
    dtypes = {
        'X_num': np.float32,
        'X_bin': np.float32,
        'X_cat': np.int64,
        'X_meta': np.int64,
        'Y': np.float32 if task_type == TaskType.REGRESSION else np.int64,
    }
    array = frame.to_numpy().astype(dtypes[name], copy=False)
    if name == 'Y':
        array = array.reshape(-1)
    return array


def _check_split(name: str, split: dict[str, np.ndarray], size: int) -> None:
    keys = {f'{part}_idx' for part in PARTS}
    if set(split) != keys:
        raise ValueError(
            f'Split {name!r} must contain keys {sorted(keys)}, got {sorted(split)}'
        )

    seen: dict[str, np.ndarray] = {}
    for part in PARTS:
        key = f'{part}_idx'
        idx = np.asarray(split[key])
        if idx.ndim != 1:
            raise ValueError(f'{name}/{key} must be one-dimensional')
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f'{name}/{key} must contain integer indices')
        if len(idx) and (idx.min() < 0 or idx.max() >= size):
            raise ValueError(
                f'{name}/{key} contains out-of-range indices for size={size}'
            )
        seen[part] = np.unique(idx)
        if len(seen[part]) != len(idx):
            raise ValueError(f'{name}/{key} contains duplicate row indices')

    for i, left in enumerate(PARTS):
        for right in PARTS[i + 1 :]:
            intersection = np.intersect1d(seen[left], seen[right], assume_unique=True)
            if len(intersection):
                raise ValueError(
                    f'Split {name!r}: {left} and {right} intersect '
                    f'({len(intersection)} rows)'
                )


def _dump_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=4), encoding='utf-8')


def save_dataset(
    name: str,
    task_type: str | TaskType,
    data: dict[str, pl.DataFrame],
    splits: dict[str, dict[str, np.ndarray]],
    *,
    output_dir: str | Path | None = None,
    save_full_format: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a preprocessed dataset in the common full-array format.

    Files are written as lowercase full arrays (x_num.npy, x_cat.npy, y.npy, ...)
    plus splits/default/{train,val,test}.npy. lib.data supports this format.
    """

    task_type = TaskType(task_type)
    output_dir = (
        DATA_DIR if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    out_path = output_dir / name
    out_path.mkdir(parents=True, exist_ok=True)

    if 'Y' not in data:
        raise ValueError('The data dictionary must contain Y')

    size = data['Y'].height
    for key, frame in data.items():
        if frame.height != size:
            raise ValueError(f'{key} has {frame.height} rows, but Y has {size}')
    for split_name, split in splits.items():
        _check_split(split_name, split, size)
    if 'default' not in splits:
        raise ValueError('A "default" split is required')

    default_split = splits['default']
    score = {
        TaskType.BINCLASS: 'average-precision',
        TaskType.MULTICLASS: 'accuracy',
        TaskType.REGRESSION: 'rmse',
    }[task_type]

    # v5/common format: full arrays with lowercase names + splits/default/*.npy.
    # For train-size variants, "full" means the compact variant dataset:
    # [sampled train, fixed val, fixed test].
    compact_idx = np.concatenate(
        [np.asarray(default_split[f'{part}_idx'], dtype=np.int64) for part in PARTS]
    )
    is_identity = len(compact_idx) == size and all(
        np.array_equal(
            compact_idx[start : start + 1_000_000],
            np.arange(start, min(start + 1_000_000, size)),
        )
        for start in range(0, size, 1_000_000)
    )
    for key, frame in data.items():
        array = _to_numpy(key, frame, task_type)
        if not is_identity:
            array = array[compact_idx]
        np.save(out_path / f'{key.lower()}.npy', array, allow_pickle=False)
        del array

    split_dir = out_path / 'splits' / 'default'
    split_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for part in PARTS:
        size_part = len(default_split[f'{part}_idx'])
        np.save(
            split_dir / f'{part}.npy',
            np.arange(start, start + size_part, dtype=np.int32),
            allow_pickle=False,
        )
        start += size_part

    info: dict[str, Any] = {
        # v5/common metadata.
        'task': {
            'type': task_type.value,
            'score': score,
        },
        # Backward-compatible metadata for this repository's current loader.
        'name': name,
        'id': f'{name}--default',
        'task_type': task_type.value,
        'score': score,
        'n_num_features': data['X_num'].width if 'X_num' in data else 0,
        'n_bin_features': data['X_bin'].width if 'X_bin' in data else 0,
        'n_cat_features': data['X_cat'].width if 'X_cat' in data else 0,
        'train_size': len(default_split['train_idx']),
        'val_size': len(default_split['val_idx']),
        'test_size': len(default_split['test_idx']),
    }
    if metadata is not None:
        conflicting_keys = info.keys() & metadata.keys()
        if conflicting_keys:
            raise ValueError(
                f'Custom dataset metadata overrides reserved keys: '
                f'{sorted(conflicting_keys)}'
            )
        info.update(metadata)
    _dump_json(info, out_path / 'info.json')


def _write_split_variants(
    splits_dir: Path,
    splits: dict[str, Split],
    *,
    name: str,
    train_sizes: Sequence[str | int | None],
    sampling_variants: tuple[TrainSampling, ...],
    seed: int,
    train_size_presets: Mapping[str, int | None] | None,
    canonical_names: bool = False,
) -> dict[str, Any]:
    """Write indices only; every protocol uses the same sampling implementation."""
    # Named protocols expose only requested size variants. The complete base
    # pool remains in memory for IID construction and nested sampling.
    if not canonical_names:
        default_dir = splits_dir / 'default'
        default_dir.mkdir(parents=True, exist_ok=True)
        for part in PARTS:
            np.save(
                default_dir / f'{part}.npy',
                np.asarray(splits['default'][f'{part}_idx'], dtype=np.int32),
                allow_pickle=False,
            )
    else:
        for obsolete in ('default', 'big'):
            path = splits_dir / obsolete
            if path.exists():
                shutil.rmtree(path)

    split_metadata: dict[str, Any] = {}
    samplers = {
        key: TrainSampler(split['train_idx'], seed=seed)
        for key, split in splits.items()
    }
    for train_size_value in train_sizes:
        label, train_size = resolve_train_size(train_size_value)
        if train_size_presets is not None and label in train_size_presets:
            train_size = train_size_presets[label]
            if train_size is not None and train_size <= 0:
                raise ValueError(
                    f'Train-size override for {label!r} must be positive or None'
                )
        requested_label = label
        base_key = label if label in splits else 'default'
        base_split = splits[base_key]
        if canonical_names and train_size is not None and requested_label != 'full':
            label = str(train_size)
        if requested_label == 'full' and name.partition('/')[0] in TIME_IID_DATASETS:
            train_size = min(train_size or 10_000_000, 10_000_000)
        original_train_size = len(base_split['train_idx'])
        if train_size is not None and train_size > original_train_size:
            print(
                f'Warning: requested {train_size:,} train rows for {name}/{label}, '
                f'but only {original_train_size:,} are available; '
                'using all available rows.'
            )

        size_dir = splits_dir / label
        size_dir.mkdir(parents=True, exist_ok=True)
        for stale_sampling in {'random', 'latest'} - set(sampling_variants):
            stale_dir = size_dir / stale_sampling
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
        for part in ('val', 'test'):
            np.save(
                size_dir / f'{part}.npy',
                np.asarray(base_split[f'{part}_idx'], dtype=np.int32),
                allow_pickle=False,
            )

        sampling_metadata = {}
        for sampling in sampling_variants:
            train_idx = samplers[base_key].sample(train_size, sampling=sampling)
            sampling_dir = size_dir / sampling
            sampling_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                sampling_dir / 'train.npy',
                np.asarray(train_idx, dtype=np.int32),
                allow_pickle=False,
            )
            sampling_metadata[sampling] = {'train_size': len(train_idx)}
            print(
                f'Writing {name}/{label}/{sampling}: '
                f'train={len(train_idx):,}, '
                f'val={len(base_split["val_idx"]):,}, '
                f'test={len(base_split["test_idx"]):,}'
            )
        split_metadata[label] = {
            'target_train_size': train_size,
            'val_size': len(base_split['val_idx']),
            'test_size': len(base_split['test_idx']),
            'sampling': sampling_metadata,
        }

        if canonical_names:
            for alias in train_size_aliases(
                label,
                train_size,
                requested_label=requested_label,
                presets=train_size_presets,
            ):
                if alias == 'big':
                    continue
                _copy_split_alias(size_dir, splits_dir / alias)
                split_metadata[alias] = {**split_metadata[label], 'alias_of': label}

    return split_metadata


def compact_split_indices(splits_dir: Path, n_rows: int) -> np.ndarray:
    """Keep the union of referenced rows and remap all split files in place.

    Use on a staged split tree, before publishing matching feature arrays.
    Returned indices map compact rows back to the original feature table.
    """
    paths = sorted(splits_dir.rglob('*.npy'))
    used = np.zeros(n_rows, dtype=bool)
    for path in paths:
        idx = np.load(path)
        if idx.ndim != 1 or idx.dtype.kind not in 'iu':
            raise ValueError(f'Invalid split indices: {path}')
        if idx.size and (idx.min() < 0 or idx.max() >= n_rows):
            raise ValueError(f'Split indices out of range: {path}')
        used[idx] = True
    rows = np.flatnonzero(used)
    if not len(rows):
        raise ValueError('No rows are referenced by the requested splits')
    mapping = np.full(n_rows, -1, dtype=np.int64)
    mapping[rows] = np.arange(len(rows))
    for path in paths:
        np.save(path, mapping[np.load(path)].astype(np.int32), allow_pickle=False)
    return rows


def _write_compact_frame(
    path: Path, key: str, frame: pl.DataFrame, rows: np.ndarray, task_type: TaskType
) -> None:
    """Select/convert once and save row-major arrays for efficient row gathering."""
    started = perf_counter()
    print(f'{path.name}: selecting and converting {len(rows):,} rows', flush=True)
    if len(rows) and np.all(rows[1:] == rows[:-1] + 1):
        selected = frame.slice(int(rows[0]), len(rows))
    else:
        selected = frame[rows]
    array = np.ascontiguousarray(_to_numpy(key, selected, task_type))
    del selected
    print(
        f'{path.name}: converted in {perf_counter() - started:.1f}s; '
        f'writing {array.nbytes / 2**30:.2f} GiB ({array.dtype}, C-order)',
        flush=True,
    )
    started = perf_counter()
    np.save(path, array, allow_pickle=False)
    print(f'{path.name}: saved in {perf_counter() - started:.1f}s', flush=True)


def save_dataset_variants(
    name: str,
    task_type: str | TaskType,
    data: dict[str, pl.DataFrame],
    splits: dict[str, dict[str, np.ndarray]],
    *,
    train_sizes: Sequence[str | int | None],
    output_dir: str | Path | None = None,
    seed: int = 0,
    train_sampling: TrainSamplingSelection = 'both',
    train_size_presets: Mapping[str, int | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
    default_protocol: str | None = None,
    train_variants: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    train_sampling_algorithms: Mapping[str, str] | None = None,
) -> None:
    """Save one feature store with size/sampling variants represented by indices.

    The output is ``<name>``. Feature and target arrays are written once;
    variants live under ``splits/<size>/<random|latest>``. Validation and test
    indices are stored at the size level and inherited by both sampling variants.

    ``splits['default']`` supplies the base split. A split named after a requested
    size overrides it for that size. ``train_size_presets`` can override the
    global small/medium/big row counts for a dataset whose full train pool is
    too small for the usual 100k/1m/10m grid.

    With ``default_protocol``, ``splits`` maps evaluation protocol names to base
    splits sharing one train pool. Arrays live in ``arrays/``, training indices
    in ``splits/train/<sampling>/<size>.npy``, and evaluation indices in
    ``splits/eval/<protocol>/{val,test}.npy``. ``full`` means the entire train
    pool without a cap. Only referenced rows are stored; source_row_idx.npy
    maps them back to the pre-compaction table.

    ``train_variants`` optionally supplies precomputed indices by sampling policy
    and canonical size label for the independent layout. This allows group-aware
    sampling without changing the row sampler used by other recipes. Supply
    ``train_sampling_algorithms`` to describe each policy.
    """

    assert train_variants is None or default_protocol is not None
    if default_protocol is None:
        if 'default' not in splits:
            raise ValueError('A "default" split is required')
        default_split = splits['default']
    else:
        if default_protocol not in splits:
            raise ValueError(f'Missing default protocol: {default_protocol}')
        for protocol in splits:
            if not re.fullmatch(r'[a-z][a-z_-]*', protocol) or protocol in {
                'default',
                *TRAIN_SIZE_PRESETS,
            }:
                raise ValueError(f'Invalid split protocol name: {protocol!r}')
        shared_train_idx = splits[default_protocol]['train_idx']
        for protocol, split in splits.items():
            if not np.array_equal(split['train_idx'], shared_train_idx):
                raise ValueError(
                    'Named evaluation protocols must share one train pool; '
                    f'{protocol!r} differs from {default_protocol!r}'
                )
        default_split = splits[default_protocol]
    if train_sampling == 'both':
        sampling_variants: tuple[TrainSampling, ...] = ('random', 'latest')
    else:
        sampling_variants = (train_sampling,)

    task_type = TaskType(task_type)
    output_dir = (
        DATA_DIR if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    dataset_name = name
    out_path = output_dir / dataset_name
    out_path.mkdir(parents=True, exist_ok=True)

    if 'Y' not in data:
        raise ValueError('The data dictionary must contain Y')
    size = data['Y'].height
    for key, frame in data.items():
        if frame.height != size:
            raise ValueError(f'{key} has {frame.height} rows, but Y has {size}')
    for split_name, split in splits.items():
        _check_split(split_name, split, size)

    staging = tempfile.TemporaryDirectory(prefix='.prepare-', dir=out_path)
    staged_path = Path(staging.name)
    variants_by_protocol = {}
    if default_protocol is not None:
        if train_variants is None:
            sampler = TrainSampler(default_split['train_idx'], seed=seed)
            train_variants = {policy: {} for policy in sampling_variants}
            for value in train_sizes:
                label, count = resolve_train_size(value)
                if (
                    label != 'full'
                    and train_size_presets is not None
                    and label in train_size_presets
                ):
                    count = train_size_presets[label]
                label = 'full' if count is None else str(count)
                for policy in sampling_variants:
                    train_variants[policy][label] = sampler.sample(
                        count, sampling=policy
                    )
        for policy, variants in train_variants.items():
            directory = staged_path / 'splits/train' / policy
            directory.mkdir(parents=True, exist_ok=True)
            for label, indices in variants.items():
                assert np.isin(indices, default_split['train_idx']).all()
                np.save(
                    directory / f'{label}.npy',
                    indices.astype(np.int32),
                    allow_pickle=False,
                )
        for protocol, split in splits.items():
            directory = staged_path / 'splits/eval' / protocol
            directory.mkdir(parents=True, exist_ok=True)
            for part in ('val', 'test'):
                np.save(
                    directory / f'{part}.npy',
                    split[f'{part}_idx'].astype(np.int32),
                    allow_pickle=False,
                )
        stored_rows = compact_split_indices(staged_path / 'splits', size)
    else:
        variants_by_protocol[''] = _write_split_variants(
            staged_path / 'splits',
            splits,
            name=name,
            train_sizes=train_sizes,
            sampling_variants=sampling_variants,
            seed=seed,
            train_size_presets=train_size_presets,
        )
        stored_rows = np.arange(size)
    print(
        f'Writing shared arrays for {dataset_name}: '
        f'{len(stored_rows):,} referenced rows of {size:,}',
        flush=True,
    )
    for key, frame in data.items():
        _write_compact_frame(
            staged_path / f'{key.lower()}.npy', key, frame, stored_rows, task_type
        )
    np.save(staged_path / 'source_row_idx.npy', stored_rows, allow_pickle=False)
    array_root = out_path / 'arrays' if default_protocol is not None else out_path
    array_root.mkdir(exist_ok=True)
    for array_path in staged_path.glob('*.npy'):
        array_path.replace(array_root / array_path.name)
    if (out_path / 'splits').exists():
        shutil.rmtree(out_path / 'splits')
    (staged_path / 'splits').replace(out_path / 'splits')
    staging.cleanup()

    split_metadata = variants_by_protocol.get('', {})

    score = {
        TaskType.BINCLASS: 'average-precision',
        TaskType.MULTICLASS: 'accuracy',
        TaskType.REGRESSION: 'rmse',
    }[task_type]
    info: dict[str, Any] = {
        'task': {'type': task_type.value, 'score': score},
        'name': dataset_name,
        'id': f'{dataset_name}--default',
        'task_type': task_type.value,
        'score': score,
        'n_num_features': data['X_num'].width if 'X_num' in data else 0,
        'n_bin_features': data['X_bin'].width if 'X_bin' in data else 0,
        'n_cat_features': data['X_cat'].width if 'X_cat' in data else 0,
        'train_size': len(default_split['train_idx']),
        'val_size': len(default_split['val_idx']),
        'test_size': len(default_split['test_idx']),
        'split_variants': split_metadata,
        'train_sampling_algorithm': TRAIN_SAMPLING_ALGORITHM,
        'source_row_count': size,
        'stored_row_count': len(stored_rows),
    }
    if default_protocol is not None:
        del info['split_variants']
        info['split_layout'] = 'independent-v1'
        info['train_sampling_seed'] = seed
    if train_sampling_algorithms is not None:
        del info['train_sampling_algorithm']
        info['train_sampling_algorithms'] = dict(train_sampling_algorithms)
    if metadata is not None:
        conflicting_keys = info.keys() & metadata.keys()
        if conflicting_keys:
            raise ValueError(
                f'Custom dataset metadata overrides reserved keys: '
                f'{sorted(conflicting_keys)}'
            )
        info.update(metadata)
    if default_protocol is not None:
        info = clean_independent_info(info)
    _dump_json(info, out_path / 'info.json')
