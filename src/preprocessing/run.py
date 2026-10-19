import argparse
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

from preprocessing.util import DATA_DIR, default_train_sizes

_DATASETS = {
    'cooking-time': 'cooking-time.py',
    'cooking': 'cooking-time.py',
    'delivery-eta': 'delivery-eta.py',
    'delivery': 'delivery-eta.py',
    'maps-routing': 'maps-routing.py',
    'maps': 'maps-routing.py',
    'weather': 'weather.py',
    'jane-street': 'jane-street.py',
    'jane_street': 'jane-street.py',
    'janestreet': 'jane-street.py',
    'istella': 'istella.py',
    'aeolus-flight-delay': 'aeolus-flight-delay.py',
    'aeolus': 'aeolus-flight-delay.py',
    'flight-delay': 'aeolus-flight-delay.py',
    'bank-fraud-datafusion': 'bank-fraud-datafusion.py',
}


def _load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import preprocessing script: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
    )
    parser = argparse.ArgumentParser(
        description=(
            'Preprocess a TabReD dataset from local raw files without downloading.'
        )
    )
    parser.add_argument('dataset', choices=sorted(_DATASETS))
    parser.add_argument(
        'raw_data_dir',
        type=Path,
        help=(
            'Local extracted dataset directory, parent directory, or local Kaggle zip.'
        ),
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
        default=None,
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
        choices=['random', 'latest', 'query', 'both'],
        default=None,
        help=(
            'Training policies: random/latest, or random/query for Istella. '
            'Defaults to both supported policies.'
        ),
    )
    parser.add_argument(
        '--reference-data-dir',
        type=Path,
        default=None,
        help=(
            'Optional directory with reference processed datasets. If provided, '
            'default val/test splits are reproduced from the reference.'
        ),
    )
    parser.add_argument(
        '--max-categories',
        type=int,
        default=None,
        help='Aeolus category limit (default 512); 0 keeps legacy IDs.',
    )
    args = parser.parse_args()

    script_name = _DATASETS[args.dataset]
    is_istella = script_name == 'istella.py'
    is_datafusion = script_name == 'bank-fraud-datafusion.py'
    is_aeolus = _DATASETS[args.dataset] == 'aeolus-flight-delay.py'

    if args.max_categories is not None and not is_aeolus:
        parser.error('--max-categories is only supported for Aeolus')

    if args.train_sampling == 'query' and not is_istella:
        parser.error('--train-sampling query is only supported for Istella')
    if is_istella and args.train_sampling == 'latest':
        parser.error('Istella supports random/query sampling, not latest')

    script = Path(__file__).resolve().parent / script_name
    module = _load_script(script)

    if is_istella:
        kwargs = {
            'output_dir': args.output_dir,
            'train_sizes': args.train_size,
            'train_sampling': args.train_sampling or 'both',
        }
    elif is_datafusion:
        kwargs = {
            'output_dir': args.output_dir,
            'train_sizes': args.train_size,
            'train_sampling': args.train_sampling or 'both',
        }
    else:
        kwargs = {
            'output_dir': args.output_dir,
            'train_sizes': args.train_size
            or default_train_sizes(Path(script_name).stem),
            'train_sampling': args.train_sampling or 'both',
            'reference_data_dir': args.reference_data_dir,
        }
    if is_aeolus and args.max_categories is not None:
        kwargs['max_categories'] = args.max_categories

    module.main(
        args.raw_data_dir,
        **kwargs,
    )


if __name__ == '__main__':
    main()
