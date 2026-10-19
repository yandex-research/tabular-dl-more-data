"""Generate main recipes with IID validation and the original main test."""

import argparse
import importlib.util
from pathlib import Path

import lib.env

OUTPUT_ROOT = Path(__file__).resolve().parent
METHODS = ('mlp/muon', 'lightgbm/10000')

_spec = importlib.util.spec_from_file_location(
    'main_make', OUTPUT_ROOT.parent / 'main/make.py'
)
assert _spec is not None and _spec.loader is not None
_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main)
DATA_ROOT = _main.DATA_ROOT
DATASETS = _main.DATASETS


def build_plan(
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    datasets: tuple[str, ...] = DATASETS,
    methods: tuple[str, ...] = METHODS,
):
    if not methods or set(methods) - set(METHODS):
        raise ValueError(f'Invalid IID-validation methods: {methods}')
    return _main.build_plan(
        data_root, output_root, datasets, methods, val_key='iid'
    )


def main() -> None:
    lib.env.initialize()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--sizes', nargs='+', help='Select train sizes')
    args = parser.parse_args()
    datasets = (
        tuple(args.datasets)
        if args.datasets is not None
        else tuple(
            name for name in DATASETS if (DATA_ROOT / name / 'info.json').is_file()
        )
    )
    if args.datasets is None:
        missing = [name for name in DATASETS if name not in datasets]
        if missing:
            print(f'Skipping datasets without info.json: {", ".join(missing)}')
    if not datasets:
        parser.error(f'No prepared datasets found in {DATA_ROOT}')
    plan = build_plan(datasets=datasets, methods=tuple(args.methods))
    if args.sizes is not None:
        available = {exp.parent.name for exp, _ in plan}
        if set(args.sizes) - available:
            parser.error(f'Unknown train sizes: {sorted(set(args.sizes) - available)}')
        plan = [(exp, config) for exp, config in plan if exp.parent.name in args.sizes]
    command_files = {}
    for method in args.methods:
        root = OUTPUT_ROOT / method
        command_files[root / 'commands.sh'] = _main.make_commands(
            [(exp, config) for exp, config in plan if root in exp.parents]
        )
    created = _main.generate(plan, dry_run=args.dry_run)
    if not args.dry_run:
        for path, commands in command_files.items():
            path.write_text(commands)
    action = 'Would create' if args.dry_run else 'Created'
    print(
        f'{action} {created} configs; {len(plan) - created} unchanged. No training run.'
    )


if __name__ == '__main__':
    main()
