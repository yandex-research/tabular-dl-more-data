"""Generate one model-size sweep evaluating both default and IID holdouts.

The default run uses time (core for Istella) as its primary validation split and
also selects best steps independently on IID validation in best_by_validation.
"""

import argparse
import shlex
from itertools import product
from pathlib import Path

import lib.env
import lib.experiment

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DATASETS = (
    'weather',
    'maps-routing',
    'cooking-time',
    'delivery-eta',
    'aeolus-flight-delay',
    'jane-street',
    'istella',
    'bank-fraud-datafusion',
)
PROTOCOLS = ('default', 'iid')
PRETRANSFORMED_DATASETS = {'cooking-time', 'delivery-eta', 'maps-routing'}


DEPTH = 3
BATCH_SIZE = 1024
SEEDS = (0,)

# Eleven approximately logarithmic targets between 10k and 10M. Widths are rounded to
# multiples of 16 and shared verbatim by every dataset. Since the first-layer size
# is dataset-dependent, exact counts differ.
WIDTH_SPECS = (
    (10_000, 64),
    (20_000, 96),
    (40_000, 128),
    (80_000, 176),
    (160_000, 240),
    (320_000, 352),
    (640_000, 496),
    (1_280_000, 704),
    (2_560_000, 992),
    (5_120_000, 1392),
    (10_000_000, 1936),
)


def geomspace(start: float, stop: float, n_steps: int) -> tuple[float, ...]:
    ratio = (stop / start) ** (1.0 / (n_steps - 1))
    values = [float(f'{start * ratio**i:.12g}') for i in range(n_steps)]
    values[0] = start
    values[-1] = stop
    return tuple(values)


LEARNING_RATES = geomspace(3e-5, 1e-3, 10)
WEIGHT_DECAYS = geomspace(0.005, 5.0, 10)


def get_eval_key(dataset: str, protocol: str) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown evaluation protocol: {protocol}')
    if protocol == 'default':
        return 'core' if dataset == 'istella' else 'time'
    return protocol


def get_experiment_dir(
    *,
    protocol: str,
    dataset: str,
    size: str,
    target_n_parameters: int,
    width: int,
) -> Path:
    shape = f'p{target_n_parameters:08d}-w{width}'
    return EXPERIMENTS_DIR / dataset / size / shape


def make_config(
    *,
    protocol: str,
    dataset: str,
    size: str,
    width: int,
) -> dict:
    eval_key = get_eval_key(dataset, protocol)
    other_eval_key = get_eval_key(
        dataset, 'iid' if protocol == 'default' else 'default'
    )
    data = {
        'dataset_dir': f'data/{dataset}',
        'train': {
            'size': 'full' if size == 'full' else int(size),
            'sampling': 'random',
        },
        'val_key': eval_key,
        'test_key': eval_key,
    }
    if dataset not in PRETRANSFORMED_DATASETS:
        data['num_transform'] = 'noisy-quantile'
    return {
        'data': data,
        'model': {
            'backbone': {
                'n_blocks': DEPTH,
                'd_block': width,
                'dropout': 0.0,
            }
        },
        'optimizer': {
            'type': 'torch.optim.AdamW',
        },
        'batch_size': BATCH_SIZE,
        'learning_rates': list(LEARNING_RATES),
        'weight_decays': list(WEIGHT_DECAYS),
        'seeds': list(SEEDS),
        'n_steps': 100_000,
        'eval_every_n_steps': 1_000,
        'eval_parts': [
            'val',
            'test',
            f'val_{other_eval_key}',
            f'test_{other_eval_key}',
        ],
        'checkpoint_every_n_steps': 5_000,
        'gradient_clipping_norm': 1.0,
        'amp_dtype': 'bfloat16',
        'num_dtype': 'bfloat16',
    }


def make_command(exp: Path) -> str:
    relative_exp = exp.relative_to(lib.env.get_project_dir())
    return f'uv run -m project.pack.model --resume {shlex.quote(str(relative_exp))}'


def build_plan(protocols: tuple[str, ...] = ('default',)) -> list[tuple[Path, dict]]:
    if protocols != ('default',):
        raise ValueError('The paper uses one shared sweep with both evaluation splits')
    plan = []
    for dataset in DATASETS:
        dataset_dir = lib.env.get_project_dir() / 'data' / dataset
        train_dir = dataset_dir / 'splits/train/random'
        sizes = sorted(
            (
                p.stem
                for p in train_dir.glob('*.npy')
                if p.stem == 'full' or (p.stem.isdecimal() and int(p.stem) > 0)
            ),
            key=lambda size: float('inf') if size == 'full' else int(size),
        )
        if not sizes:
            raise FileNotFoundError(f'No training splits in {train_dir}')
        # Both holdout protocols are required by each training run.
        for evaluation_protocol in PROTOCOLS:
            eval_key = get_eval_key(dataset, evaluation_protocol)
            for part in ('val', 'test'):
                path = dataset_dir / 'splits/eval' / eval_key / f'{part}.npy'
                if not path.is_file():
                    raise FileNotFoundError(path)
        for protocol in protocols:
            for size, (target_n_parameters, width) in product(sizes, WIDTH_SPECS):
                exp = get_experiment_dir(
                    protocol=protocol,
                    dataset=dataset,
                    size=size,
                    target_n_parameters=target_n_parameters,
                    width=width,
                )
                config = make_config(
                    protocol=protocol,
                    dataset=dataset,
                    size=size,
                    width=width,
                )
                plan.append((exp, config))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--protocols',
        choices=('default',),
        nargs='+',
        default=('default',),
        help='Primary validation protocol; each run also evaluates the other protocol.',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--overwrite-configs',
        action='store_true',
        help='Replace existing config.json files, preserving reports and checkpoints.',
    )
    args = parser.parse_args()
    protocols = tuple(dict.fromkeys(args.protocols))
    plan = build_plan(protocols)
    # Validate all targets before writing; only configs may be overwritten.
    for exp, config in plan:
        if exp.exists() and (
            not lib.experiment.is_experiment(exp)
            or (
                not args.overwrite_configs and lib.experiment.load_config(exp) != config
            )
        ):
            raise ValueError(
                f'Existing experiment differs or is invalid: {exp}. '
                'Use --overwrite-configs to replace configs of existing experiments.'
            )

    commands: list[str] = []
    for exp, config in plan:
        if not args.dry_run:
            if not exp.exists():
                lib.experiment.create(exp, config=config, parents=True)
            elif args.overwrite_configs:
                lib.experiment.dump_config(exp, config)
        if not exp.exists() or not lib.experiment.is_completed(exp):
            commands.append(make_command(exp))

    if not args.dry_run:
        EXPERIMENTS_DIR.joinpath('commands.sh').write_text(
            '\n'.join(commands) + '\n'
        )
    print(
        f'{"Planned" if args.dry_run else "Prepared"} {len(plan)} configs '
        f'({len(DATASETS)} datasets, protocols: {", ".join(protocols)}, '
        f'available random train sizes per dataset, '
        f'{len(WIDTH_SPECS)} widths, '
        f'{len(LEARNING_RATES) * len(WEIGHT_DECAYS) * len(SEEDS)} models per pack)'
    )


if __name__ == '__main__':
    lib.env.initialize()
    main()
