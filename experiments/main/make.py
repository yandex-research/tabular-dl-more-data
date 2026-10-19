"""Generate the main benchmark configs; never launch training or replace runs."""

import argparse
import shlex
from pathlib import Path
from typing import Any, cast
from collections import defaultdict

import numpy as np

import lib.env
import lib.experiment


def get_batch_size(n_rows: int) -> int:
    if n_rows >= 5_120_000:
        return 4096
    if n_rows >= 1_280_000:
        return 2048
    if n_rows >= 160_000:
        return 1024
    return 256 if n_rows <= 20_000 else 512


DATA_ROOT = Path('data')
OUTPUT_ROOT = Path(__file__).resolve().parent
DATASETS = (
    'cooking-time',
    'delivery-eta',
    'maps-routing',
    'weather',
    'jane-street',
    'aeolus-flight-delay',
    'istella',
    'bank-fraud-datafusion',
)


VAL_KEY = defaultdict(lambda: 'time', {'istella': 'core'})
TEST_KEY = defaultdict(lambda: 'time', {'istella': 'core'})
N_TRIALS = 100
N_SEEDS = 10
N_WORKERS = 2
METHODS = (
    'mlp/adamw',
    'mlp/muon',
    'mlp-ple/adamw',
    'mlp-ple/muon',
    'tabm/muon',
    'tabpack/muon',
    'tabpack-cosine/muon',
    'xgboost/10000',
    'lightgbm/10000',
    'tabpfn/v3',
    'tabpfn/v3.5',
    'tabpfn/v3.5_fast',
    'tabicl/v2',
)
PRETRANSFORMED_DATASETS = {'cooking-time', 'delivery-eta', 'maps-routing'}


def make_config(
    dataset_dir: Path,
    size: str,
    n_rows: int,
    method: str = 'mlp/adamw',
    *,
    eval_key: str | None = None,
    val_key: str | None = None,
    test_key: str | None = None,
) -> dict[str, Any]:
    """Build a model recipe with independent splits."""
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    data: dict[str, Any] = {
        'dataset_dir': str(dataset_dir),
        'train': {
            'size': 'full' if size == 'full' else int(size),
            'sampling': 'random',
        },
        'val_key': (
            val_key
            if val_key is not None
            else (eval_key if eval_key is not None else VAL_KEY[dataset_dir.name])
        ),
        'test_key': (
            test_key
            if test_key is not None
            else (eval_key if eval_key is not None else TEST_KEY[dataset_dir.name])
        ),
    }
    if method == 'xgboost/10000':
        return make_xgboost_config(data)
    if method == 'lightgbm/10000':
        return make_lightgbm_config(data)
    if method == 'tabpfn/v3':
        return make_tabpfn_config(data)
    if method == 'tabicl/v2':
        return make_tabicl_config(data)
    if method in (
        'tabpfn/v3.5',
        'tabpfn/v3.5_fast',
        ):
        config = make_tabpfn_config(data)
        if n_rows >= 1_280_000:
            config['n_seeds'] = 3
        config['base_config']['model']['model_path'] = (
            'checkpoints/tabpfn-v3/tabpfn-v3.5-20260909.safetensors'
        )
        if method == 'tabpfn/v3.5_fast':
            config['n_seeds'] = N_SEEDS
            config['base_config']['model']['model_path'] = (
                'checkpoints/tabpfn-v3/tabpfn-v3.5-fast-20260909.safetensors'
            )
        return config
    if dataset_dir.name not in PRETRANSFORMED_DATASETS:
        data['num_transform'] = 'noisy-quantile'
    config: dict[str, Any] = {
        'seed': 0,
        'objective': {
            'function': 'project.mlp.main',
            'output_keys': ['metrics', 'val', 'score'],
        },
        'space': {
            'seed': 0,
            'data': data,
            'model': {
                'backbone': {
                    'n_blocks': ['_tune_', 'int', 1, 6],
                    'd_block': ['_tune_', 'int', 64, 1024, 16],
                    'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
                },
            },
            'optimizer': {
                'type': 'torch.optim.AdamW',
                'lr': ['_tune_', 'loguniform', 3e-05, 0.001],
                'weight_decay': ['_tune_', 'loguniform', 0.005, 5.0],
            },
            'batch_size': get_batch_size(n_rows),
            'patience': 16,
            'gradient_clipping_norm': 1.0,
            'amp_dtype': 'bfloat16',
            'num_dtype': 'bfloat16',
        },
        'sampler': {'type': 'TPESampler'},
        'n_trials': N_TRIALS,
        'n_workers': N_WORKERS,
    }
    model, optimizer = method.split('/')
    if model.startswith('tabpack'):
        pack = make_tabpack_config(data, n_rows, cosine=model == 'tabpack-cosine')
        return pack

    space = cast(dict[str, Any], config['space'])
    if optimizer == 'muon':
        space['optimizer'].update(
            {
                'type': 'vendor.muon.SingleDeviceMuonWithAuxAdam',
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'muon': {'lr': ['_tune_', 'loguniform', 0.0001, 0.03]},
            }
        )
    if model == 'mlp-ple':
        space['model']['num_embeddings'] = {
            'type': 'PiecewiseLinearEmbeddings',
            'd_embedding': ['_tune_', 'int', 8, 32, 4],
            'activation': False,
            'version': 'B',
        }
        space['model_num_embeddings_bins'] = {'n_bins': ['_tune_', 'int', 2, 128]}
    elif model == 'tabm':
        config['objective'] = {
            'function': 'project.tabm_.main',
            'output_keys': ['metrics', 'val', 'score'],
        }
        config['sampler'] = {'type': 'TPESampler', 'n_startup_trials': 20}
        space['share_training_batches'] = False
        space['model']['type'] = 'tabm'
        space['model']['backbone'].update(
            {
                'arch_type': 'tabm',
                'k': 16,
                'n_blocks': ['_tune_', 'int', 1, 5],
            }
        )
        space['optimizer'].update(
            {
                'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
                'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
                'muon': {'lr': ['_tune_', 'loguniform', 0.001, 0.1]},
            }
        )
        space['compile'] = True
    return config


def make_tabpfn_config(data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the TabPFN v3 example recipe over independent seeds."""
    return {
        'function': 'project.tabpfn_.main',
        'base_config': {
            'data': {**data, 'num_transform': None, 'cat_transform': None},
            'model': {},
            'eval_parts': ['val', 'test'],
            'concat_val': False,
        },
        'n_seeds': N_SEEDS,
        'save_experiments': False,
        'n_workers': 1,
    }


def make_tabicl_config(data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate TabICL with its default task-specific checkpoint over ten seeds."""
    return {
        'function': 'project.tabicl_.main',
        'base_config': {
            'data': {**data, 'num_transform': None, 'cat_transform': None},
            'model': {},
            'eval_parts': ['val', 'test'],
            'concat_val': False,
        },
        'n_seeds': N_SEEDS,
        'save_experiments': True,
        'n_workers': 1,
    }


def make_xgboost_config(data: dict[str, Any]) -> dict[str, Any]:
    """Preserve the random-splits XGBoost 10000-tree tuning recipe."""
    return {
        'seed': 0,
        'objective': {
            'function': 'project.xgboost_.main',
            'output_keys': ['metrics', 'val', 'score'],
        },
        'space': {
            'seed': 0,
            'data': data,
            'model': {
                'booster': 'gbtree',
                'n_estimators': 10000,
                'early_stopping_rounds': 300,
                'n_jobs': 8,
                'tree_method': 'hist',
                'device': 'cuda',
                'use_label_encoder': False,
                'max_depth': ['_tune_', 'int', 3, 10],
                'min_child_weight': ['_tune_', 'loguniform', 0.0001, 100.0],
                'subsample': ['_tune_', 'uniform', 0.5, 1.0],
                'colsample_bytree': ['_tune_', 'uniform', 0.5, 1.0],
                'learning_rate': ['_tune_', 'loguniform', 0.001, 1.0],
                'gamma': ['_tune_', '?loguniform', 0, 0.001, 100.0],
                'alpha': ['_tune_', '?loguniform', 0.0, 0.0001, 5.0],
                'lambda': ['_tune_', '?loguniform', 0.0, 0.0001, 5.0],
                'grow_policy': ['_tune_', 'categorical', ['depthwise', 'lossguide']],
            },
            'fit': {'verbose': True},
        },
        'sampler': {'type': 'TPESampler'},
        'n_trials': N_TRIALS,
        'n_workers': N_WORKERS,
    }


# GPU training never produced results for these datasets, so they run on cpu
# (see be4fa3d1).
LIGHTGBM_CPU_DATASETS = {'jane-street', 'aeolus-flight-delay'}


def make_lightgbm_config(data: dict[str, Any]) -> dict[str, Any]:
    """Tune the 10000-tree LightGBM recipe over 100 trials."""
    dataset = Path(data['dataset_dir']).name
    return {
        'seed': 0,
        'objective': {
            'function': 'project.lightgbm_.main',
            'output_keys': ['metrics', 'val', 'score'],
        },
        'space': {
            'seed': 0,
            'data': data,
            'model': {
                'n_estimators': 10000,
                'stopping_rounds': 300,
                'device_type': ('cpu' if dataset in LIGHTGBM_CPU_DATASETS else 'gpu'),
                'verbose': -1,
                'n_jobs': 8,
                'feature_fraction': ['_tune_', 'uniform', 0.5, 1.0],
                'lambda_l2': ['_tune_', '?loguniform', 0.0, 0.1, 10.0],
                'learning_rate': ['_tune_', 'loguniform', 0.005, 0.1],
                'num_leaves': ['_tune_', 'int', 2, 256],
                'min_child_samples': ['_tune_', 'int', 5, 256],
                'min_sum_hessian_in_leaf': 0.001,
                'bagging_fraction': ['_tune_', 'uniform', 0.5, 1.0],
                'bagging_freq': 1,
                'extra_trees': ['_tune_', 'categorical', [False, True]],
            },
            'fit': {},
        },
        'sampler': {'type': 'TPESampler'},
        'n_trials': 100,
        'n_workers': N_WORKERS,
    }


def make_tabpack_config(
    data: dict[str, Any], n_rows: int, *, cosine: bool
) -> dict[str, Any]:
    model: dict[str, Any] = {'activation': 'ReLU', 'd_block': 384}
    model_space: dict[str, Any] = {
        'n_blocks': ['_tune_', 'int', 1, 3 if cosine else 4],
        'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
    }
    if cosine:
        model['num_embeddings'] = {'type': 'CosineEmbeddingsPack'}
        max_embedding = 20 if Path(data['dataset_dir']).name == 'maps-routing' else 32
        model_space['num_embeddings'] = {
            'd_embedding': ['_tune_', 'int', 8, max_embedding, 4],
            'init_scale': ['_tune_', 'loguniform', 0.01, 10.0],
        }
    return {
        'seed': 0,
        'data': data,
        'n_models': 64,
        'model': model,
        'optimizer': {'type': 'MuonAdamWPack', 'shared_step': True},
        'batch_size': get_batch_size(n_rows),
        'n_epochs': -1,
        'patience': 16,
        'online_ensembles': {
            'greedy': {
                'type': 'greedy',
                'update_type': 'latest',
                'include_current_ensemble_in_pool': True,
                'patience': 32,
                'options': {'max_ensemble_size': 32},
            },
        },
        'sampler': {
            'type': 'RandomSampler',
            'space': {
                'model': model_space,
                'optimizer': {
                    'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
                    'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
                    'muon_lr': ['_tune_', 'loguniform', 0.001, 0.1],
                },
            },
        },
        'amp_dtype': 'bfloat16',
        'num_dtype': 'bfloat16',
        'save_final_predictions': False,
        'save_all_predictions': False,
        'track_online_ensemble_history': True,
        'track_experiments': True,
    }


def build_plan(
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    datasets: tuple[str, ...] = DATASETS,
    methods: tuple[str, ...] = METHODS,
    *,
    eval_key: str | None = None,
    val_key: str | None = None,
    test_key: str | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f'Invalid methods: {methods}')
    plan = []
    for dataset in datasets:
        dataset_dir = data_root / dataset
        dataset_methods = methods
        val_keys = {
            val_key
            if val_key is not None
            else (eval_key if eval_key is not None else VAL_KEY[dataset])
            for method in dataset_methods
        }
        for path in (
            dataset_dir / 'info.json',
            dataset_dir / 'arrays/y.npy',
            *(
                dataset_dir / 'splits/eval' / key / 'val.npy'
                for key in sorted(val_keys)
            ),
            dataset_dir
            / 'splits/eval'
            / (
                test_key
                if test_key is not None
                else (eval_key if eval_key is not None else TEST_KEY[dataset])
            )
            / 'test.npy',
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        sizes = {
            path.stem: path
            for path in (dataset_dir / 'splits/train/random').glob('*.npy')
            if path.stem == 'full' or (path.stem.isdecimal() and int(path.stem) > 0)
        }
        if not sizes:
            raise ValueError(f'No random train sizes in {dataset_dir}')
        for size in sorted(
            sizes, key=lambda x: float('inf') if x == 'full' else int(x)
        ):
            indices = np.load(sizes[size], mmap_mode='r', allow_pickle=False)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(f'Invalid train indices: {sizes[size]}')
            n_rows = len(indices)
            del indices
            if not n_rows:
                raise ValueError(f'Empty train: {sizes[size]}')
            for method in dataset_methods:
                stage = 'main' if method.startswith('tabpack') else 'tuning'
                if method.startswith(('tabpfn/', 'tabicl/')):
                    stage = 'evaluation'
                exp = output_root / method / dataset / size / stage
                plan.append(
                    (
                        exp,
                        make_config(
                            dataset_dir,
                            size,
                            n_rows,
                            method,
                            eval_key=eval_key,
                            val_key=val_key,
                            test_key=test_key,
                        ),
                    )
                )
    return plan


def generate(plan: list[tuple[Path, dict[str, Any]]], *, dry_run: bool = False) -> int:
    # Check every target before writing anything. Even completed runs must match.
    for exp, config in plan:
        if exp.exists() and (
            not lib.experiment.is_experiment(exp)
            or lib.experiment.load_config(exp) != config
        ):
            raise ValueError(
                f'Existing experiment differs; refusing to overwrite: {exp}'
            )
    created = 0
    for exp, config in plan:
        if not exp.exists():
            created += 1
            if not dry_run:
                lib.experiment.create(exp, config=config, parents=True)
    return created


def make_commands(plan: list[tuple[Path, dict[str, Any]]]) -> str:
    project_dir = lib.env.get_project_dir()
    lines = []
    for exp, config in plan:
        if exp.name == 'main':
            path = shlex.quote(str(exp.parent.relative_to(project_dir)))
            lines.extend(
                [
                    '',
                    f'uv run --no-sync -m project.tabpack.run {path}'
                    ' --eval-online-ensembles greedy'
                    f' --eval-online-ensembles-n-seeds {N_SEEDS}',
                ]
            )
            continue
        path = shlex.quote(str(exp.relative_to(project_dir)))
        if exp.name == 'evaluation':
            uv_options = '--no-sync'
            lines.extend(
                ['', f'uv run {uv_options} -m lib.tools.evaluate --resume {path}']
            )
            continue
        lines.extend(
            [
                '',
                f'uv run --no-sync -m lib.tools.tune --resume {path}',
                'uv run --no-sync -m lib.tools.evaluate from-tuning'
                f' --resume --save-experiments --n-seeds {N_SEEDS} {path}',
            ]
        )
    return '\n'.join(lines) + '\n'


def main() -> None:
    lib.env.initialize()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument(
        '--sizes', nargs='+', help='Select train sizes, e.g. 10000 20000'
    )
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
    command_files: dict[Path, str] = {}
    for method in args.methods:
        root = OUTPUT_ROOT / method
        command_files[root / 'commands.sh'] = make_commands(
            [(exp, config) for exp, config in plan if root in exp.parents]
        )
    created = generate(plan, dry_run=args.dry_run)
    if not args.dry_run:
        for path, commands in command_files.items():
            path.write_text(commands)
    action = 'Would create' if args.dry_run else 'Created'
    print(
        f'{action} {created} configs; {len(plan) - created} unchanged. No training run.'
    )


if __name__ == '__main__':
    main()
