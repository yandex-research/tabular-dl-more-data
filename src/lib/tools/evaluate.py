import dataclasses
import logging
import shutil
from pathlib import Path
from typing import Any

import lib.env
import lib.experiment
import lib.tools._parallel
import lib.tools._utils
import lib.utils.dataclasses
import lib.utils.misc
import lib.utils.time

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    function: str
    base_config: dict[str, Any]
    n_seeds: int
    # The `n_workers` and `n_gpus_per_worker` fields are described in `bin/tune.py`.
    n_workers: int | None = None
    n_gpus_per_worker: int | None = None
    save_experiments: bool = False

    def __post_init__(self):
        if 'seed' in self.base_config:
            raise ValueError('The base config must not contain the `seed` field')

        if self.n_gpus_per_worker is not None and self.n_workers is None:
            raise ValueError(
                'The `n_gpus_per_worker` option cannot be passed'
                ' without the `n_workers` option'
            )


def _evaluate_seed(config: Config, exp: Path, seed: int) -> None:
    seed_exp = exp / 'seeds' / str(seed)
    with lib.tools._parallel.lock():
        if seed_exp.exists():
            _LOGGER.warning(f'Removing the incomplete seed experiment: {seed_exp}')
            # `lib.experiment.remove` is not used, because `seed_exp` may be an
            # empty directory, which is not a valid experiment directory.
            shutil.rmtree(seed_exp)

    worker_id = lib.tools._parallel.get_worker_id()
    function = lib.utils.misc.import_(config.function)
    function = lib.experiment.check_experiment_function(function)

    # All operations with the experiment directory must be performed under the lock.
    with lib.tools._parallel.lock():
        checkpoint = lib.tools._utils.try_load_checkpoint(exp)
    if checkpoint is None:
        timer = lib.utils.time.Timer()
    else:
        del checkpoint['report']
        # Each worker resumes with the latest saved timer.
        timer = checkpoint.pop('timer')
        assert not checkpoint, (
            f'Some checkpoint fields were not used, namely: {sorted(checkpoint)}'
        )

    timer.run()

    seed_config: dict[str, Any] = {'seed': seed, **config.base_config}
    with lib.experiment.create_temporary(
        seed_config, suffix=f'_seed_{seed}'
    ) as tmp_exp:
        print()
        seed_report = function(tmp_exp)
        assert seed_report is not None
        if worker_id is not None:
            seed_report['evaluation'] = {'worker_id': worker_id}

        if config.save_experiments:
            # Keep config/report: move() requires a valid experiment, and saved
            # seeds should remain independently inspectable alongside their model.
            with lib.tools._parallel.lock():
                seed_exp.parent.mkdir(exist_ok=True)
                lib.experiment.move(tmp_exp, seed_exp)

    # All operations with the experiment directory must be performed under the lock.
    with lib.tools._parallel.lock():
        report = lib.experiment.load_report(exp)
        report.setdefault('experiments', []).append(
            {'config': seed_config, 'report': seed_report}
        )
        report['experiments'].sort(key=lambda x: x['config']['seed'])
        report['time'] = timer.elapsed()
        if worker_id is not None:
            report['worker_id'] = worker_id

        lib.tools._utils.dump_checkpoint(exp, {'report': report, 'timer': timer})
        lib.experiment.dump_report(exp, report)


def main(
    exp: str | Path, *, force: bool = False, resume: bool = False
) -> lib.experiment.ReportDict | None:
    exp = lib.experiment.check(exp)
    config = lib.utils.dataclasses.from_dict(Config, lib.experiment.load_config(exp))
    recovered = None
    if resume and not force:
        checkpoint = lib.tools._utils.try_load_checkpoint(exp)
        if checkpoint is not None:
            recovered = checkpoint['report']
        elif lib.experiment.get_report_path(exp).exists():
            recovered = lib.experiment.load_report(exp)
        if recovered is not None:
            # Downloaded evaluations may retain only config.json/report.json.
            # Validate before touching their completion state or any results.
            seen = set()
            for member in recovered.get('experiments', []):
                seed = member['config']['seed']
                if (
                    seed in seen
                    or not isinstance(seed, int)
                    or not 0 <= seed < config.n_seeds
                ):
                    raise ValueError(f'Invalid or duplicate saved seed: {seed}')
                if member['config'] != {'seed': seed, **config.base_config}:
                    raise ValueError(f'Saved seed {seed} has a different base config')
                seen.add(seed)
            if checkpoint is None:
                timer = lib.utils.time.Timer()
                timer.__setstate__(
                    {**timer.__getstate__(), '_shift': float(recovered.get('time', 0))}
                )
                lib.tools._utils.dump_checkpoint(
                    exp, {'report': recovered, 'timer': timer}
                )

    extending = (
        recovered is not None
        and lib.experiment.is_completed(exp)
        and len(recovered.get('experiments', [])) < config.n_seeds
    )
    if extending:
        # start() requires a fresh directory; reopen without resetting results.
        (exp / '_RUNNING').touch(exist_ok=False)
        lib.experiment.resume(exp, main)
    elif not lib.experiment.try_start(exp, main, force=force, resume=resume):
        print(f'The experiment is already completed: {exp}')
        return None

    report = lib.experiment.create_report(exp, main)
    print(f'{lib.experiment.format_config(config)}\n')

    checkpoint = lib.tools._utils.try_load_checkpoint(exp)
    if checkpoint is None:
        completed_seeds = frozenset()
    else:
        report = checkpoint['report']
        # A failure before the first completed seed leaves only report metadata.
        completed_seeds = frozenset(
            x['config']['seed'] for x in report.get('experiments', [])
        )
        print(
            'Resuming from a checkpoint.'
            f' Completed {len(completed_seeds)}/{config.n_seeds} experiments'
        )
        report.setdefault('resumed_after_n_seeds', []).append(len(completed_seeds))

    lib.experiment.dump_report(exp, report)
    del report

    remaining_seeds = [x for x in range(config.n_seeds) if x not in completed_seeds]
    n_remaining_seeds = len(remaining_seeds)

    if config.n_workers is None:
        for seed in remaining_seeds:
            _evaluate_seed(config, exp, seed)
    elif n_remaining_seeds:
        if config.n_workers > n_remaining_seeds:
            _LOGGER.info(
                f'Reducing the number of workers'
                f' from {config.n_workers=} to {n_remaining_seeds=}'
            )
            n_workers = n_remaining_seeds
        else:
            n_workers = config.n_workers
        _LOGGER.info(f'The number of workers: {n_workers}')

        lib.tools._parallel.map(
            _evaluate_seed,
            [{'config': config, 'exp': exp, 'seed': seed} for seed in remaining_seeds],
            n_workers=n_workers,
            n_gpus_per_worker=(
                1 if config.n_gpus_per_worker is None else config.n_gpus_per_worker
            ),
        )

    report = lib.experiment.load_report(exp)
    print(f'Report\n{lib.utils.misc.add_frame(lib.experiment.format_report(report))}\n')
    lib.experiment.finish(exp, main, report=report)
    return report


def from_tuning(
    tuning_exp: str | Path,
    *,
    n_seeds: int,
    save_experiments: bool = False,
    force: bool = False,
    resume: bool = False,
) -> lib.experiment.ReportDict | None:
    tuning_exp = Path(tuning_exp)
    if not lib.experiment.is_completed(tuning_exp):
        raise RuntimeError(f'The tuning experiment is not completed: {tuning_exp}')

    tuning_config = lib.experiment.load_config(tuning_exp)
    tuning_report = lib.experiment.load_report(tuning_exp)

    evaluation_config = {
        'function': tuning_config['objective']['function'],
        'base_config': tuning_report['best']['config'],
        'n_seeds': n_seeds,
        'save_experiments': save_experiments,
    }
    for key in ['n_workers', 'n_gpus_per_worker']:
        if key in tuning_config:
            evaluation_config[key] = tuning_config[key]
        del key
    evaluation_config['base_config'].pop('seed', None)

    evaluation_exp = tuning_exp.with_name('evaluation')
    if lib.experiment.get_config_path(evaluation_exp).exists():
        existing = lib.experiment.load_config(evaluation_exp)
        if (
            resume
            and existing['function'] == evaluation_config['function']
            and existing['base_config'] == evaluation_config['base_config']
            and existing['n_seeds'] <= n_seeds
        ):
            # Preserve execution/storage preferences when extending an evaluation.
            existing['n_seeds'] = n_seeds
            lib.experiment.dump_config(evaluation_exp, existing)
        elif existing != evaluation_config:
            raise RuntimeError(
                'The evaluation experiment already exists and has a different config'
            )
    else:
        lib.experiment.create(evaluation_exp, config=evaluation_config, force=True)

    return main(evaluation_exp, force=force, resume=resume)


def script():
    lib.env.initialize()

    import cyclopts

    app = cyclopts.App()
    app.default(main)
    app.command(from_tuning)
    app(result_action='return_value')


if __name__ == '__main__':
    script()
