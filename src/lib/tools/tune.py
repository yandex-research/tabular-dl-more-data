import dataclasses
import logging
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Literal

import optuna
import optuna.samplers
import optuna.trial

import lib._utils
import lib.env
import lib.experiment
import lib.tools._parallel
import lib.tools._utils
import lib.utils.dataclasses
import lib.utils.misc
import lib.utils.time
import lib.utils.torch
from lib.types import KWArgs

_LOGGER = logging.getLogger(__name__)

type ConfigSpace = dict[str, Any]


def _get_storage_url(study_path: str | Path) -> str:
    study_path = Path(study_path)
    assert study_path.suffix == '.sqlite'
    return f'sqlite:///{study_path}'


def _get_seed(base_seed: int, worker_id: int | None) -> int:
    # The `1000000` multiplier is needed to avoid overlapping worker seeds for runs
    # with adjacent base seeds (e.g. base_seed=0 and base_seed=1).
    return base_seed if worker_id is None else base_seed + 1000000 * worker_id


def _filter_trials(
    study_path: Path, *, keep_trial_ids: list[int], **study_kwargs
) -> None:
    # This function can remove trials from the study file, and thus must be called
    # only from the main process when there are no trials running in parallel.
    assert lib.tools._parallel.get_worker_id() is None, (
        'This function can be called only from the main process'
    )

    study_storage_url = _get_storage_url(study_path)
    study = optuna.load_study(study_name=None, storage=study_storage_url)

    # `all_trials` are sorted by ID,
    # which can be different from the order in `keep_trial_ids`
    # in the multi-process setup.
    all_trials = study.get_trials(deepcopy=False)
    id_to_trial = {t.number: t for t in all_trials}

    # Filter the trials by ID
    # and reorder the trials to follow the order in `keep_trial_ids`.
    trials = [id_to_trial[x] for x in keep_trial_ids]

    if len(trials) < len(all_trials):
        _LOGGER.info(
            f'Removing {len(all_trials) - len(trials)} trials'
            f' from the study "{study_path}"'
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_study_path = Path(tmp_dir) / study_path.name
            study = optuna.create_study(
                storage=_get_storage_url(tmp_study_path), **study_kwargs
            )
            study.add_trials(trials)
            tmp_study_path.rename(study_path)


def _sample_value(
    trial: optuna.trial.Trial,
    distribution: Literal['int', 'uniform', 'loguniform', 'categorical'],
    label: str,
    *args,
):
    trial_suggest, kwargs = {
        'int': (trial.suggest_int, {}),
        'uniform': (trial.suggest_float, {}),
        'loguniform': (trial.suggest_float, {'log': True}),
        'categorical': (trial.suggest_categorical, {}),
    }[distribution]
    if distribution in ('int', 'uniform', 'loguniform') and len(args) == 3:
        args, kwargs['step'] = args[:2], args[2]
    return trial_suggest(label, *args, **kwargs)


def _sample_config(
    trial: optuna.trial.Trial,
    space: bool | float | str | bytes | list | dict,
    label_parts: list,
) -> Any:
    if isinstance(space, bool | int | float | str | bytes):
        # This is a constant value, nothing to sample from.
        return space

    elif isinstance(space, list):
        if space and space[0] == '_tune_':
            # space: ["_tune_", distribution, arg_0, arg_1, ...]
            _, distribution, *args = space
            label = '.'.join(map(str, label_parts))

            # At this point, `distribution` can be one of the following:
            # 1. One of the built-in Optuna distributions expected in `_sample_value`.
            # 2. Same as 1., but prefixed with "?".
            # 3. Custom distributions. By convention, they start with "$".

            if distribution.startswith('?'):
                # space: ["_tune_", "?distribution", default_value, *actual_args]
                default, args_ = args[0], args[1:]
                if trial.suggest_categorical(f'?{label}', [False, True]):
                    return _sample_value(trial, distribution.lstrip('?'), label, *args_)
                else:
                    return default

            elif distribution == '$list':
                # space: ["_tune_", "$list", size, distribution, *actual_args]
                # A list of hyperparameters of a fixed size. For example, this
                # can be useful if a model allows configuring some hyperparameter
                # separately for each feature (in this case, `size` is the number
                # of features).
                size, item_distribution, *item_args = args
                return [
                    _sample_value(trial, item_distribution, f'{label}.{i}', *item_args)
                    for i in range(size)
                ]

            else:
                return _sample_value(trial, distribution, label, *args)

        else:
            return [
                _sample_config(trial, subspace, [*label_parts, i])
                for i, subspace in enumerate(space)
            ]

    elif isinstance(space, dict):
        return {
            key: _sample_config(trial, subspace, [*label_parts, key])
            for key, subspace in space.items()
        }


def _objective(
    trial: optuna.trial.Trial,
    *,
    exp: Path,
    function: lib.experiment.ExperimentFunction,
    output_keys: list[str],
    space: ConfigSpace,
    timer: lib.utils.time.Timer,
    save_experiment: bool,
) -> float:
    trial_exp = exp / 'trials' / str(trial.number)
    with lib.tools._parallel.lock():
        if trial_exp.exists():
            _LOGGER.warning(f'Removing the incomplete trial experiment: {trial_exp}')
            # `lib.experiment.remove` is not used, because `trial_exp` may be an
            # empty directory, which is not a valid experiment directory.
            shutil.rmtree(trial_exp)

    worker_id = lib.tools._parallel.get_worker_id()
    trial_config = _sample_config(trial, space, [])

    with lib.experiment.create_temporary(
        trial_config, suffix=f'_trial_{trial.number}'
    ) as tmp_exp:
        print()
        trial_report = function(tmp_exp)

        if save_experiment:
            lib.experiment.remove_tracked_files(tmp_exp)
            with lib.tools._parallel.lock():
                trial_exp.parent.mkdir(parents=True)
                lib.experiment.move(tmp_exp, trial_exp)

    assert trial_report is not None

    trial_report['tuning'] = {'trial_id': trial.number, 'time': timer.elapsed()}
    if worker_id is not None:
        trial_report['tuning']['worker_id'] = worker_id
    trial.set_user_attr('experiment', {'config': trial_config, 'report': trial_report})

    # Free the GPU memory (allocations, caches) left over from the trial's
    # experiment function before the next trial starts in this same process/worker.
    lib.utils.torch.free_memory()

    output = trial_report
    for key in output_keys:
        output = output[key]
    if not isinstance(output, float):
        raise RuntimeError(  # noqa: TRY004
            f'The objective output must be a float (the actual type is {type(output)})'
        )
    return output


def _summarize_study(
    study: optuna.study.Study,
    *,
    max_n_trials: int | None,
    previously_completed_trial_ids: list[int] | None,
) -> tuple[list[optuna.trial.FrozenTrial], optuna.trial.FrozenTrial]:
    """
    The function extracts:

    * A list of completed trials from `study`.
    * The best trial within this list.

    The list has the following properties:

    * Trials with IDs from `previously_completed_trial_ids` will go before
      the rest of the trials. This property is important for the multi-process setup,
      where the order of completion can differ from the order of IDs
      (note that `study.get_trials` returns trials sorted by IDs).
    * The size of the list will be at most `max_n_trials` (if provided).
      This is also important for the multi-process setup,
      where the number of trials can exceed `n_trials`.
    """
    # `get_trials` must be the only query to the underlying study storage,
    # because the study storage is concurrently updated by other workers.
    # Thus, for example, the best trial must be found manually,
    # without using `study.best_trial`.
    original_completed_trials = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    direction = study.direction
    del study

    previously_completed_trial_ids_set = frozenset(
        [] if previously_completed_trial_ids is None else previously_completed_trial_ids
    )
    completed_trials = []
    new_completed_trials = []
    for trial in original_completed_trials:
        if trial.number in previously_completed_trial_ids_set:
            completed_trials.append(trial)
        else:
            new_completed_trials.append(trial)

    if max_n_trials is not None:
        if len(completed_trials) < max_n_trials:
            new_completed_trials = new_completed_trials[
                : max_n_trials - len(completed_trials)
            ]
        else:
            new_completed_trials = []

    completed_trials.extend(new_completed_trials)
    if max_n_trials is not None:
        # At this point, the following condition must always be true.
        assert len(completed_trials) <= max_n_trials

    if direction == optuna.study.StudyDirection.MAXIMIZE:
        find_best = max
    elif direction == optuna.study.StudyDirection.MINIMIZE:
        find_best = min
    else:
        raise ValueError(f'The study direction "{direction}" is not supported')
    best_trial = find_best(completed_trials, key=lambda x: x.value)

    return completed_trials, best_trial


# The only purpose of the following function is to create a checkpoint based on the
# content of the study file. Thus, in the multi-process setup, it is not important in
# what order workers call this function.
def _callback(
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
    *,
    exp: Path,
    n_trials: int | None,
    timer: lib.utils.time.Timer,
    track_experiments: bool,
):
    worker_id = lib.tools._parallel.get_worker_id()

    # All operations with the experiment directory must be performed under the lock.
    with lib.tools._parallel.lock():
        checkpoint = lib.tools._utils.try_load_checkpoint(exp)
        if checkpoint is None:
            checkpoint = {}

        # `_summarize_study` must be the only place where the study storage is queried.
        completed_trials, best_trial = _summarize_study(
            study,
            max_n_trials=n_trials,
            previously_completed_trial_ids=checkpoint.get('completed_trial_ids'),
        )
        checkpoint['completed_trial_ids'] = [x.number for x in completed_trials]

        # The following check ensures the overall correctness of the implementation,
        # especially when resuming from a checkpoint in the multi-process setup.
        # Note that there is a small chance of identical trials, when the processes
        # is interrupted just after one of the workers commits its completed trial
        # but before this worker saves a checkpoint with its new random state.
        # If this happens frequently, it should be possible to solve this issue
        # by baking the number of restarts into the random seed:
        # `seed = base_seed + 1000 * n_restarts + 1000000 * worker_id`
        for t in completed_trials:
            if t.number != trial.number:
                assert t.params != trial.params, (
                    f'The trials {t.number} and {trial.number}'
                    f' have identical sampled parameters: {trial.params}'
                )
        # The trial is not needed beyond the above check.
        del trial

        report = lib.experiment.load_report(exp)
        report['best'] = best_trial.user_attrs['experiment']
        report['time'] = timer.elapsed()
        report['n_completed_trials'] = len(completed_trials)
        if worker_id is not None:
            report['worker_id'] = worker_id
        if track_experiments:
            report['experiments'] = [
                x.user_attrs['experiment'] for x in completed_trials
            ]

        checkpoint['report'] = report
        checkpoint['timer'] = timer
        if worker_id is None:
            checkpoint['sampler'] = study.sampler
        else:
            workers_checkpoint = checkpoint.setdefault('workers', {})
            workers_checkpoint[worker_id] = {
                # Saving worker states does not give any reproducibility.
                # It only makes it possible to resume the run with diverse
                # states across workers, and can potentially help with debugging.
                'sampler': study.sampler,
                'timer': timer,
            }

        lib.tools._utils.dump_checkpoint(exp, checkpoint)
        lib.experiment.dump_report(exp, report)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ObjectiveConfig:
    function: str
    output_keys: list[str]

    def __post_init__(self):
        if not self.output_keys:
            raise ValueError('output_keys must not be empty')


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    # `seed` is used exclusively for seeding the Optuna sampler. It is stored separately
    # (instead of being a required part of the `sampler` field), because in
    # a multi-process setup, it is _not_ passed directly to the sampler, but instead is
    # used as the base seed for computing per-worker seeds.
    seed: int
    objective: ObjectiveConfig
    space: ConfigSpace
    sampler: KWArgs | None = None
    n_trials: int | None = None
    timeout: int | None = None

    # `n_workers` defines the number of processes ("workers") that will be evaluating
    # Optuna trials in parallel, which can significantly accelerate the tuning.
    # Carefully read this comment, in particular the "Limitations" paragraph,
    # before using this option.
    #
    # **General notes**
    #
    # - Lightweight workloads (e.g. tuning a plain MLP) that underutilize GPU can
    #   benefit from `n_workers > 1` even when all workers are running on the same GPU.
    #   Heavy workloads that fully utilize GPU can still benefit from `n_workers > 1`
    #   by running each worker on a separate GPU (see the next point).
    #
    # - To use multiple GPUs, set the `CUDA_VISIBLE_DEVICES` environment variable
    #   accordingly. For example, when `n_workers=4` and the command is as follows:
    #   ```
    #   CUDA_VISIBLE_DEVICES="0,1" uv run bin/tune.py path/to/exp
    #   ```
    #   then two workers will run with `CUDA_VISIBLE_DEVICES="0"`,
    #   and two workers will run with `CUDA_VISIBLE_DEVICES="1"`.
    #
    # **Practical tips**
    #
    # - Ensure that the machine has at least `n_workers + 1` CPU cores.
    # - If one decides to use `n_workers`, then `n_workers=2` is a reasonable default
    #   value. Even for plain MLPs, setting `n_workers` to higher values does not result
    #   in _significant_ speed-ups.
    # - The execution time scales well with the number of GPUs, i.e. using
    #   N GPUs instead of 1 will reduce the tuning time almost by the factor of N.
    #
    # **Limitations**
    #
    # - Setting `n_workers > 4` can hurt the tuning results performance. Currently,
    #   there is no single explanation for why this happens. One hypothesis is that
    #   the multi-process tuning trajectory "lags" behind the single-process one,
    #   and the lag increases with the number of workers.
    # - When `n_workers` is used, bitwise reproducibility is not possible anymore.
    # - When `n_workers` is used, interrupting the tuning and resuming it from a
    #   checkpoint affects the optimization trajectory. For example, in the edge case
    #   of interrupting and resuming the run after each trial, the tuning behaves
    #   as there is only one worker.
    # - The workers are not allowed to spawn new processes. That in particular means
    #   that `torch.utils.data.DataLoader(..., num_workers > 0)` will lead to an error.
    #
    # P.S. One more technical thing to keep in mind. In `optuna.samplers.TPESampler`,
    # the argument `n_startup_trials` defines the number of _completed and pruned_
    # trials before activating the TPE algorithm, i.e. running trials are _not_ taken
    # into account. For example, `n_workers=32` and `n_startup_trials=20` will lead to
    # 32 random trials followed by 19 more (51 in total) random trials before starting
    # the TPE phase (modulo edge cases when workers finish trials simultaneously).
    n_workers: int | None = None

    # The following option is implemented only for completeness. At the moment of
    # writing, the codebase does not offer scripts that run on multiple GPUs.
    n_gpus_per_worker: int | None = None

    track_experiments: bool = False
    save_experiments: bool = False

    def __post_init__(self):
        if self.n_gpus_per_worker is not None and self.n_workers is None:
            raise ValueError(
                'The `n_gpus_per_worker` option cannot be passed'
                ' without the `n_workers` option'
            )


def _make_sampler(config: Config, worker_id: int | None) -> optuna.samplers.BaseSampler:
    sampler_config = {} if config.sampler is None else config.sampler.copy()
    sampler_cls = getattr(optuna.samplers, sampler_config.pop('type', 'TPESampler'))
    return sampler_cls(seed=_get_seed(config.seed, worker_id), **sampler_config)


def _worker_main(exp: Path, config: Config, study_storage_url: str) -> None:
    worker_id = lib.tools._parallel.get_worker_id()

    objective_function = lib.utils.misc.import_(config.objective.function)
    objective_function = lib.experiment.check_experiment_function(objective_function)
    n_remaining_trials = config.n_trials
    timeout = config.timeout

    # All operations with the experiment directory must be performed under the lock.
    with lib.tools._parallel.lock():
        study = optuna.load_study(
            study_name=None,
            storage=study_storage_url,
            sampler=_make_sampler(config, worker_id),
        )
        checkpoint = lib.tools._utils.try_load_checkpoint(exp)

    if checkpoint is None:
        timer = lib.utils.time.Timer()

    else:
        del checkpoint['completed_trial_ids']
        if n_remaining_trials is not None:
            n_remaining_trials -= checkpoint.pop('report')['n_completed_trials']

        # Each worker resumes with the latest saved timer.
        timer = checkpoint.pop('timer')
        if timeout is not None:
            timeout -= timer.elapsed()

        if worker_id is None:
            study.sampler = checkpoint.pop('sampler')
        else:
            workers_checkpoint = checkpoint.pop('workers')
            if worker_id in workers_checkpoint:
                # Restoring worker states does not give any reproducibility.
                # This is done only to obtain constructive diverse starting points.
                study.sampler = workers_checkpoint[worker_id]['sampler']

        assert not checkpoint, (
            f'Some checkpoint fields were not used, namely: {sorted(checkpoint)}'
        )

    callbacks: list[Any] = [
        partial(
            _callback,
            exp=exp,
            n_trials=config.n_trials,
            timer=timer,
            track_experiments=config.track_experiments,
        )
    ]
    if config.n_trials is not None:
        # `study.optimize` runs `n_trials` in the current process, i.e. _per worker_.
        # The following callback stops the study after achieving `n_trials` _globally_.
        # However, workers will commit their latest active trials anyway, thus usually
        # exceeding `n_trials`. This is taken into account in two places:
        # - The extra trials are removed in `_summarize_study`.
        # - `_filter_trials` is used in `main`.
        callbacks.append(optuna.study.MaxTrialsCallback(config.n_trials))

    # The study optimization loop roughly looks as follows:
    #
    # 1. Create a trial. Mark its state as "running" and add to the study storage.
    # 2. Call the objective function. The objective function is allowed to modify
    #    the trial object, e.g. by setting user attributes.
    # 3. Finalize all the trial data in the study storage.
    # 4. Call the callbacks with a read-only view of the trial.
    #
    # In the _single-process_ setup, a list of trial statuses in the study storage
    # usually looks like this:
    #
    # [COMPLETE, COMPLETE, ..., COMPLETE, <RUNNING OR COMPLETE>]
    #  ID=0      ID=1           ID=N      ID=N+1
    #
    # The above means that the order of trial IDs (`trial.number`) follows the order
    # in which the trials are completed. By contrast, in the _multi-process_ setup,
    # the following is possible (note that the trial N+1 is already completed,
    # while the trial N is still running):
    #
    # [COMPLETE, COMPLETE, ..., RUNNING, COMPLETE, ...]
    #  ID=0      ID=1           ID=N     ID=N+1
    #
    # From the "algorithmic" perspective, the order of completion is more important
    # (e.g. for analyzing how the best score evolves during the optimization process).
    # This is why there is the `completed_trial_ids` list in the checkpoint storing
    # the trials IDs in the order of completion, which is used, for example,
    # in `_filter_trials`.
    timer.run()
    study.optimize(
        partial(
            _objective,
            exp=exp,
            function=objective_function,
            output_keys=config.objective.output_keys,
            space=config.space,
            timer=timer,
            save_experiment=config.save_experiments,
        ),
        n_trials=n_remaining_trials,
        timeout=timeout,
        callbacks=callbacks,
    )


def main(
    exp: str | Path, *, force: bool = False, resume: bool = False
) -> lib.experiment.ReportDict | None:
    if not lib.experiment.try_start(exp, main, force=force, resume=resume):
        print(f'The experiment is already completed: {exp}')
        return None

    exp = lib.experiment.check(exp)
    config = lib.utils.dataclasses.from_dict(Config, lib.experiment.load_config(exp))
    report = lib.experiment.create_report(exp, main)
    print(f'{lib.experiment.format_config(config)}\n')

    study_path = exp / 'study.sqlite'
    study_storage_url = _get_storage_url(study_path)
    study_kwargs: KWArgs = {
        'study_name': lib._utils.get_function_full_name(main),
        'direction': 'maximize',
    }
    n_completed_trials = 0

    checkpoint = lib.tools._utils.try_load_checkpoint(exp)
    if checkpoint is None:
        if study_path.exists():
            study_path.unlink()
        # The sampler will be set in `_worker_main`.
        optuna.create_study(storage=study_storage_url, **study_kwargs)
    else:
        report = checkpoint['report']
        n_completed_trials = checkpoint['report']['n_completed_trials']
        print(
            'Resuming from a checkpoint.'
            f' Completed {n_completed_trials}'
            f'/{"inf" if config.n_trials is None else config.n_trials} trials'
        )
        # Remove incomplete trials from the study.
        _filter_trials(
            study_path,
            keep_trial_ids=checkpoint['completed_trial_ids'],
            **study_kwargs,
        )
        report.setdefault('resumed_after_n_trials', []).append(n_completed_trials)

    lib.experiment.dump_report(exp, report)
    del report

    if config.n_workers is None:
        _worker_main(exp, config, study_storage_url)
    else:
        n_workers = config.n_workers
        if config.n_trials is not None:
            n_remaining_trials = config.n_trials - n_completed_trials
            if config.n_workers > n_remaining_trials:
                _LOGGER.info(
                    f'Reducing the number of workers'
                    f' from {n_workers=} to {n_remaining_trials=}'
                )
                n_workers = n_remaining_trials
        _LOGGER.info(f'The number of workers: {n_workers}')

        lib.tools._parallel.map(
            _worker_main,
            [
                {
                    'exp': exp,
                    'config': config,
                    'study_storage_url': study_storage_url,
                }
                for _ in range(n_workers)
            ],
            n_workers=n_workers,
            n_gpus_per_worker=(
                1 if config.n_gpus_per_worker is None else config.n_gpus_per_worker
            ),
        )

    # In the multi-process setup, the total number of trials usually exceeds
    # config['n_trials']. The following function call prunes the extra trials.
    _filter_trials(
        study_path,
        keep_trial_ids=lib.tools._utils.load_checkpoint(exp)['completed_trial_ids'],
        **study_kwargs,
    )

    report = lib.experiment.load_report(exp)
    print(f'Report\n{lib.utils.misc.add_frame(lib.experiment.format_report(report))}\n')
    lib.experiment.finish(exp, main, report=report)
    return report


def script():
    lib.env.initialize()

    import cyclopts

    cyclopts.run(main, result_action='return_value')


if __name__ == '__main__':
    script()
