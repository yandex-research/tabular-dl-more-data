from pathlib import Path

import lib.env
import lib.experiment
import lib.utils.misc
from lib.types import KWArgs


def _check_directory(directory: str | Path) -> Path:
    directory = Path(directory)
    if not directory.is_dir():
        raise RuntimeError(f'Not an existing directory: {directory}')

    experiments_dir = lib.env.get_experiments_dir()
    directory_resolved = directory.resolve()
    if (
        experiments_dir != directory_resolved
        and experiments_dir not in directory.resolve().parents
    ):
        raise RuntimeError(
            f'The directory "{directory}" is located outside the experiment root folder'
        )

    return directory


def _collect_exps(
    directories: list[str] | list[Path] | list[str | Path],
    *,
    fresh: bool = False,
    running: bool = False,
    completed: bool = False,
) -> list[Path]:
    directories = [_check_directory(x) for x in directories]

    if not fresh and not running and not completed:
        # List all experiments regardless of their statuses.
        fresh = True
        running = True
        completed = True

    exps = []
    for directory in directories:
        if lib.experiment.is_experiment(directory):
            exps.append(directory)
        else:
            exps.extend(
                x
                for x in directory.rglob('*')
                if lib.experiment.is_experiment(x)
                and (
                    (fresh and lib.experiment.is_fresh(x))
                    or (running and lib.experiment.is_running(x))
                    or (completed and lib.experiment.is_completed(x))
                )
            )

    return exps


def list_(
    directories: list[str] | list[Path] | list[str | Path],
    *,
    fresh: bool = False,
    running: bool = False,
    completed: bool = False,
    sort: bool = True,
) -> None:
    exps = _collect_exps(directories, fresh=fresh, running=running, completed=completed)
    if not exps:
        print('No experiments found')
        return

    if sort:
        exps.sort()

    for exp in exps:
        print(exp)
    print(f'\nFound {len(exps)} experiment{"s" if len(exps) > 1 else ""}')


def reset(
    directories: list[str] | list[Path] | list[str | Path],
    *,
    running: bool = False,
    completed: bool = False,
    yes: bool = False,
) -> None:
    if not running and not completed:
        # Reset all non-fresh experiments.
        running = True
        completed = True

    exps = _collect_exps(directories, fresh=False, running=running, completed=completed)
    if not exps:
        print('Nothing to reset')
        return

    for exp in exps:
        print(exp)

    if not yes:
        print(
            f'\nReset {len(exps)} experiment{"s" if len(exps) > 1 else ""}? [y/N]',
            end=' ',
        )
        if input() != 'y':
            print('User canceled the operation')
            return

    from tqdm import tqdm

    for exp in tqdm(exps, desc='Resetting'):
        lib.experiment.reset(exp)


def reproduce(
    exp: str | Path, *, force: bool = False, resume: bool = False, tmp: bool = False
) -> None:
    if tmp and force:
        raise ValueError('tmp and force cannot be True simultaneously')
    if tmp and resume:
        raise ValueError('tmp and resume cannot be True simultaneously')
    if not lib.experiment.is_completed(exp):
        raise RuntimeError(
            f'Cannot reproduce the experiment, because it is not completed: {exp}'
        )

    function = lib.utils.misc.import_(lib.experiment.load_report(exp)['function'])
    kwargs: KWArgs = (
        {'resume': resume}
        if lib.experiment.is_resumable_experiment_function(function)
        else {}
    )
    exp = Path(exp)
    config = lib.experiment.load_config(exp)

    if tmp:
        with lib.experiment.create_temporary(config) as tmp_exp:
            function(tmp_exp, **kwargs)
    else:
        reproduced_exp = lib.experiment.create(
            exp / 'reproduced', config=config, parents=True, force=force
        )
        reproduced_exp.joinpath('.gitignore').write_text('*\n')
        function(reproduced_exp, force=force, **kwargs)


def script():
    lib.env.initialize()

    import cyclopts

    app = cyclopts.App()
    app.command(list_, 'list')
    app.command(reset)
    app.command(reproduce)
    app(result_action='return_value')


if __name__ == '__main__':
    script()
