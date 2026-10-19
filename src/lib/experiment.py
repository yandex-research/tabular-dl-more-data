import inspect
import json
import logging
import pprint
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lib._utils
import lib.env
import lib.utils.misc


_LOGGER = logging.getLogger(__name__)

type ReportDict = dict[str, Any]  # Must be JSON-serializable.
type ConfigDict = dict[str, Any]  # Must be JSON-serializable.

_CONFIG_FILE_NAME = 'config.json'
_INPUT_FILE_NAMES = frozenset([_CONFIG_FILE_NAME])


# >>> Utilities


def check(exp: str | Path) -> Path:
    exp = Path(exp)

    if not is_experiment(exp):
        if not exp.exists():
            raise ValueError(f'The experiment does not exist: {exp}')
        elif not exp.is_dir():
            raise ValueError(f'The path does not point to a directory: {exp}')
        else:
            raise ValueError(f'Invalid experiment directory: {exp}')

    return exp


# >>> Config


def get_config_path(exp: str | Path) -> Path:
    return Path(exp) / _CONFIG_FILE_NAME


def load_config(exp: str | Path) -> ConfigDict:
    exp = check(exp)
    with open(get_config_path(exp)) as f:
        return json.load(f)


def _dump_config_unchecked(exp: str | Path, config: ConfigDict) -> None:
    assert Path(exp).is_dir()
    with open(get_config_path(exp), 'w') as f:
        json.dump(config, f, indent=4)


def dump_config(exp: str | Path, config: ConfigDict) -> None:
    exp = check(exp)
    _dump_config_unchecked(exp, config)


def format_config(config: Any) -> str:
    return pprint.pformat(config)


# >>> Report


def get_report_path(exp: str | Path) -> Path:
    return Path(exp) / 'report.json'


def load_report(exp: str | Path) -> ReportDict:
    exp = check(exp)
    with open(get_report_path(exp)) as f:
        return json.load(f)


def dump_report(exp: str | Path, report: ReportDict) -> None:
    exp = check(exp)
    with open(get_report_path(exp), 'w') as f:
        json.dump(report, f, indent=4, allow_nan=False)


def format_report(report: ReportDict, *, float_precision: int = 4) -> str:
    def dict_to_lines(d: dict[str, Any], current_indent: int, lines: list[str]) -> None:
        for key, value in d.items():
            prefix = f'{" " * current_indent}{key}:'
            if isinstance(value, list):
                lines.append(f'{prefix} {"[...]" if value else "[]"}')
            elif isinstance(value, dict):
                lines.append(prefix)
                dict_to_lines(value, current_indent + 4, lines)
            else:
                lines.append(
                    f'{prefix}'
                    f' {round(value, float_precision) if isinstance(value, float) else value}'  # noqa: E501
                )

    lines = []
    dict_to_lines(report, 0, lines)
    return '\n'.join(lines)


# >>> Operations


def create(
    path: str | Path, *, config: ConfigDict, parents: bool = False, force: bool = False
) -> Path:
    path = Path(path)

    if path.exists():
        if force:
            shutil.rmtree(path)
        else:
            raise RuntimeError(f'The directory already exists: {path}')

    if parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_exp:
        tmp_exp = Path(tmp_exp)
        _dump_config_unchecked(tmp_exp, config)
        move(tmp_exp, path)

    return path


def create_temporary(config: ConfigDict, **kwargs) -> tempfile.TemporaryDirectory:
    directory = tempfile.TemporaryDirectory(**kwargs)
    create(directory.name, config=config, force=True)
    return directory


def reset(exp: str | Path) -> None:
    _LOGGER.info(f'Resetting the experiment: {exp}')

    exp = check(exp)
    for path in Path(exp).iterdir():
        if path.name not in _INPUT_FILE_NAMES:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def copy(exp: str | Path, dst_exp: str | Path, **copytree_kwargs) -> None:
    exp = check(exp)
    shutil.copytree(exp, dst_exp, **copytree_kwargs)


def move(exp: str | Path, dst_exp: str | Path) -> None:
    exp = check(exp)
    # Due to how `shutil.move` behaves when the destination path exists,
    # the case of existing `dst` is explicitly forbidden.
    if Path(dst_exp).exists():
        raise RuntimeError(f'The destination experiment path already exists: {dst_exp}')
    shutil.move(exp, dst_exp)


def remove(exp: str | Path) -> None:
    _LOGGER.info(f'Removing the experiment: {exp}')

    exp = check(exp)
    shutil.rmtree(exp)


def remove_tracked_files(exp: str | Path) -> None:
    """Remove files that are tracked by a VCS."""
    get_config_path(exp).unlink(missing_ok=True)
    get_report_path(exp).unlink(missing_ok=True)


# >>> Status


def is_experiment(path: str | Path) -> bool:
    return get_config_path(path).exists()


def is_fresh(exp: str | Path) -> bool:
    exp = check(exp)
    return frozenset(x.name for x in Path(exp).iterdir()) == _INPUT_FILE_NAMES


def _get_running_marker_path(exp: str | Path) -> Path:
    return Path(exp).joinpath('_RUNNING')


def _start_running(exp: str | Path) -> None:
    exp = check(exp)
    if not is_fresh(exp):
        raise RuntimeError(
            f'Cannot start the experiment, because it is not fresh: {exp}'
        )
    _get_running_marker_path(exp).touch()


def _stop_running(exp: str | Path) -> None:
    exp = check(exp)
    if not is_running(exp):
        raise RuntimeError(
            f'Cannot stop the experiment, because it is not running: {exp}'
        )
    _get_running_marker_path(exp).unlink()


def is_running(exp: str | Path) -> bool:
    exp = check(exp)
    return _get_running_marker_path(exp).exists()


def is_completed(exp: str | Path) -> bool:
    exp = check(exp)
    return not is_running(exp) and get_report_path(exp).exists()


# >>> ExperimentFunction


# The Python type system does not allow expressing the signature of
# `ExperimentFunction`, so `...` is used in place of arguments.
type ExperimentFunction = Callable[..., ReportDict | None]
"""The experiment function "protocol".

Any function satisfying the following signature is considered a valid experiment
function:

```
(
    exp: str | Path,
    *,
    force: bool = False,
    [resume: bool = False,]
    [<any other keyword-only arguments>]
) -> ReportDict | None
```
"""


def _check_experiment_function_argument(
    signature: inspect.Signature,
    name: str,
    *,
    required: bool,
    expected_annotation: Any,
    expected_default: Any,
) -> None:
    parameter = signature.parameters.get(name)

    if parameter is None:
        if required:
            raise TypeError(f'The experiment function is missing the argument `{name}`')
        else:
            return

    if parameter.annotation != expected_annotation:
        raise TypeError(
            f'The argument `{name}` of the experiment function has the type'
            f' annotation `{parameter.annotation}` instead of the expected '
            f' `{expected_annotation}`'
        )

    if expected_default is inspect.Parameter.empty:
        if parameter.default is not inspect.Parameter.empty:
            raise TypeError(
                f'The argument `{name}` of the experiment function must not have a'
                ' default value'
            )
    else:
        if parameter.default is inspect.Parameter.empty:
            raise TypeError(
                f'The argument `{name}` of the experiment function is missing'
                ' the default value'
            )
        if parameter.default != expected_default:
            raise TypeError(
                f'The default value of the argument `{name}` of the experiment function'
                f' is {parameter.default} instead of the expected {expected_default}'
            )


def check_experiment_function(function: Callable) -> ExperimentFunction:
    """Check that a function is a valid experiment function.

    If the function does not pass the check, `TypeError` is raised,
    indicating that the "type" of the function is not `ExperimentFunction`.
    """
    signature = inspect.signature(function)

    for p in signature.parameters.values():
        if p.name == 'exp':
            if p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TypeError(
                    'In the experiment function, the argument `exp` is not positional'
                )
        else:
            if p.kind != inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(
                    f'In the experiment function, the argument `{p.name}` is not'
                    ' keyword-only'
                )

    _check_experiment_function_argument(
        signature,
        'exp',
        required=True,
        expected_annotation=str | Path,
        expected_default=inspect.Parameter.empty,
    )
    _check_experiment_function_argument(
        signature,
        'force',
        required=True,
        expected_annotation=bool,
        expected_default=False,
    )
    _check_experiment_function_argument(
        signature,
        'resume',
        required=False,
        expected_annotation=bool,
        expected_default=False,
    )

    expected_return_annotation = ReportDict | None
    if signature.return_annotation != expected_return_annotation:
        raise ValueError(
            'The return type annotation of the experiment function is'
            f' `{signature.return_annotation}` instead of the expected'
            f' {expected_return_annotation}'
        )

    return function


def is_resumable_experiment_function(function: ExperimentFunction) -> bool:
    return 'resume' in inspect.signature(function).parameters


# >>> High-level API


def _make_title(exp: str | Path, function: ExperimentFunction) -> str:
    return f'{lib._utils.get_function_full_name(function)} @ {exp}'


def start(exp: str | Path, function: ExperimentFunction) -> None:
    check_experiment_function(function)
    exp = check(exp)

    print(f'>>> {_make_title(exp, function)}\n')
    _start_running(exp)


def resume(exp: str | Path, function: ExperimentFunction) -> None:
    check_experiment_function(function)
    if not is_resumable_experiment_function(function):
        raise ValueError('Cannot resume an experiment with a non-resumable function')
    exp = check(exp)

    print(f'>>> {_make_title(exp, function)}\n')


_resume_alias = resume


def try_start(
    exp: str | Path,
    function: ExperimentFunction,
    *,
    force: bool = False,
    resume: bool = False,
) -> bool:
    check_experiment_function(function)
    if force and resume:
        raise ValueError('force and resume must not be True simultaneously')

    if force and not is_fresh(exp):
        reset(exp)

    if is_completed(exp):
        return False
    elif is_fresh(exp):
        start(exp, function)
        return True
    elif resume:
        _resume_alias(exp, function)
        return True
    else:
        raise RuntimeError(
            'Cannot start the experiment from the "running" state.'
            ' Consider passing resume=True if the experiment function is resumable'
        )


def finish(
    exp: str | Path, function: ExperimentFunction, *, report: ReportDict
) -> None:
    check_experiment_function(function)
    exp = check(exp)

    dump_report(exp, report)
    _stop_running(exp)
    print(f'<<< {_make_title(exp, function)}')


def create_report(exp: str | Path, function: ExperimentFunction) -> ReportDict:
    report: ReportDict = {}
    data_config = load_config(exp).get('data', {})
    if 'val_key' in data_config:
        report['evaluation_roles'] = {
            'val': data_config['val_key'],
            'test': data_config['test_key'],
        }
    report['function'] = lib._utils.get_function_full_name(function)
    return report
