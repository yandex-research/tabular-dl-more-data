# By convention, this submodule must be:
#
# - Importable as an independent module from arbitrary (PEP-723-based) scripts.
# - Always cheap to import.
#
# As such, this submodule must not import any third-party modules.

import os
from pathlib import Path


_PROJECT_DIR: Path | None = None
_IS_NOTEBOOK = None
_INITIALIZED = False

_PYPROJECT_FILE_NAME = 'pyproject.toml'


def get_project_dir() -> Path:
    global _PROJECT_DIR

    if _PROJECT_DIR is None:
        path = Path.cwd()
        pyproject_path = path / _PYPROJECT_FILE_NAME
        while str(path) != path.root and not pyproject_path.exists():
            path = path.parent
            pyproject_path = path / _PYPROJECT_FILE_NAME

        if pyproject_path.exists():
            if pyproject_path.parent != Path(__file__).parent.parent.parent:
                raise RuntimeError(
                    'Failed to find the project directory. '
                    f' Most likely, you are running the code'
                    ' in a virtual environment of a different project,'
                    f' namely, of this one: {pyproject_path.parent}'
                )
            _PROJECT_DIR = path
        else:
            raise RuntimeError(
                'Failed to find the project directory.'
                ' Most likely, you are outside of the project directory.'
            )

    return _PROJECT_DIR


def get_cache_dir() -> Path:
    path = get_project_dir() / '.cache'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    return get_project_dir() / 'data'


def get_experiments_dir() -> Path:
    return get_project_dir() / 'experiments'


def get_src_dir() -> Path:
    return get_project_dir() / 'src'


def is_notebook() -> bool:
    global _IS_NOTEBOOK

    if _IS_NOTEBOOK is None:
        try:
            import IPython
        except ImportError:
            _IS_NOTEBOOK = False
        else:
            ipython = IPython.get_ipython()  # pyright: ignore[reportPrivateImportUsage]
            _IS_NOTEBOOK = (
                ipython is not None
                and ipython.__class__.__name__ == 'ZMQInteractiveShell'
            )

    return _IS_NOTEBOOK


def initialize() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        raise RuntimeError('Initialization has already been done')

    # Check the working directory.
    project_dir = get_project_dir()
    if is_notebook():
        os.chdir(project_dir)
    if Path.cwd() != project_dir:
        raise RuntimeError('The code must be run from the project root')

    # Set up logging.
    import lib.utils.logging as _lib_utils_logging

    _lib_utils_logging.configure()

    _INITIALIZED = True
