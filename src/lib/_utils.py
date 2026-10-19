import inspect
from collections.abc import Callable
from pathlib import Path

import lib.env


def get_function_full_name(function: Callable) -> str:
    """Get the full function name.

    **Usage**

    >>> get_function_full_name(get_function_full_name)
    'lib._utils.get_function_full_name'
    """
    module = inspect.getmodule(function)
    if module is None:
        raise RuntimeError('Failed to locate the module of the function.')

    module_path = getattr(module, '__file__', None)
    if module_path is None:
        raise RuntimeError(
            'Failed to locate the module of the function.'
            ' This can happen if the code is running in a Jupyter notebook.'
        )
    module_path = Path(module_path).resolve()

    src_dir = lib.env.get_src_dir()
    if src_dir not in module_path.parents:
        raise RuntimeError(
            f'The module of the function must be located within "{src_dir}"'
        )

    module_full_name = str(module_path.relative_to(src_dir).with_suffix('')).replace(
        '/', '.'
    )
    return f'{module_full_name}.{function.__name__}'  # ty: ignore[unresolved-attribute]
