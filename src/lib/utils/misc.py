import functools
import hashlib
import importlib
import inspect
import logging
import pickle
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lib._utils
import lib.env

_LOGGER = logging.getLogger(__name__)


def import_(fullname: str) -> Any:
    """Import an object by its full name.

    **Usage**

    >>> import_('pathlib.Path') is Path
    True
    """
    if fullname.count('.') == 0:
        raise ValueError('qualname must contain at least one dot')
    module_name, attr = fullname.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def cache_on_disk[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    this_fn_full_name = lib._utils.get_function_full_name(cache_on_disk)

    fn_source = inspect.getsource(fn)
    fn_full_name = lib._utils.get_function_full_name(fn)
    fn_signature = inspect.signature(fn)

    @functools.wraps(fn)
    def new_fn(*args: P.args, **kwargs: P.kwargs) -> R:
        key = {'$source': fn_source, **fn_signature.bind(*args, **kwargs).arguments}
        key_hash = hashlib.md5(str(key).encode('utf-8')).hexdigest()
        cwd = Path.cwd()
        path = (
            lib.env.get_cache_dir()
            / this_fn_full_name
            / fn_full_name
            / f'{key_hash}.pickle'
        )
        if path.exists():
            cached_key, cached_value = pickle.loads(path.read_bytes())
            if key != cached_key:
                raise RuntimeError(f'Hash collision for the cache keys at {path}')
            _LOGGER.debug(
                f'Function {fn_full_name} returned a cached output:'
                f' {path.relative_to(cwd) if cwd in path.parents else path}'
            )
            return cached_value
        else:
            value = fn(*args, **kwargs)
            with tempfile.NamedTemporaryFile('wb') as tmp_cache_file:
                tmp_cache_file.write(pickle.dumps((key, value)))
                tmp_cache_file.flush()
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tmp_cache_file.name, path)
            return value

    return new_fn


def add_frame(text: str) -> str:
    lines = text.splitlines()
    width = max(map(len, lines))
    hline = '─' * (width + 2)
    return '\n'.join(
        [
            f'╭{hline}╮',
            *(f'│ {line}{" " * (width - len(line))} │' for line in lines),
            f'╰{hline}╯',
        ]
    )
