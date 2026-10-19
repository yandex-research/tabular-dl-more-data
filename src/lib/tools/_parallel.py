import contextlib
import itertools
import logging
import multiprocessing
import multiprocessing.managers
import os
import queue
import threading
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import lib.env
from lib.types import KWArgs

_LOGGER = logging.getLogger(__name__)
_CUDA_VISIBLE_DEVICES_VARNAME = 'CUDA_VISIBLE_DEVICES'

type _EnvironmentVariables = dict[str, str]


@dataclass(kw_only=True, frozen=True)
class _Worker:
    id: int
    lock: threading.Lock


_WORKER: _Worker | None = None


def _get_worker() -> _Worker | None:
    return _WORKER


def _set_worker(worker: _Worker) -> None:
    global _WORKER
    assert _WORKER is None, 'The worker is already set'
    _WORKER = worker


def get_worker_id() -> int | None:
    worker = _get_worker()
    return None if worker is None else worker.id


def lock() -> contextlib.AbstractContextManager:
    worker = _get_worker()
    return contextlib.nullcontext() if worker is None else worker.lock


def _make_cuda_visible_devices_pool(n_devices_per_worker: int) -> list[str]:
    """Prepare a pool of `CUDA_VISIBLE_DEVICES` values.

    **Usage**

    >>> os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    >>> _make_cuda_visible_devices_pool(1)
    ['0']
    >>> os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    >>> _make_cuda_visible_devices_pool(1)
    ['4', '5', '6', '7']
    >>> _make_cuda_visible_devices_pool(2)
    ['4,5', '6,7']
    >>> _make_cuda_visible_devices_pool(4)
    ['4,5,6,7']
    """
    cuda_visible_devices = os.environ.get(_CUDA_VISIBLE_DEVICES_VARNAME)
    if cuda_visible_devices is None:
        raise RuntimeError(f'{_CUDA_VISIBLE_DEVICES_VARNAME} must be set explicitly')

    # An empty value explicitly disables CUDA, including when tune/evaluate
    # use their default of one GPU per worker for a CPU-only experiment.
    if cuda_visible_devices == '':
        return ['']

    values = cuda_visible_devices.split(',')
    values_set = frozenset(values)
    # Only the simplest format of CUDA_VISIBLE_DEVICES is supported.
    if not (values_set <= {'0', '1', '2', '3', '4', '5', '6', '7'}):
        raise RuntimeError(
            f'{_CUDA_VISIBLE_DEVICES_VARNAME} must be a comma-separated sequence of'
            ' digits from 0 to 7 (inclusive)'
        )
    if len(values_set) != len(values):
        raise RuntimeError(
            f'{_CUDA_VISIBLE_DEVICES_VARNAME} contains duplicated values'
        )
    if len(values) % n_devices_per_worker != 0:
        raise ValueError(
            'The number of available GPUs must be a multiple of n_devices_per_worker'
        )
    values.sort(key=int)
    return [','.join(x) for x in itertools.batched(values, n_devices_per_worker)]


def _make_worker_queue(
    manager: multiprocessing.managers.SyncManager,
    *,
    n_workers: int,
    n_gpus_per_worker: int | None = None,
) -> queue.Queue[tuple[_Worker, _EnvironmentVariables]]:
    cuda_visible_devices_pool = (
        None
        if n_gpus_per_worker is None
        else _make_cuda_visible_devices_pool(n_gpus_per_worker)
    )
    workers = manager.Queue()
    lock = manager.Lock()
    for worker_id in range(n_workers):
        environ_update = {}
        if cuda_visible_devices_pool is not None:
            environ_update[_CUDA_VISIBLE_DEVICES_VARNAME] = cuda_visible_devices_pool[
                worker_id % len(cuda_visible_devices_pool)
            ]
        workers.put((_Worker(id=worker_id, lock=lock), environ_update))
    return workers


def _worker_initializer(
    workers: queue.Queue[tuple[_Worker, _EnvironmentVariables]],
) -> None:
    worker, environ_update = workers.get()
    _set_worker(worker)

    if _CUDA_VISIBLE_DEVICES_VARNAME in environ_update:
        import torch

        assert not torch.cuda.is_initialized()

    os.environ.update(environ_update)
    lib.env.initialize()
    _LOGGER.info(f'Worker {get_worker_id()} is initialized')


def _call[T](fn: Callable[..., T], kwargs: KWArgs) -> T:
    return fn(**kwargs)


def map[T](
    fn: Callable[..., T],
    kwargs_list: Iterable[KWArgs],
    *,
    n_workers: int,
    n_gpus_per_worker: int | None,
) -> list[T]:
    if _get_worker() is not None:
        raise RuntimeError('The function can be called only from the main process')
    if (
        n_gpus_per_worker is not None
        and _CUDA_VISIBLE_DEVICES_VARNAME not in os.environ
    ):
        raise RuntimeError(
            'If n_gpus_per_worker is not None, then the'
            f' "{_CUDA_VISIBLE_DEVICES_VARNAME}" environment variable must be set'
            ' explicitly'
        )

    n_cpus = multiprocessing.cpu_count()
    if n_workers > n_cpus:
        warnings.warn(
            f'The requested number of workers {n_workers}'
            f' exceeds the number of available hardware threads {n_cpus},'
            ' which can hurt efficiency'
        )

    # The "spawn" method is required for using PyTorch with CUDA
    # in the spawned processes.
    ctx = multiprocessing.get_context('spawn')
    with ctx.Manager() as manager:
        workers = _make_worker_queue(
            manager, n_workers=n_workers, n_gpus_per_worker=n_gpus_per_worker
        )
        with ctx.Pool(n_workers, _worker_initializer, (workers,)) as pool:
            return pool.starmap(_call, [(fn, kwargs) for kwargs in kwargs_list])
