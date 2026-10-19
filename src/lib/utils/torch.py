import contextlib
import functools
import gc
import inspect
import logging
import os
import random
from collections.abc import Callable
from typing import Any

import torch

_LOGGER = logging.getLogger(__name__)


# Inspired by: https://github.com/Yura52/delu/blob/a76e3139f83703b02500f96b9031cd592619d28e/delu/cuda.py
def free_memory() -> None:
    torch.compiler.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def get_device() -> torch.device:
    return torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'mps:0'
        if (
            torch.mps.is_available()
            # The requirement to set the following environment variable to use MPS
            # is an informal convention of this project.
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'
        )
        else 'cpu'
    )


@contextlib.contextmanager
def set_num_threads(n: int):
    current_num_threads = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(current_num_threads)


# References:
# - https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
# - https://github.com/Lightning-AI/pytorch-lightning/pull/6960#issuecomment-819672341
# - https://github.com/Yura52/delu/blob/a76e3139f83703b02500f96b9031cd592619d28e/delu/random.py#L17
def seed(base_seed: int) -> None:
    _2_pow_64 = 1 << 64
    if not (0 <= base_seed < _2_pow_64):
        raise ValueError(
            '`base_seed` must be a non-negative integer from [0, 2**64).'
            f' The provided value: {base_seed=}'
        )

    import numpy as np

    sequence = np.random.SeedSequence(base_seed)

    def generate_state(*args, **kwargs) -> np.ndarray:
        new_sequence = sequence.spawn(1)[0]
        return new_sequence.generate_state(*args, **kwargs)

    # To generate a 128-bit seed for the standard library,
    # two uint64 numbers are generated and concatenated (literally).
    stdlib_state = generate_state(2, dtype=np.uint64).tolist()
    random.seed(stdlib_state[0] * _2_pow_64 + stdlib_state[1])
    np.random.seed(generate_state(4))
    torch.manual_seed(generate_state(1, dtype=np.uint64).item())


# Inspired by: https://github.com/Yura52/delu/blob/a76e3139f83703b02500f96b9031cd592619d28e/delu/random.py#L106
def get_random_state() -> dict[str, Any]:
    import numpy as np

    return {
        'random': random.getstate(),
        'numpy.random': np.random.get_state(),
        'torch.random': torch.random.get_rng_state(),
        'torch.cuda': torch.cuda.get_rng_state_all(),
    }


# Inspired by: https://github.com/Yura52/delu/blob/a76e3139f83703b02500f96b9031cd592619d28e/delu/random.py#L169
def set_random_state(state: dict[str, Any]) -> None:
    import numpy as np

    random.setstate(state['random'])
    np.random.set_state(state['numpy.random'])
    torch.random.set_rng_state(state['torch.random'])
    torch.cuda.set_rng_state_all(state['torch.cuda'])


# Inspired by: https://github.com/Yura52/delu/blob/a76e3139f83703b02500f96b9031cd592619d28e/delu/random.py#L199
@contextlib.contextmanager
def preserve_random_state():
    state = get_random_state()
    try:
        yield
    finally:
        set_random_state(state)


def _is_cuda_oom_exception(err: RuntimeError) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


def adjust_gpu_memory_usage[**P, T](
    memory_parameter: str,
) -> Callable[[Callable[P, T]], Callable[P, tuple[T, int]]]:
    def decorator(f: Callable[P, T]) -> Callable[P, tuple[T, int]]:
        p = inspect.signature(f).parameters.get(memory_parameter)
        if p is None or p.kind != inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                f'The function must have the keyword-only argument "{memory_parameter}"'
            )
        del p

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, int]:
            value: int = kwargs[memory_parameter]  # pyright: ignore[reportAssignmentType]  # ty:ignore[invalid-assignment]
            if value <= 0:
                raise ValueError(f'{memory_parameter} must be a positive integer')
            while value:
                kwargs[memory_parameter] = value  # ty: ignore[invalid-assignment]
                try:
                    return f(*args, **kwargs), value
                except RuntimeError as err:
                    if not _is_cuda_oom_exception(err):
                        raise
                    new_value = value // 2
                    message = (
                        f'Calling the function `{f.__name__}`'  # ty: ignore[unresolved-attribute]
                        f' with {memory_parameter}={value} triggers GPU OOM'
                    )
                    if new_value:
                        message += f'. Retrying with {memory_parameter}={new_value}'
                    _LOGGER.warning(message)
                    value = new_value
            raise RuntimeError(f'Not enough memory even for {memory_parameter}=1')

        return wrapper

    return decorator
