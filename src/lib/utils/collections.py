from collections.abc import Iterable
from typing import Any


def _flatten_dict(d: dict, key_prefix: str, result: dict) -> None:
    for k, v in d.items():
        new_k = f'{key_prefix}.{k}' if key_prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, new_k, result)
        else:
            if result.setdefault(new_k, v) is not v:
                raise RuntimeError(
                    'Different parts of the dictionary resulted'
                    f' in the same flat key "{new_k}"'
                )


def flatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    flat_d: dict[str, Any] = {}
    _flatten_dict(d, '', flat_d)
    return flat_d


def transpose_dicts[K, V](iterable: Iterable[dict[K, V]]) -> dict[K, list[V]]:
    iterator = iter(iterable)

    try:
        first = next(iterator)
    except StopIteration:
        raise RuntimeError('The input must be non-empty')

    transposed = {k: [v] for k, v in first.items()}
    keys = transposed.keys()
    for i, dictionary in enumerate(iterator):
        if dictionary.keys() != keys:
            raise RuntimeError(
                f'Inconsistent keys among the input dictionaries.'
                f' The keys of the first dictionary: {keys}.'
                f' The keys of the dictionary {i}: {dictionary.keys()}.'
            )
        for key, value in dictionary.items():
            transposed[key].append(value)
    return transposed
