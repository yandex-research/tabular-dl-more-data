import dataclasses
import enum
import functools
import types
import typing
from pathlib import Path
from typing import Any


def _parse_object_exception_handler(parse_object_fn) -> Any:
    """Add a note with the full key path to the innermost exception."""

    @functools.wraps(parse_object_fn)
    def wrapper(type_, object_, key):
        try:
            return parse_object_fn(type_, object_, key)
        except Exception as err:
            if key and not hasattr(err, '__notes__'):
                err.add_note(f'Problematic key: {".".join(key)}')
            raise

    return wrapper


@_parse_object_exception_handler
def _parse_object(annotation: Any, object_: Any, key: tuple[str, ...]):
    # Unpack the original type.
    while isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__

    # Handle types.
    if isinstance(annotation, type):
        # Handle Any.
        if annotation is Any:
            return object_

        # Handle simple types.
        elif annotation in (
            types.NoneType,
            types.EllipsisType,
            bool,
            int,
            float,
            str,
            bytes,
        ):
            # Check the object type strictly.
            if type(object_) is not annotation:
                raise TypeError(
                    f'The expected object type is {annotation}, but the actual type is'
                    f' {type(object_)}'
                )
            return object_

        # Handle paths.
        elif annotation is Path:  # noqa: SIM114
            return annotation(object_)

        # Handle enums.
        elif issubclass(annotation, enum.Enum):
            return annotation(object_)

        # Handle dataclasses.
        elif dataclasses.is_dataclass(annotation):
            if isinstance(object_, annotation):
                # The object is already an instance of the dataclass.
                return object_
            if not isinstance(object_, dict):
                raise TypeError(
                    'To be parsed as a dataclass, the object must be either an instance'
                    ' of the dataclass or a dictionary'
                )

            # Collect dataclass initialization fields.
            init_fields = {}
            for field in dataclasses.fields(annotation):
                if not field.init:
                    if field.name in object_:
                        raise ValueError(
                            f'The object contains a non-init field {field.name}'
                            f' of the dataclass, which is not allowed'
                        )
                    continue
                if (
                    field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                    and field.name not in object_
                ):
                    raise ValueError(
                        f'The object is missing the required field "{field.name}"'
                        f' of the dataclass {annotation}'
                    )
                init_fields[field.name] = field

            # Parse the object's nested values.
            object_with_parsed_values = {}
            for k, v in object_.items():
                field = init_fields.get(k)
                if field is None:
                    raise ValueError(
                        f'The dataclass {annotation} does not have the field "{k}"'
                    )
                object_with_parsed_values[k] = _parse_object(field.type, v, (*key, k))
                del k, v

            # Return a dataclass instance.
            return annotation(**object_with_parsed_values)

        else:
            raise ValueError(f'{annotation=} is not supported')

    # Handle type-like annotations.
    else:
        type_origin = typing.get_origin(annotation)
        type_args = typing.get_args(annotation)

        # Handle sequences and sets.
        if type_origin in (tuple, list, set, frozenset):
            if not type_args:
                raise ValueError(
                    f'The provided {annotation=} is missing type parameters'
                )
            # Handle all cases except for unnamed fixed-size tuples.
            if not issubclass(type_origin, tuple) or (
                len(type_args) == 2 and type_args[1] is ...
            ):
                return type_origin(
                    _parse_object(type_args[0], x, (*key, str(i)))
                    for i, x in enumerate(object_)
                )
            else:
                # Handle unnamed fixed-size tuples.
                if len(object_) != len(type_args):
                    raise ValueError(
                        f'The expected object size is {len(type_args)},'
                        f' but the actual size is {len(object_)}'
                    )
                return type_origin(
                    _parse_object(type_args[i], x, (*key, str(i)))
                    for i, x in enumerate(object_)
                )

        # Handle dictionaries.
        elif type_origin is dict:
            if len(type_args) < 2:
                raise ValueError(
                    f'The provided {annotation=} is missing type parameters'
                )
            return type_origin(
                (
                    _parse_object(type_args[0], k, (*key, k, '<key>')),
                    _parse_object(type_args[1], v, (*key, k, '<value>')),
                )
                for k, v in object_.items()
            )

        # Handle literals.
        elif type_origin is typing.Literal:
            object_type = type(object_)
            if any(
                type(x) is object_type and x == object_
                for x in typing.get_args(annotation)
            ):
                return object_
            else:
                raise ValueError(
                    f'The object {object_} is not a valid instance of {annotation}'
                )

        # Handle unions.
        elif type_origin in (
            types.UnionType,  # T1 | T2 | ...
            typing.Union,  # typing.Optional[T], typing.Union[T1, T2, ...]
        ):
            for type_candidate in type_args:
                try:
                    return _parse_object(type_candidate, object_, key)
                except Exception:  # noqa: BLE001, S110
                    pass
            raise ValueError(
                f'The object does not match any of the type variants {type_args}'
            )

        else:
            raise ValueError(f'{annotation=} is not supported')


def from_dict[T](datacls: type[T], dict_: dict[str, Any]) -> T:
    """Create a dataclass instance from a dictionary and perform type checks.

    Supported field types:

    * Simple types: `None`, `...`, `bool`, `int`, `float`, `str`, `bytes`
    * `pathlib.Path`
    * Enums
    * Optional fields
    * Unions
    * Literals
    * Dataclasses
    * Built-in collections: `tuple`, `list`, `set`, `frozenset`, `dict`.
    * All combinations of the above field types. For example:
      `int | dict[str, tuple[MyDataClass, Path]] | None`

    .. note::
        All underlying type checks compare the content types _exactly_ against
        the expected types (i.e. using `is`, not `isinstance`).
    """
    if not dataclasses.is_dataclass(datacls):
        raise ValueError('The first argument must be a dataclass')

    return _parse_object(datacls, dict_, ())
