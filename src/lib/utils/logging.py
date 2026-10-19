import logging
import os


def configure():
    level_name = os.environ.get('LIB_LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f'Unknown logging level: {level_name}')  # noqa: TRY004

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
