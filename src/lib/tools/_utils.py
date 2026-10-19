import pickle
from pathlib import Path
from typing import Any

import lib.experiment

type CheckpointDict = dict[str, Any]


def get_checkpoint_path(exp: str | Path) -> Path:
    return Path(exp) / 'checkpoint.pickle'


def dump_checkpoint(exp: str | Path, checkpoint: CheckpointDict) -> None:
    exp = lib.experiment.check(exp)
    with open(get_checkpoint_path(exp), 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(exp: str | Path) -> CheckpointDict:
    exp = lib.experiment.check(exp)
    with open(get_checkpoint_path(exp), 'rb') as f:
        return pickle.load(f)


def try_load_checkpoint(exp: str | Path) -> CheckpointDict | None:
    return load_checkpoint(exp) if get_checkpoint_path(exp).exists() else None
