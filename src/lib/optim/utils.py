from collections.abc import Iterable
from typing import Any

import torch.nn as nn  # noqa: PLR0402


class ParameterGroups:
    def __init__(self):
        self._parameters = set()
        self._parameter_groups = []

    def add(self, params: Iterable[nn.Parameter], **hyperparameters) -> None:
        deduplicated_params = []
        for parameter in params:
            if parameter not in self._parameters:
                self._parameters.add(parameter)
                deduplicated_params.append(parameter)
        self._parameter_groups.append(
            {'params': deduplicated_params, **hyperparameters}
        )

    def unwrap(self) -> list[dict[str, Any]]:
        return self._parameter_groups
