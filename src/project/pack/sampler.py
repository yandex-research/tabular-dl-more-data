"""Independent, resumable shuffled batch streams for packed models."""

from typing import Any

import torch
from torch import Tensor


class PackedEpochSampler:
    """Yield shuffled batches, sharing a permutation between equal member seeds.

    The current epoch permutation is not stored in checkpoints.  Instead, the
    generator states from the beginning of the epoch and the current batch offset
    are stored, which is enough to reconstruct the permutation after a restart.
    This keeps checkpoints independent of the training-set size.
    """

    def __init__(
        self,
        *,
        train_size: int,
        batch_size: int,
        seeds: list[int],
        device: torch.device,
    ) -> None:
        if train_size <= 0:
            raise ValueError('train_size must be positive')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if not seeds:
            raise ValueError('seeds must not be empty')

        self.train_size = train_size
        self.batch_size = batch_size
        self.device = device
        unique_seeds = list(dict.fromkeys(seeds))
        seed_to_stream = {seed: index for index, seed in enumerate(unique_seeds)}
        self._member_streams = [seed_to_stream[seed] for seed in seeds]
        self._stream_first_members = [seeds.index(seed) for seed in unique_seeds]
        self._member_stream_indices = torch.tensor(self._member_streams, device=device)
        self._generators = [
            torch.Generator(device).manual_seed(seed) for seed in unique_seeds
        ]
        self._orders: Tensor | None = None
        self._epoch_start_generator_states: list[Tensor] | None = None
        self._batch_index = 0

    @property
    def epoch_size(self) -> int:
        return (self.train_size + self.batch_size - 1) // self.batch_size

    def _make_orders(self, generators: list[torch.Generator]) -> Tensor:
        return torch.stack(
            [
                torch.randperm(
                    self.train_size,
                    generator=generator,
                    device=self.device,
                )
                for generator in generators
            ]
        )

    def _start_epoch(self) -> None:
        self._epoch_start_generator_states = [
            generator.get_state() for generator in self._generators
        ]
        self._orders = self._make_orders(self._generators)
        self._batch_index = 0

    def next(self, *, shared: bool = False) -> Tensor:
        """Return member indices, or one 1D batch when all members share a seed.

        ``shared=True`` lets the runner gather/encode that batch only once. With
        multiple distinct seeds the result retains the original 2D member layout.
        """
        if self._orders is None or self._batch_index >= self.epoch_size:
            self._start_epoch()

        assert self._orders is not None
        begin = self._batch_index * self.batch_size
        end = min(begin + self.batch_size, self.train_size)
        self._batch_index += 1
        batch = self._orders[:, begin:end]
        if len(self._generators) == 1:
            return batch[0] if shared else batch.expand(len(self._member_streams), -1)
        if len(self._generators) == len(self._member_streams):
            return batch
        return batch.index_select(0, self._member_stream_indices)

    def state_dict(self) -> dict[str, Any]:
        # Preserve the old per-member checkpoint layout in both directions.
        states = [generator.get_state() for generator in self._generators]
        return {
            'generator_states': [states[index] for index in self._member_streams],
            'epoch_start_generator_states': (
                None
                if self._epoch_start_generator_states is None
                else [
                    self._epoch_start_generator_states[index]
                    for index in self._member_streams
                ]
            ),
            'batch_index': self._batch_index,
            'has_orders': self._orders is not None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        def unique_states(states: list[Tensor]) -> list[Tensor]:
            if len(states) != len(self._member_streams):
                raise ValueError('The checkpoint has a different pack size')
            for member, stream in enumerate(self._member_streams):
                if not torch.equal(
                    states[member], states[self._stream_first_members[stream]]
                ):
                    raise ValueError(
                        'Checkpoint RNG states differ for members with the same seed'
                    )
            return [states[index] for index in self._stream_first_members]

        generator_states = unique_states(state_dict['generator_states'])
        for generator, state in zip(self._generators, generator_states, strict=True):
            generator.set_state(state)

        self._batch_index = state_dict['batch_index']
        epoch_states = state_dict['epoch_start_generator_states']
        self._epoch_start_generator_states = (
            None if epoch_states is None else unique_states(epoch_states)
        )
        if state_dict['has_orders']:
            if self._epoch_start_generator_states is None:
                raise ValueError('Missing epoch-start generator states')
            replay_generators = []
            for state in self._epoch_start_generator_states:
                generator = torch.Generator(self.device)
                generator.set_state(state)
                replay_generators.append(generator)
            self._orders = self._make_orders(replay_generators)
        else:
            self._orders = None
