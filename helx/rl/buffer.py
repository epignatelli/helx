import logging
from collections import deque
import dm_env

import jax
import jax.numpy as jnp

from ..types import Transition, Key


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        # public:
        self.capacity = capacity
        self.seed = seed

        # private:
        self._rng = jnp.random.PRNGKey(seed)
        self._data = deque(maxlen=capacity)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        self._data.append(
            Transition(
                x_0=timestep.observation,
                a_0=action,
                r_1=timestep.reward,
                x_1=new_timestep.observation,
                gamma=timestep.discount,
            )
        )
        return

    def sample(self, n: int, rng: Key = None) -> Transition:
        high = len(self) - n
        if high <= 0:
            logging.warning(
                "The buffer contains less elements than requested: {} <= {}\n"
                "Returning all the available elements".format(len(self), n)
            )
            indices = range(len(self))
        elif rng is None:
            rng, _ = jax.random.split(self._rng)
        else:
            indices = jnp.random.randint(rng, 0, high, size=n)

        return Transition(zip(*(map(lambda idx: self._data[idx], indices))))
