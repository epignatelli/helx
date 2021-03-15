import logging
from collections import deque
import jax.numpy as jnp

from .types import Transition


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        # public:
        self.capacity = capacity
        self.rng = jnp.random.PRNGKey(seed)

        # private:
        self._data = deque(maxlen=capacity)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def add(self, timestep, action, new_timestep):
        self._data.append(
            Transition(
                timestep.observation, action, timestep.reward, new_timestep.observation
            )
        )
        return

    def sample(self, n):
        high = len(self) - n
        if high <= 0:
            logging.warning(
                "The buffer contains less elements than requested: {} <= {}\n"
                "Returning all the available elements".format(len(self), n)
            )
            indices = range(len(self))
        else:
            indices = jnp.random.randint(self.rng, 0, high, size=n)

        return tuple(zip(*(map(lambda idx: self._data[idx], indices))))
