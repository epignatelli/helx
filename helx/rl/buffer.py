import logging
from collections import deque
from typing import List, NamedTuple
from dataclasses import dataclass

import dm_env
import jax
import jax.numpy as jnp
import numpy as onp

from ..types import Key


@dataclass
class Transition:
    observations: List[
        onp.ndarray
    ]  #  observations at t=0. Note that observation contains
    #     one more item than the rest of the fields (the last observation)
    actions: List[onp.ndarray]  #  actions at t=0
    rewards: List[onp.ndarray]  #  rewards at t=1
    gamma: List[onp.ndarray] = (1.0,)  #  discount factor
    trace_decay: List[onp.ndarray] = (1.0,)  # trace decay for lamba returns

    @staticmethod
    def empty():
        return Transition([], [], [], [], [])


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        n_steps: int = 1,
        cumulative: bool = True,
        seed: int = 0,
    ):
        #  public:
        self.capacity = capacity
        self.n_steps = n_steps
        self.cumulative = cumulative
        self.seed = seed

        #  private:
        self._rng = jax.random.PRNGKey(seed)
        self._episodes = deque(maxlen=capacity)
        self._current_episode_idx = 0
        self._t = 0

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, idx):
        return self._episodes[idx]

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
        trace_decay: float = 1.0,
    ) -> None:
        #  start of a new episode
        if timestep.first() or self._t == 0:
            self._episodes.append(Transition.empty())

        #  append current features
        idx = self._current_episode_idx % self.capacity
        self._episodes[idx].observations.append(timestep.observation)
        self._episodes[idx].actions.append(action)
        self._episodes[idx].rewards.append(new_timestep.reward)
        self._episodes[idx].gamma.append(new_timestep.discount)
        self._episodes[idx].trace_decay.append(trace_decay)
        self._t += 1

        #  if last(), pack the sequences into arrays
        if (new_timestep.last()) or (self._t >= self.n_steps):
            #  append final observation
            self._episodes[idx].observations.append(new_timestep.observation)
            #  prepare the transition for sampling
            self._episodes[idx].observations = jnp.stack(
                self._episodes[idx].observations
            )
            self._episodes[idx].actions = onp.stack(self._episodes[idx].actions)
            self._episodes[idx].rewards = onp.stack(self._episodes[idx].rewards)
            self._episodes[idx].gamma = onp.stack(self._episodes[idx].gamma)
            self._episodes[idx].trace_decay = onp.stack(self._episodes[idx].trace_decay)

            self._current_episode_idx += 1
            self._t = 0

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
        indices = jax.random.randint(rng, (n,), 0, high)

        return [self._episodes[idx] for idx in indices]
