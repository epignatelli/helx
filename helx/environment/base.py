"""A set of functions to interoperate between the most common
RL environment interfaces, like `gym`, `gymnasium`, `dm_env`, `bsuite and others."""
from __future__ import annotations

import abc

import jax
from chex import Array

from .mdp import Timestep
from .spaces import Space


class IEnvironment(abc.ABC):
    def __init__(self):
        self._action_space = None
        self._observation_space = None
        self._reward_space = None
        self._current_observation = None
        self._seed = 0
        self._key = jax.random.PRNGKey(self._seed)

    @abc.abstractmethod
    def action_space(self) -> Space:
        ...

    @abc.abstractmethod
    def observation_space(self) -> Space:
        ...

    @abc.abstractmethod
    def reward_space(self) -> Space:
        ...

    @abc.abstractmethod
    def state(self) -> Array:
        ...

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> Timestep:
        ...

    @abc.abstractmethod
    def step(self, action: int) -> Timestep:
        ...

    @abc.abstractmethod
    def seed(self, seed: int) -> None:
        ...

    @abc.abstractmethod
    def render(self, mode: str = "human"):
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def __del__(self):
        self.close()
