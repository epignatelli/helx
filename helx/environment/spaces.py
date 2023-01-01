from __future__ import annotations

import abc

import dm_env.specs
import gym.spaces
import gymnasium.spaces
import jax.numpy as jnp
from chex import Array
import jax


class Space(abc.ABC):
    @classmethod
    def from_gym(cls, gym_space: gym.spaces.Space) -> Space:
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_gymnasium(cls, gymnasium_space: gymnasium.spaces.Space) -> Space:
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_dm_env(cls, dm_space: dm_env.specs.Array) -> Space:
        # TODO
        raise NotImplementedError()


class DiscreteSpace(Space):
    def __init__(self, n_dimensions: int):
        self.n_dimensions = n_dimensions

    def __len__(self) -> int:
        return self.n_dimensions

    def sample(self, key) -> Array:
        return jax.random.randint(key, (1,), 0, self.n_dimensions)

    @classmethod
    def from_gym(cls, gym_space: gym.spaces.Discrete) -> DiscreteSpace:
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_gymnasium(
        cls, gymnasium_space: gymnasium.spaces.Discrete
    ) -> DiscreteSpace:
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_dm_env(cls, dm_space: dm_env.specs.BoundedArray) -> DiscreteSpace:
        # TODO
        raise NotImplementedError()


class ContinuousSpace(Space):
    def __init__(self, n_dimensions: int = 1):
        self.n_dimensions = n_dimensions

    def sample(self, key) -> Array:
        return jax.random.uniform(key, (self.n_dimensions,), -float("inf"), float("inf"))

    @classmethod
    def from_gym(cls, gym_space: gym.spaces.Box) -> ContinuousSpace:
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_gymnasium(cls, gymnasium_space: gymnasium.spaces.Space) -> Space:
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_dm_env(cls, dm_space: dm_env.specs.BoundedArray) -> ContinuousSpace:
        # TODO
        raise NotImplementedError()


class BoundedRange(Space):
    def __init__(self, minimum: float = -float("inf"), maximum: float = float("inf")):
        self.min: Array = jnp.asarray(minimum)
        self.max: Array = jnp.asarray(maximum)

    def sample(self, key) -> Array:
        return jax.random.uniform(key, (1,), self.min, self.max)

    def from_gym(self, gym_space: gym.spaces.Box) -> BoundedRange:
        # TODO
        raise NotImplementedError()

    def from_gymnasium(self, gymnasium_space: gymnasium.spaces.Space) -> Space:
        # TODO
        raise NotImplementedError()

    def from_dm_env(self, dm_space: dm_env.specs.BoundedArray) -> BoundedRange:
        # TODO
        raise NotImplementedError()
