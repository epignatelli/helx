from __future__ import annotations

import abc

import dm_env.specs
import gym.spaces
import gymnasium.spaces
import jax.numpy as jnp
from chex import Array


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
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

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


class BoundedSpace(Space):
    def __init__(self, minimum: float = -float("inf"), maximum: float = float("inf")):
        self.min: Array = jnp.asarray(minimum)
        self.max: Array = jnp.asarray(maximum)

    def from_gym(self, gym_space: gym.spaces.Box) -> BoundedSpace:
        # TODO
        raise NotImplementedError()

    def from_gymnasium(self, gymnasium_space: gymnasium.spaces.Space) -> Space:
        # TODO
        raise NotImplementedError()

    def from_dm_env(self, dm_space: dm_env.specs.BoundedArray) -> BoundedSpace:
        # TODO
        raise NotImplementedError()
