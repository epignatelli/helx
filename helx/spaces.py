from __future__ import annotations

import abc
from typing import Sequence

import dm_env.specs
import gym.spaces
import gymnasium.spaces
import jax.numpy as jnp
from chex import Array, Shape
import jax


class Space(abc.ABC):
    @classmethod
    def from_gym(cls, gym_space: gym.spaces.Space) -> Space:
        if isinstance(gym_space, gym.spaces.Discrete):
            return Discrete.from_gym(gym_space)
        elif isinstance(gym_space, gym.spaces.Box):
            return Continuous.from_gym(gym_space)
        else:
            raise NotImplementedError(
                "Cannot convert gym space of type {}".format(type(gym_space))
            )

    @classmethod
    def from_gymnasium(cls, gymnasium_space: gymnasium.spaces.Space) -> Space:
        if isinstance(gymnasium_space, gymnasium.spaces.Discrete):
            return Discrete.from_gymnasium(gymnasium_space)
        elif isinstance(gymnasium_space, gymnasium.spaces.Box):
            return Continuous.from_gymnasium(gymnasium_space)
        else:
            raise NotImplementedError(
                "Cannot convert gymnasium space of type {}".format(
                    type(gymnasium_space)
                )
            )

    @classmethod
    def from_dm_env(cls, dm_space: dm_env.specs.Array) -> Space:
        if isinstance(dm_space, dm_env.specs.DiscreteArray):
            return Discrete.from_dm_env(dm_space)
        elif isinstance(dm_space, dm_env.specs.BoundedArray):
            return Continuous.from_dm_env(dm_space)
        else:
            raise NotImplementedError(
                "Cannot convert dm_env space of type {}".format(type(dm_space))
            )


class Discrete(Space):
    def __init__(self, n_dimensions: int):
        self.n_dimensions: int = n_dimensions

    def __len__(self) -> int:
        return self.n_dimensions

    def sample(self, key) -> Array:
        return jax.random.randint(key, (1,), 0, self.n_dimensions)

    @classmethod
    def from_gym(cls, gym_space: gym.spaces.Discrete) -> Discrete:
        return cls(gym_space.n)

    @classmethod
    def from_gymnasium(cls, gymnasium_space: gymnasium.spaces.Discrete) -> Discrete:
        return cls(int(gymnasium_space.n))

    @classmethod
    def from_dm_env(cls, dm_space: dm_env.specs.DiscreteArray) -> Discrete:
        return cls(dm_space.num_values)


class Continuous(Space):
    def __init__(
        self,
        shape: Shape = (1,),
        minimum: float | Sequence[float] | Array = -1.0,
        maximum: float | Sequence[float] | Array = 1.0,
    ):
        self.shape: Shape = shape
        self.min: Array = jnp.broadcast_to(jnp.asarray(minimum), shape=shape)
        self.max: Array = jnp.broadcast_to(jnp.asarray(maximum), shape=shape)

        assert (
            self.min.shape == self.max.shape == shape
        ), "minimum and maximum must have the same length as n_dimensions, got {} and {} for n_dimensions={}".format(
            self.min.shape, self.max.shape, shape
        )

    def sample(self, key) -> Array:
        return jax.random.uniform(key, self.shape, minval=self.min, maxval=self.max)

    @classmethod
    def from_gym(cls, gym_space: gym.spaces.Box) -> Continuous:
        shape = gym_space.shape
        minimum = jnp.asarray(gym_space.low)
        maximum = jnp.asarray(gym_space.high)
        return cls(shape, minimum, maximum)

    @classmethod
    def from_gymnasium(cls, gymnasium_space: gymnasium.spaces.Box) -> Continuous:
        shape = gymnasium_space.shape
        minimum = jnp.asarray(gymnasium_space.low)
        maximum = jnp.asarray(gymnasium_space.high)
        return cls(shape, minimum, maximum)

    @classmethod
    def from_dm_env(cls, dm_space: dm_env.specs.BoundedArray) -> Continuous:
        shape = dm_space.shape
        minimum = jnp.asarray(dm_space.minimum)
        maximum = jnp.asarray(dm_space.maximum)
        return cls(shape, minimum, maximum)
