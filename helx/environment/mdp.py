from __future__ import annotations
from enum import IntEnum

from functools import partial
from typing import Any, List, Tuple, SupportsFloat

import dm_env
import gym.core
import gymnasium.core
import jax
import jax.numpy as jnp
from chex import Array, dataclass
from dm_env import TimeStep
from jax.tree_util import register_pytree_node_class


def tree_stack(pytree, axis=0):
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=axis), *pytree)


GymnasiumTimestep = Tuple[gymnasium.core.ObsType, SupportsFloat, bool, bool, dict[str, Any]]
GymTimestep = Tuple[gym.core.ObsType, float, bool, bool, dict]


class Action(int, float, Array):
    ...


class StepType(IntEnum):
    TRANSITION = 0
    TRUNCATION = 1
    TERMINATION = 2


@register_pytree_node_class
class Timestep:
    def __init__(self, observation: Array, reward: Array | None, step_type: StepType):
        self.observation: Array = observation
        self.reward: Array | None = reward
        self.step_type: StepType = step_type

    @classmethod
    def from_gym(cls, gym_step: GymnasiumTimestep) -> Timestep:
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_gymnasium(cls, gymnasium_step: GymnasiumTimestep) -> Timestep:
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_dm_env(cls, dm_step: dm_env.TimeStep) -> Timestep:
        # TODO
        raise NotImplementedError()


@register_pytree_node_class
class Episode:
    """A collection of (s)
    The class uses the following data structure:
        >>> s₀, s₁, s₂, s₃, ..., sₜ,
        >>> a₀, a₁, a₂, a₃, ..., aₜ,
        >>> /  , r₁, r₂, r₃, ..., rₜ,
        >>> /  , d₁, d₂, d₃, ..., dₜ,
    ```
    And corresponds to the following sarsa unroll:
    (s₀, a₀, r₁, s₁, d₁, a₁)

    Where, a_0 corresponds to the action taken with respect to
    observation s_0, and r_1 is the reward received during the transition
    from s_0 to s_1.
    """

    def __init__(self, s: List[Array], a: List[Array], r: List[Array], d: List[Array]):
        self._s: List[Array] = s
        self._a: List[Array] = a
        self._r: List[Array] = r
        self._d: List[Array] = d

    @property
    def s(self) -> Array:
        return jnp.stack(self._s)

    @property
    def a(self) -> Array:
        return jnp.stack(self._a)

    @property
    def r(self) -> Array:
        return jnp.stack(self._r)

    @property
    def d(self) -> Array:
        return jnp.stack(self._d)

    def __getitem__(self, idx):
        return jax.tree_util.tree_map(lambda x: x[idx], self.sarsa())

    def __len__(self):
        # the number of transitions is 1 minus the number of stored states
        # since states contain also the state for the next timestep
        return len(self.s) - 1

    def tree_flatten(self):
        return (self.s, self.a, self.r, self.d), ()

    @classmethod
    def tree_unflatten(
        cls,
        aux: Tuple[int],
        children: Tuple[List[Array], List[Array], List[Array], List[Array]],
    ) -> Episode:
        """Decodes the PyTree into the `Episode` python object"""
        return cls(*children)

    @classmethod
    def start(cls, timestep: TimeStep):
        return cls([timestep.observation], [], [], [])

    def add(
        self,
        timestep: dm_env.TimeStep,
        action: Array,
    ) -> None:
        """Adds a new timestep to the episode.
        Args:
            timestep (dm_env.TimeStep): The timestep to add
            action (Array): the action taken at `timestep.observation`
        Returns:
            None
        """
        # add new transition to the trajectory
        self._s.append(jnp.asarray(timestep.observation, dtype=jnp.float32))
        self._a.append(action)
        self._r.append(jnp.asarray([timestep.reward], dtype=jnp.float32))
        # `self.d` stores whether the timestep is *terminal*
        # reaching env.max_steps does not cause termination, but truncation
        # γ(truncation) != 0, while γ(termination) = 0
        truncated = bool(timestep.discount)
        self._d.append(jnp.array([timestep.last() * int(truncated)], dtype=jnp.int32))
        return

    def sars(self, axis=0):
        """Computes a (s₀, a₀, r₁, s₁, d₁) unroll of the episode.
        Args:
            axis (int): The temporala axis to index into
        Returns:
            (Tuple[Array, Array, Array, Array, Array]) a 5D-tuple containing
            each transition in the episode, where the first axis it the
            temporal axis
        """
        assert len(self.s) - 1 == len(self.r) == len(self.d) == len(self.a)
        take = partial(jax.lax.slice_in_dim, axis=axis)
        pairs = []
        for t in range(0, len(self.s) - 1):
            transition = (
                take(self.s, t, t + 1),
                take(self.a, t, t + 1),
                take(self.r, t, t + 1),
                take(self.s, t + 1, t + 2),
                take(self.d, t, t + 1),
            )
            pairs.append(transition)
        return pairs

    def sarsa(self, axis=0):
        """Returns a (s₀, a₀, r₁, s₁, d₁, a₁) unroll of the episode
        Args:
            axis (int): The temporala axis to index into
        Returns:
            (Tuple[Array, Array, Array, Array, Array, Array]) a 6D-tuple containing
            each transition in the episode, where the first axis it the
            temporal axis
        """
        assert len(self.s) == len(self.r) + 1 == len(self.d) + 1 == len(self.a)
        take = partial(jax.lax.slice_in_dim, axis=axis)
        pairs = []
        for t in range(0, len(self.s) - 1):
            transition = (
                take(self.s, t, t + 1),
                take(self.a, t, t + 1),
                take(self.r, t, t + 1),
                take(self.s, t + 1, t + 2),
                take(self.d, t, t + 1),
                take(self.a, t + 1, t + 2),
            )
            pairs.append(transition)
        return pairs
