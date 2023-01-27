# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from enum import IntEnum
from functools import partial
from typing import List, Sequence, Tuple, TypeVar

import dm_env
import jax
import jax.numpy as jnp
from chex import Array
from gym.utils.step_api_compatibility import TerminatedTruncatedStepType as GymTimestep
from gymnasium.utils.step_api_compatibility import (
    TerminatedTruncatedStepType as GymnasiumTimestep,
)
from jax.tree_util import register_pytree_node_class

T = TypeVar("T")


def tree_stack(pytree: Sequence[T], axis: int = 0) -> T:
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=axis), *pytree)


class StepType(IntEnum):
    TRANSITION = 0
    TRUNCATION = 1
    TERMINATION = 2


Action = Array
Observation = Array
Reward = Array
Condition = Array
Transition = Tuple[Observation, Action, Reward, Observation, Condition]


class Timestep:
    def __init__(self, observation: Array, reward: Array | None, step_type: StepType):
        self.observation: Array = observation
        self.reward: Array | None = reward
        self.step_type: StepType = step_type

    def is_terminated(self) -> bool:
        return self.step_type == StepType.TERMINATION

    def is_truncated(self) -> bool:
        return self.step_type == StepType.TRUNCATION

    def is_final(self) -> bool:
        return self.is_terminated() or self.is_truncated()

    @classmethod
    def from_gymnasium(cls, gymnasium_step: GymnasiumTimestep) -> Timestep:
        obs, reward, terminated, truncated, _ = gymnasium_step
        obs = jnp.asarray(obs)
        reward = jnp.asarray(reward)
        if terminated:
            step_type = StepType.TERMINATION
        elif truncated:
            step_type = StepType.TRUNCATION
        else:
            step_type = StepType.TRANSITION
        return cls(obs, reward, step_type)

    @classmethod
    def from_gym(cls, gym_step: GymTimestep) -> Timestep:
        obs, reward, terminated, truncated, _ = gym_step
        obs = jnp.asarray(obs)
        reward = jnp.asarray(reward)
        if terminated:
            step_type = StepType.TERMINATION
        elif truncated:
            step_type = StepType.TRUNCATION
        else:
            step_type = StepType.TRANSITION
        return cls(obs, reward, step_type)

    @classmethod
    def from_dm_env(cls, dm_step: dm_env.TimeStep) -> Timestep:
        step_type = dm_step.step_type
        obs = jnp.asarray(dm_step.observation)
        reward = jnp.asarray(dm_step.reward)
        discount = dm_step.discount
        if dm_step.step_type == dm_env.StepType.LAST:
            step_type = StepType.TERMINATION
        elif discount is not None and float(discount) == 0.0:
            step_type = StepType.TRUNCATION
        else:
            step_type = StepType.TRANSITION
        return cls(obs, reward, step_type)


@register_pytree_node_class
class Episode:
    """A collection of (s)
    The class uses the following data structure:
        >>> $s_0, s_1, s_2, s_3, ..., s_t$
        >>> $a_0, a_1, a_2, a_3, ..., a_t$
        >>> $r_1, r_2, r_3, ..., r_t$
        >>> $d_1, d_2, d_3, ..., d_t$
    ```
    And corresponds to the following sarsa unroll:
    $\\langle s_0, a_0, r_1, s_1, d_1, a_1 \\rangle$

    Where, $a_0$ corresponds to the action taken with respect to
    observation $s_0$, and $r_1$ is the reward received during the transition
    from $s_0$ to $s_1$.
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

    @property
    def s_t(self) -> Array:
        return self.s[:-1]

    @property
    def s_tp1(self) -> Array:
        return self.s[1:]

    @property
    def a_t(self) -> Array:
        return self.a

    @property
    def r_tp1(self) -> Array:
        return self.r

    @property
    def d_tp1(self) -> Array:
        return self.d

    def __getitem__(self, idx):
        return (
            self.s_t[idx],
            self.a_t[idx],
            self.r_tp1[idx],
            self.s_tp1[idx],
            self.d_tp1[idx],
        )

    def __len__(self):
        # the number of transitions is 1 minus the number of stored states
        # since states contain also the state for the next timestep
        return len(self._s) - 1

    def tree_flatten(self):
        return (self._s, self._a, self._r, self._d), ()

    @classmethod
    def tree_unflatten(
        cls,
        aux: Tuple[int],
        children: Tuple[List[Array], List[Array], List[Array], List[Array]],
    ) -> Episode:
        """Decodes the PyTree into the `Episode` python object"""
        return cls(*children)

    @classmethod
    def start(cls, timestep: Timestep):
        return cls([timestep.observation], [], [], [])

    def add(
        self,
        timestep: Timestep,
        action: Array,
    ) -> None:
        """Adds a new timestep to the episode.
        Args:
            timestep (dm_env.TimeStep): The timestep to add
            action (Array): the action taken at `timestep.observation`
        Returns:
            None
        """
        self._s.append(jnp.asarray(timestep.observation, dtype=jnp.float32))
        self._a.append(action)
        self._r.append(jnp.asarray([timestep.reward], dtype=jnp.float32))
        self._d.append(jnp.asarray([timestep.step_type], dtype=jnp.int32))
        return

    def transitions(self) -> List[Transition]:
        """Computes a (s₀, a₀, r₁, s₁, d₁) unroll of the episode.
        Returns:
            (Tuple[Array, Array, Array, Array, Array]) a 5D-tuple containing
            each transition in the episode, where the first axis it the
            temporal axis
        """
        assert len(self.s) - 1 == len(self.r) == len(self.d) == len(self.a)
        return list(zip(self.s_t, self.a_t, self.r_tp1, self.s_tp1, self.d_tp1))

    def is_complete(self) -> Array:
        """Returns whether the episode is completed, i.e., whether the last
        timestep was a termination."""
        return self.d[-1] == StepType.TERMINATION

    def returns(self, axis: int | None = None) -> Array:
        return jnp.sum(self.r, axis=axis)
