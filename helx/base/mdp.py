# Copyright 2023 The Helx Authors.
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

from typing import Any, Dict

from jax import Array
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct


class StepType(struct.PyTreeNode):
    """The type of a timestep in an MDP"""

    TRANSITION = jnp.asarray(0)
    TRUNCATION = jnp.asarray(1)
    TERMINATION = jnp.asarray(2)


class Timestep(struct.PyTreeNode):
    """A timestep in an MDP.
    We follow the following convention:
    - $s_t$ is the state at time $t$;
    - $o_t = O(s_t)$ is an observation from the state $s_t$;
    - $a_t \\sim \\pi(s_t)$ is the action sampled from the policy conditioned on $s_t$
    - $r_t = R(s_t, a_t)$ is the reward after taking the action $a_t$ in $s_t$
    - $\\gamma_t$ is the termination/truncation probability after taking $a_t$
    When Timestep represents an episode, because env.reset() starts with a starting
    state, and info, but not with the rest, we have to add placeholders for those.
    For example:
        t =                 0,   1,   2,   3
        observation =     s_0, s_1, s_2, s_3
        action =            /, a_0, a_1, a_2,
        info =            i_0, i_1, i_2, i_3,
        reward =            /, r_0, r_1, r_2,
    where `/` is a placeholder that depends on the property(e.g., 0 for rewards,
    -1 for actions)
    """

    t: Array
    """The number of timesteps elapsed from the last reset of the environment."""
    observation: Array
    """The state $s_{t}$ or the observation $o_{t} = \\mathcal{O}(s_{t})$."""
    action: Array
    """The action $a_{t-1} \\sim \\pi(s_{t-1})$ sampled from the policy conditioned on
        $s_{t-1}$."""
    reward: Array
    """The reward $r_{t-1} = R(s_{t-1}, a_{t-1})$ is the reward after taking the action
        $a_{t-1}$ in $s_t$."""
    step_type: Array
    """The termination/truncation probability $d_{t-1}$ after taking $a_{t-1}$ is
    %s_{t-1}$. See `StepType` for possible values."""
    state: Any
    """The true state of the MDP (the environment class), $s_t$."""
    info: Dict[str, Any] = struct.field(default_factory=dict)
    """Additional information $i_t$ about the environment at $t$ before taking
    action $a_t$. Useful for accumulation, for example, returns or values."""

    def __getitem__(self, key: Any) -> Timestep:
        return jtu.tree_map(lambda x: x[key], self)

    def __setitem__(self, key: Any, value: Any) -> Timestep:
        return jtu.tree_map(lambda x: x.at[key].set(value), self)

    def __len__(self) -> int:
        return self.t.shape[-1]

    @property
    def shape(self) -> Tuple:
        return self.t.shape

    @property
    def at(self) -> Timestep:
        """Indexes the timestep from min(timestep.t) to max(timestep.t) - 1
        with a time-based rule, assuming the following structure of the timestep:
        t:               (  0,   1,   2,   3)
        observation:     (s_0, s_1, s_2, s_3)
        action:          ( -1, a_0, a_1, a_2)
        reward:          (  0, r_0, r_1, r_2)
        step_type:       (  0, d_0, d_1, d_2)
        state:           (x_0, x_1, x_2, x_3)
        info:            (i_0, i_1, i_2, i_3)

        The structure often comes from the usual environment implementation, where
        obs, info = env.reset() lacks actions, rewards and step_type
        """
        return _IndexerHelper(self)

    def length(self, axis: int = 0) -> int:
        """Returns the horizon of the sequence, given a specified time axis, excluding
        the last state, which has no corresponding actions, rewards and step_types"""
        # it is t - 1 because env.reset() starts with a starting state, and info,
        # but none of the other properties (we have one T + 1 states and infos)
        return self.t.shape[axis] - 1

    def is_first(self) -> Array:
        return self.t == jnp.asarray(0)

    def is_mid(self) -> Array:
        return jnp.logical_and(
            self.step_type != StepType.TERMINATION,
            self.step_type != StepType.TRUNCATION,
        )

    def is_last(self) -> Array:
        return jnp.logical_or(
            self.step_type == StepType.TERMINATION,
            self.step_type == StepType.TRUNCATION,
        )


class _IndexerHelper:
    def __init__(self, timestep):
        self.timestep = timestep

    def __getitem__(self, t):
        return _IndexerRef(self.timestep, t)


class _IndexerRef:
    def __init__(self, timestep, idx):
        self.timestep = timestep
        self.idx = idx

    def get(self):
        """Because timestep often represents an episode, this takes the episode
        at time `t`. The episode goes from `[min(timestep.t), max(timestep.t) - 1]`"""

        def _get(path, value):
            # everything that is not obs, info or time, needs shift
            needs_shift = not (
                isinstance(path[0], GetAttrKey)
                and (
                    path[0].name == "observation"
                    or path[0].name == "info"
                    or path[0].name == "t"
                    or path[0].name == "state"
                )
            )
            if needs_shift:
                # shift by one because of the time scheme decribed in the docstrings
                # of the `Timestep``
                # so, to align the arrays, all that is not obs and info is shifted.
                #   From:
                # t =                 0,   1,   2,   3
                # observation =     s_0, s_1, s_2, s_3
                # action =           -1, a_0, a_1, a_2,
                # info =            i_0, i_1, i_2, i_3,
                # reward =            0, r_0, r_1, r_2,
                #   To:
                # t =                 0,   1,   2,   3
                # observation =     s_0, s_1, s_2, s_3
                # action =          a_0, a_1, a_2,
                # info =            i_0, i_1, i_2, i_3,
                # reward =          r_0, r_1, r_2,
                value = value[1:]
            else:
                # otherwise we remove the last, e.g., 3, s_3, i_3 ... and get:
                # t =                 0,   1,   2,
                # observation =     s_0, s_1, s_2,
                # action =          a_0, a_1, a_2,
                # info =            i_0, i_1, i_2,
                # reward =          r_0, r_1, r_2,
                value = value[:-1]
            return value[self.idx]

        return jtu.tree_map_with_path(_get, self.timestep)
