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
    For example:
    (0, s_0, -1, 0.0, None, 0.0)
    (1, s_1, a_0, r_0, gamma_0, z_0)
    (2, s_2, a_1, r_1, gamm_1, z_0)
    (3, s_3, a_2, r_2, 1)
    (0, s_0, None, None, None)
    (1, s_1, a_0, r_0, 0)"""
    t: Array
    """The number of timesteps elapsed from the last reset of the environment."""
    observation: Array
    """The state $s_{t+1}$ or the observation $o_{t+1} = \\mathcal{O}(s_{t+1})$."""
    action: Array
    """The action $a_t \\sim \\pi(s_t)$ sampled from the policy conditioned on $s_t$."""
    reward: Array
    """The reward $r_t = R(s_t, a_t)$ is the reward after taking the action $a_t$ in
    $s_t$."""
    step_type: Array
    """The termination/truncation probability after taking $a_t$ is %s_t$.
    See `StepType` for possible values."""
    state: Any
    """The true state of the MDP, $s_t$."""
    info: Dict[str, Any] = struct.field(default_factory=dict)
    """Additional information about the environment at $t$ before taking action $a_t$.
    Useful for accumulation, for example, returns."""

    def __getitem__(self, key: Any) -> Timestep:
        return jtu.tree_map(lambda x: x[key], self)

    def __setitem__(self, key: Any, value: Any) -> Timestep:
        return jtu.tree_map(lambda x: x.at[key].set(value), self)

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
