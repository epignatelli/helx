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
from flax import struct


TRANSITION = jnp.asarray(0)
TRUNCATION = jnp.asarray(1)
TERMINATION = jnp.asarray(2)


class Timestep(struct.PyTreeNode):
    t: Array
    """The number of timesteps elapsed from the last reset of the environment"""
    observation: Array
    """The observation corresponding to the current state (for POMDPs)"""
    action: Array
    """The action taken by the agent at the current timestep a_t = $\\pi(s_t)$, where $s_t$ is `state`"""
    reward: Array
    """The reward $r_{t=1}$ received by the agent after taking action $a_t$"""
    step_type: Array
    """The type of the current timestep (see `helx.mdp`)"""
    state: Any
    """The true state of the MDP, $s_t$ before taking action `action`"""
    info: Dict[str, Any] = struct.field(default_factory=dict)
    """Additional information about the environment. Useful for accumulations (e.g. returns)"""
