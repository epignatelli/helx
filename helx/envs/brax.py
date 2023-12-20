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

import jax
import jax.numpy as jnp
import brax.envs
from jax.random import KeyArray

from helx.base.spaces import MAX_INT_ARR, Continuous
from helx.base.mdp import Timestep, TRANSITION, TERMINATION
from .environment import EnvironmentWrapper


class BraxWrapper(EnvironmentWrapper):
    """Static class to convert between Brax environments and helx environments.
    The current environments are supported:
    ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup',
    'inverted_pendulum', 'inverted_double_pendulum', 'pusher',
    'reacher', 'walker2d']"""
    env: brax.envs.Env

    @classmethod
    def wraps(cls, env: brax.envs.Env) -> BraxWrapper:
        return cls(
            env=env,
            observation_space=Continuous(shape=(env.observation_size,)),
            action_space=Continuous(shape=(env.action_size,)),
            reward_space=Continuous(),
        )

    def reset(self, key: KeyArray) -> Timestep:
        state = self.env.reset(key)
        return Timestep(
            t=jnp.asarray(0),
            observation=state.obs,
            reward=state.reward,
            step_type=TRANSITION,
            action=self.action_space.sample(key),
            state=state.pipeline_state,
            info={**state.info, **state.metrics}
        )

    def _step(self, key: KeyArray, timestep: Timestep, action: jax.Array) -> Timestep:
        # unwrap
        state = brax.envs.State(
            pipeline_state=timestep.state,
            obs=timestep.observation,
            reward=timestep.reward,
            done=timestep.step_type == TERMINATION,
            info=timestep.info,
            metrics=timestep.info
        )

        # step
        state = self.env.step(state=state, action=action)

        # wrap again
        truncation = state.info.get("truncation", state.metrics.get("truncation", MAX_INT_ARR))
        step_type = 2 * state.done + timestep.t > truncation
        return Timestep(
            t=timestep.t + 1,
            observation=state.obs,
            reward=state.reward,
            step_type=jnp.asarray(step_type, dtype=jnp.int32),
            action=action,
            state=state.pipeline_state,
            info={**state.info, **state.metrics}
        )
