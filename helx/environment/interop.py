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


from typing import Any, Tuple, overload

import bsuite.environments
import dm_env
import gym.core
import gym3.interop
import gymnasium.core
from gymnax.environments.environment import Environment as GymnaxEnvironment, EnvParams
import brax.envs
import navix as nx

from .environment import EnvironmentWrapper
from .bsuite import BsuiteWrapper
from .dm_env import DmEnvWrapper
from .gym import GymWrapper
from .gymnasium import GymnasiumWrapper
from .gymnax import GymnaxWrapper
from .brax import BraxWrapper
from .navix import NavixWrapper


@overload
def to_helx(env: gymnasium.core.Env) -> GymnasiumWrapper:
    ...


@overload
def to_helx(env: gym.core.Env) -> GymWrapper:
    ...


@overload
def to_helx(env: gym3.interop.ToGymEnv) -> GymWrapper:
    ...


@overload
def to_helx(env: dm_env.Environment) -> DmEnvWrapper:
    ...


@overload
def to_helx(env: bsuite.environments.Environment) -> BsuiteWrapper:
    ...


@overload
def to_helx(env: Tuple[GymnaxEnvironment, EnvParams]) -> GymnaxWrapper:
    ...


@overload
def to_helx(env: brax.envs.Env) -> BraxWrapper:
    ...


@overload
def to_helx(env: nx.environments.Environment) -> NavixWrapper:
    ...


def to_helx(env: Any) -> EnvironmentWrapper:
    # getting root env type for interop
    env_for_type = env
    while hasattr(env_for_type, "unwrapped") and env_for_type.unwrapped != env_for_type:
        env_for_type = env_for_type.unwrapped

    # converting the actual env, rather than the root env
    # which would remove time limits and o
    if isinstance(env_for_type, gymnasium.core.Env):
        return GymnasiumWrapper.wraps(env)
    elif isinstance(env_for_type, gym.core.Env):
        return GymWrapper.wraps(env)
    elif isinstance(env_for_type, gym3.interop.ToGymEnv):
        return GymWrapper.wraps(env)
    elif isinstance(env_for_type, dm_env.Environment):
        return DmEnvWrapper.wraps(env)
    elif isinstance(env_for_type, bsuite.environments.Environment):
        return BsuiteWrapper.wraps(env)
    elif (
        isinstance(env_for_type, tuple)
        and issubclass(type(env[0]), GymnaxEnvironment)
        # and issubclass(type(env[1]), EnvParams)  # gymnax EnvParams do not have a base class
    ):
        return GymnaxWrapper.wraps(env)
    elif isinstance(env_for_type, brax.envs.Env):
        return BraxWrapper.wraps(env)
    elif isinstance(env_for_type, nx.environments.Environment):
        return NavixWrapper.wraps(env)
    else:
        raise TypeError(
            f"Environment type {type(env)} is not supported. "
            "Only gymnasium, gym, dm_env and bsuite environments are supported."
        )
