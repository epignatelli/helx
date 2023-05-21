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


from typing import Any

import brax.envs
import bsuite.environments
import dm_env
import gym.core
import gym3.interop
import gymnasium.core

from .bsuite import BsuiteAdapter
from .dm_env import DmEnvAdapter
from .gym import GymAdapter
from .gymnasium import GymnasiumAdapter
from .gym3 import Gym3Adapter
from .brax import BraxAdapter
from .base import Environment


ADAPTERS_TABLE = {
    gymnasium.core.Env: GymnasiumAdapter,
    gym.core.Env: GymAdapter,
    gym3.interop.ToGymEnv: Gym3Adapter,
    dm_env.Environment: DmEnvAdapter,
    bsuite.environments.Environment: BsuiteAdapter,
    brax.envs.Env: BraxAdapter,
}


def to_helx(env: Any) -> Environment[Any]:
    # getting root env type for interop
    env_for_type = env
    while hasattr(env_for_type, "unwrapped") and env_for_type.unwrapped != env_for_type:
        env_for_type = env_for_type.unwrapped

    for env_type, converter in ADAPTERS_TABLE.items():
        if isinstance(env_for_type, env_type):
            return converter(env)

    raise TypeError(
        f"Environment type {type(env)} is not supported. "
        "Supported types are: {ADAPTERS_TABLE}"
    )
