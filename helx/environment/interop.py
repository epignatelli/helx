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

from .bsuite import FromBsuiteEnv
from .dm_env import FromDmEnv
from .gym import FromGymEnv
from .gymnasium import FromGymnasiumEnv
from .gym3 import FromGym3Env
from .brax import FromBraxEnv


CONVERSION_TABLE = {
    gymnasium.core.Env: FromGymnasiumEnv,
    gym.core.Env: FromGymEnv,
    gym3.interop.ToGymEnv: FromGym3Env,
    dm_env.Environment: FromDmEnv,
    bsuite.environments.Environment: FromBsuiteEnv,
    brax.envs.Env: FromBraxEnv,
}

def to_helx(env: Any) -> Any:
    # getting root env type for interop
    env_for_type = env
    while hasattr(env_for_type, "unwrapped") and env_for_type.unwrapped != env_for_type:
        env_for_type = env_for_type.unwrapped

    for env_type, converter in CONVERSION_TABLE.items():
        if isinstance(env_for_type, env_type):
            return converter(env)

    raise TypeError(
        f"Environment type {type(env)} is not supported. "
        "Supported types are: {CONVERSION_TABLE}"
    )
