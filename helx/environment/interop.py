from typing import Any

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


def make_from(env: Any) -> Any:
    # getting root env type for interop
    env_for_type = env
    while hasattr(env_for_type, "unwrapped") and env_for_type.unwrapped != env_for_type:
        env_for_type = env_for_type.unwrapped

    # converting the actual env, rather than the root env
    # which would remove time limits and o
    if isinstance(env_for_type, gymnasium.core.Env):
        return FromGymnasiumEnv(env)
    elif isinstance(env_for_type, gym.core.Env):
        return FromGymEnv(env)
    elif isinstance(env_for_type, gym3.interop.ToGymEnv):
        return FromGym3Env(env)
    elif isinstance(env_for_type, dm_env.Environment):
        return FromDmEnv(env)
    elif isinstance(env_for_type, bsuite.environments.Environment):
        return FromBsuiteEnv(env)
    else:
        raise TypeError(
            f"Environment type {type(env)} is not supported. "
            "Only gymnasium, gym, dm_env and bsuite environments are supported."
        )
