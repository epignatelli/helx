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
    while hasattr(env, "unwrapped") and env.unwrapped != env:
        env = env.unwrapped
    if isinstance(env, gymnasium.core.Env):
        return FromGymnasiumEnv(env)
    elif isinstance(env, gym.core.Env):
        return FromGymEnv(env)
    elif isinstance(env, gym3.interop.ToGymEnv):
        return FromGym3Env(env)
    elif isinstance(env, dm_env.Environment):
        return FromDmEnv(env)
    elif isinstance(env, bsuite.environments.Environment):
        return FromBsuiteEnv(env)
    else:
        raise TypeError(
            f"Environment type {type(env)} is not supported. "
            "Only gymnasium, gym, dm_env and bsuite environments are supported."
        )
