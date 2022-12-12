import gym
from bsuite.utils.gym_wrapper import DMEnvFromGym


def make(env_name, **kwargs):
    """Create a DMEnv environment using the Gym environment protocol.

    Args:
        env_name: Name of the environment to create.
        **kwargs: Additional arguments to pass to the environment.

    Returns:
        A DMEnv environment.
    """
    gym_env = gym.make(env_name, **kwargs)
    env = DMEnvFromGym(gym_env)
    return env
