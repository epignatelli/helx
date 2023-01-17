import logging

import gym
import gym_minigrid
import jax
from gym.envs.registration import registry as gym_registry

import helx.environment
import helx.logging

logging = helx.logging.get_logger()


EXCLUDES = []


"""These environments are currently not supported
because they make use of string arrays, which are not
supported by jax."""


def test_gym_minigrid():
    def test_env(env_id):
        env = gym.make(env_id)
        env = helx.environment.to_helx(env)
        env.reset()
        key = jax.random.PRNGKey(i)
        env.step(env.action_space().sample(key))
        env.close()

    # get all minigrid envs

    minigrid_env_ids = filter(lambda x: x.startswith("MiniGrid-"), gym_registry.keys())

    for i, env_id in enumerate(minigrid_env_ids):
        logging.info("Testing MiniGrid env: {}".format(env_id))
        if env_id in EXCLUDES:
            logging.warning("Skipping env {}".format(env_id))
            continue
        test_env(env_id)


if __name__ == "__main__":
    test_gym_minigrid()
