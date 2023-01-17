import logging

import gym
import jax

import helx.environment
from helx.environment.gym import list_envs
import helx.logging

logging = helx.logging.get_logger()


def test_gym_minigrid():
    def test_env(env_id):
        logging.info("Testing env: {}".format(env_id))
        env = gym.make(env_id)
        env = helx.environment.to_helx(env)
        env.reset()
        key = jax.random.PRNGKey(i)
        env.step(env.action_space().sample(key))
        env.close()

    # get all minigrid envs
    minigrid_env_ids = list_envs("minigrid")

    for i, env_id in enumerate(minigrid_env_ids):
        test_env(env_id)


if __name__ == "__main__":
    test_gym_minigrid()
