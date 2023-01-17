import logging

import gym
import jax

import helx.environment
import helx.logging
from helx.environment.gym import list_envs

logging = helx.logging.get_logger()


EXCLUDES = []


def test_procgen():
    def test_env(env_id):
        logging.info("Testing env: {}".format(env_id))
        env = gym.make(env_id)
        env = helx.environment.to_helx(env)
        env.reset()
        key = jax.random.PRNGKey(i)
        env.step(env.action_space().sample(key))
        env.close()

    mujoco_env_ids = list_envs("procgen")

    for i, env_id in enumerate(mujoco_env_ids):
        test_env(env_id)


if __name__ == "__main__":
    test_procgen()
