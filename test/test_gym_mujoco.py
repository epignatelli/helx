import logging

import gym
import jax
from gym.envs.registration import registry

from helx.environment.gym import list_envs
import helx.environment
import helx.logging


logging = helx.logging.get_logger()



EXCLUDES = [
]


def test_mujoco():
    def test_env(env_id):
        env = gym.make(env_id)
        env = helx.environment.to_helx(env)
        env.reset()
        key = jax.random.PRNGKey(i)
        env.step(env.action_space().sample(key))
        env.close()

    # get all atari envs
    mujoco_env_ids = list_envs("mujoco")

    for i, env_id in enumerate(mujoco_env_ids):
        logging.info("Testing Atari env: {}".format(env_id))
        if env_id in EXCLUDES:
            logging.debug("Skipping excluded env: {}".format(env_id))
            continue
        test_env(env_id)


if __name__ == "__main__":
    test_mujoco()
