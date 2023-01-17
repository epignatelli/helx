import logging

import gym
import jax
from gym.envs.registration import registry

import helx.environment
import helx.logging


logging = helx.logging.get_logger()


EXCLUDES = [
    "ALE/TicTacToe3D-v5",
    "ALE/TicTacToe3D-ram-v5",
]


def test_atari():
    def test_env(env_id):
        logging.info("Testing env: {}".format(env_id))
        env = gym.make(env_id)
        env = helx.environment.to_helx(env)
        env.reset()
        key = jax.random.PRNGKey(i)
        env.step(env.action_space().sample(key))
        env.close()

    # get all atari envs
    atari_env_ids = list(filter(lambda x: "ale/" in x.lower(), registry.keys()))

    for i, env_id in enumerate(atari_env_ids):
        if env_id in EXCLUDES:
            logging.debug("Skipping excluded env: {}".format(env_id))
            continue
        test_env(env_id)


if __name__ == "__main__":
    test_atari()
