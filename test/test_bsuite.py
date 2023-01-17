import logging

import bsuite
import jax
from bsuite.sweep import SWEEP as BSUITE_IDS
import helx.environment
import helx.logging


logging = helx.logging.get_logger()


def test_bsuite():
    def test_env(env_id):
        env = bsuite.load_from_id(env_id)
        env = helx.environment.to_helx(env)
        env.reset()
        key = jax.random.PRNGKey(i)
        env.step(env.action_space().sample(key))
        env.close()

    # get all bsuite envs
    for i, env_id in enumerate(BSUITE_IDS):
        logging.info("Testing bsuite env: {}".format(env_id))
        test_env(env_id)


if __name__ == "__main__":
    test_bsuite()
