import logging

import jax.numpy as jnp
from absl.testing import flagsaver

import helx
from helx import agents, flags
from helx.spaces import Continuous, Discrete

logging = helx.logging.get_logger()


def test_flag_type():
    hparams_list = [agents.DQNHparams, agents.SACHparams]
    for hparams in hparams_list:
        saved_flag_values = flagsaver.save_flag_values()
        flags.define_flags_from_hparams(hparams)
        obs_space = Continuous((84, 84, 3), dtype=jnp.float32)
        action_space = Discrete(4)
        hparams = flags.hparams_from_flags(
            hparams, obs_space=obs_space, action_space=action_space
        )
        logging.debug(hparams)
        flagsaver.restore_flag_values(saved_flag_values)


if __name__ == "__main__":
    test_flag_type()
