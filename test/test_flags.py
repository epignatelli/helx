import logging
from helx import agents, flags
import absl.flags
from absl.testing import flagsaver


def test_flag_type():
    hparams_list = [agents.DQNHparams, agents.SACHparams]
    for hparams in hparams_list:
        saved_flag_values = flagsaver.save_flag_values()
        flags.define_flags_from_hparams(hparams)
        hparams = flags.hparams_from_flags(hparams, absl.flags.FLAGS)
        logging.debug(hparams)
        flagsaver.restore_flag_values(saved_flag_values)


if __name__ == "__main__":
    test_flag_type()
