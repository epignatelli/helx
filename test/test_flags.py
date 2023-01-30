# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
