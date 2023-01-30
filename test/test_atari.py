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
