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
