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


import bsuite
import flax.linen as nn
import optax
from absl import app, flags, logging

import helx

helx.flags.define_flags_from_hparams(helx.agents.DQNHparams)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.info("Starting")

    # environment
    env = bsuite.load_from_id("catch/0")
    env = helx.environment.to_helx(env)

    # optimiser
    optimiser = optax.rmsprop(
        learning_rate=FLAGS.learning_rate,
        momentum=FLAGS.gradient_momentum,
        eps=FLAGS.min_squared_gradient,
        centered=True,
    )

    # agent
    hparams = helx.flags.hparams_from_flags(
        helx.agents.DQNHparams,
        obs_space=env.observation_space(),
        action_space=env.action_space(),
        replay_start=10,
        batch_size=2,
    )

    representation_net = nn.Sequential(
        [
            helx.networks.Flatten(),
            helx.networks.MLP(features=[32, 16]),
        ]
    )
    agent = helx.agents.DQN(
        optimiser=optimiser,
        hparams=hparams,
        seed=0,
        representation_net=representation_net,
    )

    helx.experiment.run(agent, env, 2)


if __name__ == "__main__":
    app.run(main)
