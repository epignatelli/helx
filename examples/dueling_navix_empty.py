# Copyright 2023 The Helx Authors.
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


import navix as nx
import flax.linen as nn
import optax
from absl import app, flags

import helx
import wandb

helx.config.define_flags_from_hparams(helx.agents.DQNHParams)
FLAGS = flags.FLAGS


def main(argv):
    wandb.init(mode="disabled")

    # environment
    env = nx.environments.Room(5, 5, 100, observation_fn=nx.observations.categorical)
    env = helx.environment.to_helx(env)  # type: ignore

    # optimiser
    optimiser = optax.rmsprop(
        learning_rate=FLAGS.learning_rate,
        momentum=FLAGS.gradient_momentum,
        eps=FLAGS.min_squared_gradient,
        centered=True,
    )

    # agent
    hparams = helx.config.hparams_from_flags(
        helx.agents.DuelingDQNHParams,
        obs_space=env.observation_space,
        action_space=env.action_space,
        replay_start=10,
        batch_size=2,
    )

    backbone = nn.Sequential(
        [
            helx.modules.Flatten(),
            helx.modules.MLP(features=[32, 16]),
        ]
    )
    agent = helx.agents.DuelingDQN.create(
        hparams=hparams,
        optimiser=optimiser,
        backbone=backbone,
    )

    helx.experiment.jrun(seed=FLAGS.seed, agent=agent, env=env, budget=20)


if __name__ == "__main__":
    app.run(main)
