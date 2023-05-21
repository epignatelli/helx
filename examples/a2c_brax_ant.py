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


import brax.envs
import optax
from absl import app, flags

import helx
from helx.logging import NullLogger
from helx.networks import MLP

helx.flags.define_flags_from_hparams(helx.agents.A2CHparams)
FLAGS = flags.FLAGS


def main(argv):
    del argv

    # environment
    env = brax.envs.get_environment("ant", backend="spring")
    env = brax.envs.wrapper.EpisodeWrapper(env, 5, 1)
    env = helx.environment.to_helx(env)

    # optimiser
    optimiser = optax.adam(learning_rate=FLAGS.learning_rate)

    # agent
    hparams = helx.flags.hparams_from_flags(
        helx.agents.A2CHparams,
        obs_space=env.observation_space(),
        action_space=env.action_space(),
        batch_size=2,
    )
    agent = helx.agents.A2C(
        hparams=hparams,
        optimiser=optimiser,
        seed=0,
        actor_network=MLP(features=[128, 128]),
        critic_network=MLP(features=[128, 128]),
    )

    # run
    logger = NullLogger(experiment_name=f"{env.name()}/{agent.name()}/examples")
    helx.experiment.run(agent, env, 2, logger=logger)


if __name__ == "__main__":
    app.run(main)
