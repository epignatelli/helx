from typing import cast

import flax.linen as nn
import gymnasium
import optax
from absl import app, flags, logging

import helx

helx.flags.define_flags_from_hparams(helx.agents.DQNHparams)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.info("Starting")

    # environment
    env = gymnasium.make("Ant-v4")
    env = helx.environment.make_from(env)

    # optimiser
    optimiser = optax.adam(learning_rate=FLAGS.learning_rate)

    # agent
    action_space = cast(helx.spaces.Continuous, env.action_space())
    hparams = helx.flags.hparams_from_flags(
        helx.agents.DQNHparams,
        FLAGS,
        input_shape=env.observation_space().shape,
        dim_A=action_space.shape[0],
    )

    network = nn.Sequential(
        [
            helx.networks.Flatten(),
            helx.networks.MLP(features=[32, 16]),
            nn.Dense(features=n_actions),
        ]
    )
    agent = helx.agents.DQN(
        network=network, optimiser=optimiser, hparams=hparams, seed=0
    )

    helx.experiment.run(agent, env, 100)


if __name__ == "__main__":
    app.run(main)
