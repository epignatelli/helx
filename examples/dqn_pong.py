from typing import cast

import flax.linen as nn
import gymnasium
import jax.numpy as jnp
import optax
from absl import app, flags, logging

import helx


helx.flags.define_flags_from_hparams(helx.agents.DQNhparams)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.info("Starting")

    # environment
    env = gymnasium.make("pong-v0")
    env = helx.environment.make_from(env)

    # optimiser
    optimiser = optax.rmsprop(
        learning_rate=FLAGS.learning_rate,
        momentum=FLAGS.gradient_momentum,
        eps=FLAGS.min_squared_gradient,
        centered=True,
    )

    # agent
    n_actions = len(cast(helx.spaces.Discrete, env.action_space()))
    hparams = helx.flags.hparams_from_flags(helx.agents.DQNhparams, FLAGS)
    network = nn.Sequential(
        [
            lambda x: jnp.reshape(x, (-1, 1)),
            nn.Dense(features=64),
            nn.Dense(features=n_actions),
        ]
    )
    agent = helx.agents.DQN(
        network=network, optimiser=optimiser, hparams=hparams, seed=0
    )

    helx.experiment.run(agent, env, 100)


if __name__ == "__main__":
    app.run(main)
