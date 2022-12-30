import flax.linen as nn
from absl import logging, app, flags
import optax
import helx
import gym

helx.ui.define_flags_from_hparams(helx.agents.DQNhparams)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.info("Starting")

    # environment
    gym_env = gym.make("Pong-v0")
    env = helx.environment.Environment.make(gym_env)

    # optimiser
    optimiser = optax.rmsprop(
        learning_rate=FLAGS.learning_rate,
        momentum=FLAGS.gradient_momentum,
        eps=FLAGS.min_squared_gradient,
        centered=True,
    )

    # agent
    hparams = helx.ui.hparams_from_flags(helx.agents.DQNhparams, FLAGS)
    network = nn.Sequential(
        [nn.Dense(features=64), nn.Dense(features=env.action_spec.num_values)]
    )
    agent = helx.agents.DQN(
        network=network, optimiser=optimiser, hparams=hparams, seed=0
    )

    helx.experiment.run(agent, env, 100)


if __name__ == "__main__":
    app.run(main)
