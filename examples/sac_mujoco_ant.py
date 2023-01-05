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
    action_dim = action_space.shape[0]
    hparams = helx.flags.hparams_from_flags(
        helx.agents.SACHparams,
        FLAGS,
        input_shape=env.observation_space().shape,
        dim_A=action_dim,
    )
    representation_net = nn.Sequential(
        [
            nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4)),
            nn.relu,
            nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2)),
            nn.relu,
            nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1)),
            nn.relu,
            helx.networks.Flatten(),
        ]
    )
    actor_net = nn.Dense(features=action_space.shape[0])
    critic_net = nn.Dense(features=1)
    network = helx.networks.AgentNetwork(
        representation_net=representation_net,
        actor_net=actor_net,
        critic_net=critic_net,
    )
    agent = helx.agents.SAC(
        network=network, optimiser=optimiser, hparams=hparams, seed=0
    )

    helx.experiment.run(agent, env, 100)


if __name__ == "__main__":
    app.run(main)
