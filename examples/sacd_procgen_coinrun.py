from typing import cast

import gym
import optax
from absl import app, flags, logging

import flax.linen as nn


import helx
from helx.networks import (
    CNN,
    deep_copy,
    Flatten,
)

helx.flags.define_flags_from_hparams(helx.agents.SACHparams)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.info("Starting")

    # environment
    env = gym.make(
        "procgen:procgen-coinrun-v0",
        apply_api_compatibility=True,
        max_episode_steps=100,
    )
    env = helx.environment.make_from(env)

    # optimiser
    optimiser = optax.adam(learning_rate=FLAGS.learning_rate)

    # agent
    hparams = helx.flags.hparams_from_flags(
        helx.agents.SACHparams,
        obs_space=env.observation_space(),
        action_space=env.action_space(),
        replay_start=10,
        batch_size=2,
    )
    actor_representation_net = nn.Sequential(
        [
            CNN(
                features=(4, 4),
                kernel_sizes=((4, 4), (4, 4)),
                strides=((2, 2), (2, 2)),
                paddings=("SAME", "SAME"),
            ),
            Flatten(),
        ]
    )
    critic_representation_net = deep_copy(actor_representation_net)
    agent = helx.agents.SACD(
        hparams=hparams,
        optimiser=optimiser,
        seed=0,
        actor_representation_net=actor_representation_net,
        critic_representation_net=critic_representation_net,
    )

    helx.experiment.run(agent, env, 2)


if __name__ == "__main__":
    app.run(main)
