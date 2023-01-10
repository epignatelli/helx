from typing import cast

import gymnasium
import bsuite
import optax
from absl import app, flags, logging

import helx
from helx.networks import (
    MLP,
    Actor,
    AgentNetwork,
    DoubleQCritic,
    SoftmaxPolicy,
    Temperature,
)

helx.flags.define_flags_from_hparams(helx.agents.SACHparams)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.info("Starting")

    # environment
    env = bsuite.load_from_id("catch/0")
    env = helx.environment.make_from(env)

    # optimiser
    optimiser = optax.adam(learning_rate=FLAGS.learning_rate)

    # agent
    action_space = cast(helx.spaces.Discrete, env.action_space())
    hparams = helx.flags.hparams_from_flags(
        helx.agents.SACHparams,
        obs_space=env.observation_space(),
        action_space=env.action_space(),
        dim_A=action_space.n_bins,
        replay_start=10,
        batch_size=2,
    )
    actor_net = Actor(
        representation_net=MLP(features=[128, 128]),
        policy_head=SoftmaxPolicy(action_space.n_bins),
    )
    critic_net = DoubleQCritic(
        representation_net_a=MLP(features=[128, 128]),
        representation_net_b=MLP(features=[128, 128]),
        n_actions=hparams.dim_A,
    )
    extra_net = Temperature()
    network = AgentNetwork(
        actor_net=actor_net,
        critic_net=critic_net,
        extra_net=extra_net,
    )
    agent = helx.agents.SAC(
        network=network, optimiser=optimiser, hparams=hparams, seed=0
    )

    helx.experiment.run(agent, env, 100)


if __name__ == "__main__":
    app.run(main)
