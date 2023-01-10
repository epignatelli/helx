import os
from typing import cast

import flax.linen as nn
import jax
import optax

import helx.agents
import helx.networks.modules
import helx.spaces


def test_agent_serialisation():
    hparams = helx.agents.DQNHparams()

    # optimiser
    optimiser = optax.rmsprop(
        learning_rate=hparams.learning_rate,
        momentum=hparams.gradient_momentum,
        eps=hparams.min_squared_gradient,
        centered=True,
    )

    # agent
    n_actions = cast(helx.spaces.Discrete, 3).n_bins

    critic_net = nn.Sequential(
        [
            helx.networks.modules.Flatten(),
            helx.networks.modules.MLP(features=[32, 16]),
            nn.Dense(features=n_actions),
        ]
    )
    network = helx.networks.modules.AgentNetwork(critic_net=critic_net)
    agent = helx.agents.DQN(
        network=network, optimiser=optimiser, hparams=hparams, seed=0
    )

    agent.save(os.path.join(os.path.dirname(__file__), "tmp"))
    agent_restored = helx.agents.DQN.load(os.path.join(os.path.dirname(__file__), "tmp"))

    assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), agent.params, agent_restored))