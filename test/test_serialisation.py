import os

import flax.linen as nn
import jax
import optax

import helx.agents
import helx.networks
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
    representation_net = nn.Sequential(
        [
            helx.networks.modules.Flatten(),
            helx.networks.modules.MLP(features=[32, 16]),
        ]
    )
    agent = helx.agents.DQN(
        hparams=hparams,
        optimiser=optimiser,
        seed=0,
        representation_net=representation_net
    )

    agent.save(os.path.join(os.path.dirname(__file__), "tmp"))
    agent_restored = helx.agents.DQN.load(
        os.path.join(os.path.dirname(__file__), "tmp")
    )

    assert jax.tree_util.tree_all(
        jax.tree_map(lambda x, y: (x == y).all(), agent.params, agent_restored)
    )