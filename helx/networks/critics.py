from __future__ import annotations

from typing import Tuple

import flax.linen as nn
from chex import Array

from .modules import Identity


class Critic(nn.Module):
    """Defines the critic head of an agent (i.e., the value function).
    Args:
        n_actions (int): The number of actions in the action space."""

    critic_head: nn.Module
    representation_net: nn.Module = Identity()

    @nn.compact
    def __call__(self, representation: Array, *args, **kwargs) -> Array:
        representation = self.representation_net(representation, *args, **kwargs)
        return self.critic_head(representation)


class DoubleQCritic(nn.Module):
    """Defines a double Q-critic network.
    Args:
        n_actions (int): The number of outputs of the network, i.e., the number of
        actions in a discrete action space, or the dimensionality of each action
        in a continuous action space."""

    n_actions: int
    representation_net_a: nn.Module
    representation_net_b: nn.Module

    @nn.compact
    def __call__(self, observation: Array, *args, **kwargs) -> Tuple[Array, Array]:
        representation_a = self.representation_net_a(observation, *args, **kwargs)
        representation_b = self.representation_net_b(observation, *args, **kwargs)
        q1 = nn.Dense(features=self.n_actions)(representation_a)
        q2 = nn.Dense(features=self.n_actions)(representation_b)
        return q1, q2
