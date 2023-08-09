from __future__ import annotations

import jax.numpy as jnp
import optax
from flax import struct
import flax.linen as nn
import jax.tree_util as jtu

from .dqn import DQNHParams, DQNLog, DQNState, DQN
from ..modules import Split, Merge, Parallel


class DuelingDQNHParams(DQNHParams):
    ...


class DuelingDQNLog(DQNLog):
    ...


class DuelingDQNState(DQNState):
    ...


class DuelingDQN(DQN):
    hparams: DuelingDQNHParams = struct.field(pytree_node=True)
    optimiser: optax.GradientTransformation = struct.field(pytree_node=True)
    critic: nn.Module = struct.field(pytree_node=True)

    @classmethod
    def create(
        cls,
        hparams: DuelingDQNHParams,
        optimiser: optax.GradientTransformation,
        backbone: nn.Module,
    ) -> DuelingDQN:
        critic = nn.Sequential(
            [
                backbone,
                Split(2),
                Parallel((nn.Dense(1), nn.Dense(hparams.action_space.maximum))),  # v, A
                Merge(jnp.sum) # q = v + A
            ]
        )
        return DuelingDQN(
            hparams=hparams,
            optimiser=optimiser,
            critic=critic,
        )
