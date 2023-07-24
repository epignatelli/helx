from __future__ import annotations
from abc import abstractmethod

from typing import Tuple

from jax import Array
from jax.random import KeyArray
from flax import struct
import flax.linen as nn
from optax import GradientTransformation, OptState

from ..spaces import Space
from ..mdp import Timestep, StepType


class HParams(struct.PyTreeNode):
    obs_space: Space
    action_space: Space
    discount: float = 0.99
    n_steps: int = 1
    seed: int = 0


class Log(struct.PyTreeNode):
    iteration: Array
    loss: Array
    step_type: StepType
    returns: Array


class TrainState(struct.PyTreeNode):
    iteration: Array
    opt_state: OptState


class Agent(nn.Module):
    hparams: HParams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)

    @abstractmethod
    def sample_action(self, key: KeyArray, obs: Array, eval: bool = False):
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        key: KeyArray,
        transitions: Timestep,
        cached_log: Log,
    ) -> Tuple[Agent, Log]:
        raise NotImplementedError
