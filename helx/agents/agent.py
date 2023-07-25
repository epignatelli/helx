from __future__ import annotations
from abc import abstractmethod

from typing import Collection, Tuple

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


class AgentState(struct.PyTreeNode):
    iteration: Array
    opt_state: OptState


class Agent(struct.PyTreeNode):
    hparams: HParams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)

    @abstractmethod
    def init(self, *, key: KeyArray) -> AgentState:
        raise NotImplementedError

    @abstractmethod
    def sample_action(
        self, agent_state: AgentState, obs: Array, *, key: KeyArray, eval: bool = False
    ):
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        agent_state: AgentState,
        transitions: Timestep,
        cached_log: Log,
        *,
        key: KeyArray,
    ) -> Tuple[AgentState, Log]:
        raise NotImplementedError
