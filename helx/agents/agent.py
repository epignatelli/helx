from __future__ import annotations
from abc import abstractmethod

from jax import Array
from jax.random import KeyArray
import jax.numpy as jnp
from flax import struct
from optax import GradientTransformation, OptState

from ..spaces import Space
from ..mdp import Timestep, TRANSITION


class HParams(struct.PyTreeNode):
    obs_space: Space
    action_space: Space
    discount: float = 0.99
    n_steps: int = 1
    seed: int = 0


class Log(struct.PyTreeNode):
    iteration: Array = jnp.asarray(0)
    step_type: Array = TRANSITION
    returns: Array = jnp.asarray(0.0)


class AgentState(struct.PyTreeNode):
    iteration: Array
    opt_state: OptState
    log: Log


class Agent(struct.PyTreeNode):
    hparams: HParams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)

    @abstractmethod
    def create(*args) -> Agent:
        return

    @abstractmethod
    def init(self, key: KeyArray, timestep: Timestep) -> AgentState:
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
        timesteps: Timestep,
        *,
        key: KeyArray,
    ) -> AgentState:
        raise NotImplementedError
