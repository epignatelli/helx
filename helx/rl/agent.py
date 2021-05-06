import abc
from typing import Tuple

import dm_env
import jax
from bsuite.baselines import base
from helx.rl.buffer import Transition
from jax.experimental.optimizers import OptimizerState

from ..jax import pure
from ..nn.module import Module
from ..optimise.optimisers import Optimiser
from ..typing import HParams, Loss, Params, Reward


class Agent(base.Agent):
    network: Module
    optimiser: Optimiser
    hparams: HParams

    def __init__(self, network, optimiser, hparams):
        Agent.network = network
        Agent.optimiser = optimiser
        Agent.hparams = hparams
        self._iteration = 0

    @abc.abstractmethod
    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: int
    ) -> Transition:
        """The agent's observation function defines how  it interacts with the enviroment"""

    @abc.abstractmethod
    def loss(params: Params, transition: Transition, hparams: HParams) -> Loss:
        """Specifies the loss function to be minimised by some flavour of SGD"""

    @abc.abstractmethod
    def policy(self, timestep: dm_env.TimeStep) -> int:
        """The agent's policy function that maps an observation to an action"""

    @pure
    def sgd_step(
        iteration: int,
        opt_state: OptimizerState,
        transition: Transition,
    ) -> Tuple[Loss, OptimizerState]:
        """Performs a gradient descent step"""
        params = Agent.optimiser.params(opt_state)
        backward = jax.value_and_grad(Agent.loss, argnums=2)
        error, grads = backward(Agent.network, params, transition)
        return error, Agent.optimiser.update(iteration, grads, opt_state)

    def log(self, reward: Reward, loss: Loss):
        return
