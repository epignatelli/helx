import abc

import dm_env

from ..nn.module import Module
from ..optimise.optimisers import Optimiser
from ..typing import Action, HParams, Loss


class IAgent:
    network: Module
    optimiser: Optimiser
    hparams: HParams

    def __init__(self, network, optimiser, hparams):
        IAgent.network = network
        IAgent.optimiser = optimiser
        IAgent.hparams = hparams
        self._iteration = 0

    @abc.abstractmethod
    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: int
    ) -> dm_env.TimeStep:
        """The agent's observation function defines how  it interacts with the enviroment"""

    @abc.abstractmethod
    def policy(self, timestep: dm_env.TimeStep) -> int:
        """The agent's policy function that maps an observation to an action"""

    @abc.abstractmethod
    def update(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> float:
        """The agent's policy function that maps an observation to an action"""

    def log(
        self,
        timestep: dm_env.TimeStep,
        action: Action,
        new_timestep: dm_env.TimeStep,
        loss: Loss = None,
    ):
        pass
