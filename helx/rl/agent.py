import abc
import logging

import dm_env
from helx.nn.module import Module
from helx.optimise.optimisers import Optimiser
from helx.typing import Action, HParams, Loss


class IAgent:
    network: Module
    optimiser: Optimiser
    hparams: HParams

    def __init__(
        self,
        network: Module,
        optimiser: Optimiser,
        hparams: HParams,
        logging: bool = False,
    ):
        IAgent.network = network
        IAgent.optimiser = optimiser
        IAgent.hparams = hparams
        self.logging = logging
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

    @abc.abstractmethod
    def log(
        self,
        timestep: dm_env.TimeStep,
        action: Action,
        new_timestep: dm_env.TimeStep,
        loss: Loss = None,
        log_frequency: int = 1,
    ):
        """Empty by default, can be overriden to record learning statistics"""

    def run(
        self,
        env: dm_env.Environment,
        num_episodes: int,
        eval: bool = False,
    ) -> Loss:
        logging.info(
            "Starting {} agent {} on environment {}.\nThe scheduled number of episode is {}".format(
                "evaluating" if eval else "training", self, env, num_episodes
            )
        )
        logging.info(
            "The hyperparameters for the current experiment are {}".format(
                self.hparams._asdict()
            )
        )
        for episode in range(num_episodes):
            print(
                "Episode {}/{}\t\t\t".format(episode, num_episodes - 1),
                end="\r",
            )
            #  initialise environment
            episode_reward = 0.0
            timestep = env.reset()
            while not timestep.last():
                #  apply policy
                action = self.policy(timestep)
                #  observe new state
                new_timestep = self.observe(env, timestep, action)
                episode_reward += new_timestep.reward
                print(
                    "Episode reward {}\t\t".format(episode_reward),
                    end="\r",
                )
                #  update policy
                loss = None
                if not eval:
                    loss = self.update(timestep, action, new_timestep)
                #  log update
                if self.logging:
                    self.log(timestep, action, new_timestep, loss)
                # prepare next iteration
                timestep = new_timestep
        return loss
