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
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: Action
    ) -> dm_env.TimeStep:
        """The agent's observation function defines how  it interacts with the enviroment"""

    @abc.abstractmethod
    def policy(self, timestep: dm_env.TimeStep) -> Action:
        """The agent's policy function that maps an observation to an action"""

    @abc.abstractmethod
    def update(
        self, timestep: dm_env.TimeStep, action: Action, new_timestep: dm_env.TimeStep
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
        """Learner and actors run synchronously."""
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

    def run_async_cpu(
        self,
        env: dm_env.Environment,
        num_episodes: int,
        n_actors: int,
        eval: bool = False,
    ) -> Loss:
        """Implements an IMPALA-like architecture where learners and actors run asynchronously.
        See: https://arxiv.org/abs/1802.01561
        Actors are distributed on the CPU, whilst learners are distributed on the GPU.
        Note that the communication between processes happens using UNIX pipes,
        and is limited to a single host."""
        params_buffer: mp.Queue = mp.Queue(1)
        exp_buffer: Queue = Queue(
            env.observation_spec, self.hparams.n_steps, self.hparams.batch_size
        )

        def producer(params, q):
            #  perform env.step perpetually
            timestep = env.reset()
            while True:
                action = self.policy(timestep, params)
                new_timestep = env.step(action)
                q.put(new_timestep)
                timestep = new_timestep

        def consumer():
            while True:
                trajectories = exp_buffer.sample()
                self.update(None, None, None)

    def run_async_gpu(
        self,
        env: dm_env.Environment,
        num_episodes: int,
        n_actors: int,
        eval: bool = False,
    ) -> Loss:
        """Implements a SEED RL-like architecture where learners and actors run asynchronously.
        See: https://arxiv.org/abs/1910.06591
        In contrast to an IMPALA-like architecture, both actors and learners are distributed on the GPU.
        Note that the communication between processes happens using UNIX pipes,
        and is limited to a single host."""
        raise NotImplementedError
