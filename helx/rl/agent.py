import abc
import logging
from typing import Callable, NamedTuple, Sequence
from multiprocessing import Process

import dm_env
import jax
import jax.numpy as jnp
from helx.nn.module import Module
from helx.optimise.optimisers import Optimiser
from helx.typing import Action, Batch, HParams, Loss, Params
from jaxlib.xla_extension import Device

Policy = Callable


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
        """Logging function"""

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

    def run_distributed(
        self,
        env: dm_env.Environment,
        n_actors: int = 1,
        n_learners: int = 1,
        num_episodes: int = 1000,
        eval: bool = False,
    ):
        """Start the training routine of a Reinforcement Learning agent
        using multiple instances of the environment to increase throughput"""

        envs = ParallelEnv(env, n=n_actors)

        def act(experience_buffer, policy_buffer):
            pass

        def learn(experience_buffer, policy_buffer):
            pass

        experience_buffer = None
        policy_buffer = PolicyBuffer()

        actors = Process(
            target=act,
            args=(experience_buffer, policy_buffer),
            daemon=True,
            name="actors",
        )


class PolicyBuffer(NamedTuple):
    policy: Params = None

    def get(self):
        return self.policy

    def put(self, policy):
        self.policy = policy


class Actor:
    def __init__(
        self,
        policy: Policy,
        params: Params,
        env: dm_env.Environment,
        n_steps: int = 1,
        device: Device = None,
    ) -> None:
        #  public:
        self.env: dm_env.Environment = env
        self.policy: Callable = policy
        self.n_steps: int = n_steps
        self.trajectory: Sequence[dm_env.TimeStep] = [self.reset()]

        #  private:
        self._device: Device = device if device is not None else jax.local_devices()[0]
        self._params: Params = params
        self._policy_distribution = []

    @property
    def params(self):
        return self._params

    @property
    def device(self):
        return self._device

    @property
    def probs(self):
        return self._probs

    def update(self, params: Params):
        self.params = params

    def send(self):
        return (self.trajectory, self._params)

    def step(self, policy):
        #  apply policy
        probs = policy(self.trajectory[-1].observation)
        self._probs.append(probs)
        action = jnp.argmax(probs, axis=-1)
        #  step in the environment
        new_timestep = self.env.step(action)
        #  store experience
        self.trajectory.append(new_timestep)
        return new_timestep

    def reset(self):
        return self.env.reset()
