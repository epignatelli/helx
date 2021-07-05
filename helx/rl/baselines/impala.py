from typing import Callable, NamedTuple

import dm_env
import jax
import wandb
from dm_env import specs
from helx.jax import pure
from helx.nn.module import Module, module
from helx.optimise.optimisers import Optimiser
from helx.rl.memory import Queue
from helx.typing import Action, Loss
from jax.experimental import stax
from jax.experimental.optimizers import rmsprop_momentum

from ..agent import IAgent


class HParams(NamedTuple):
    seed: int = 0
    discount: float = 1.0
    trace_decay: float = 1.0
    n_steps: int = 1
    beta: float = 0.001
    learning_rate: float = 0.001
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01
    batch_size: int = 32


@module
def Cnn(n_actions: int, hidden_size: int = 512) -> Module:
    return stax.serial(
        stax.Conv(32, (8, 8), (4, 4), "VALID"),
        stax.Relu,
        stax.Conv(64, (4, 4), (2, 2), "VALID"),
        stax.Relu,
        stax.Conv(64, (3, 3), (1, 1), "VALID"),
        stax.Relu,
        stax.Flatten,
        stax.Dense(hidden_size),
        stax.Relu,
        stax.FanOut(2),
        stax.parallel(
            stax.serial(
                stax.Dense(n_actions),
            ),  #  actor
            stax.serial(
                stax.Dense(1),
            ),  # critic
        ),
    )


class Impala(IAgent):
    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        hparams: HParams,
        preprocess: Callable = lambda x: x,
        logging: bool = False,
    ):
        # public:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.rng = jax.random.PRNGKey(hparams.seed)
        self.memory = Queue(obs_spec, 1, hparams.batch_size)
        self.preprocess = preprocess
        network = Cnn(action_spec.num_values)
        optimiser = Optimiser(
            *rmsprop_momentum(
                step_size=hparams.learning_rate,
                gamma=hparams.squared_gradient_momentum,
                momentum=hparams.gradient_momentum,
                eps=hparams.min_squared_gradient,
            )
        )
        super().__init__(network, optimiser, hparams, logging)

        # private:
        _, params = self.network.init(self.rng, (-1, *obs_spec.shape))
        self._opt_state = self.optimiser.init(params)

    @pure
    def loss(params, trajectory):
        pass

    @pure
    def sgd_step(iteration, opt_state, trajectory):
        pass

    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: int
    ) -> dm_env.TimeStep:
        #  iterate over the number of steps
        for t in range(self.hparams.n_steps):
            #  get new MDP state
            new_timestep = env.step(action)
            #  store transition into the replay buffer
            self.memory.add(timestep, action, new_timestep, preprocess=self.preprocess)
            timestep = new_timestep
        return timestep

    def policy(self, timestep: dm_env.TimeStep) -> int:
        params = self.optimiser.params(self._opt_state)
        logits, _ = self.network.apply(params, timestep.observation)
        action = jax.random.categorical(self.rng, logits).squeeze()  # on-policy action
        return int(action)

    def update(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> float:
        trajectory = self.memory.sample()
        loss, opt_state = self.sgd_step(self.network, self.optimiser, trajectory)
        return loss, opt_state

    def log(
        self,
        timestep: dm_env.TimeStep,
        action: Action,
        new_timestep: dm_env.TimeStep,
        loss: Loss,
        log_frequency: int,
    ):
        wandb.log({"Iteration": float(self._iteration)})
        wandb.log({"Action": int(action)})
        wandb.log({"Reward": float(new_timestep.reward)})
        if self._iteration % 100 == 0:
            wandb.log({"Observation": wandb.Image(timestep.observation)})
        if loss is not None:
            wandb.log({"Loss": float(loss)})
        return
