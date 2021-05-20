from typing import Callable, NamedTuple, Tuple

import dm_env
import jax
import jax.numpy as jnp
import wandb
from dm_env import TimeStep, specs
from helx.jax import pure
from helx.nn.module import Module, module
from helx.optimise.optimisers import Optimiser
from helx.typing import Action, Loss, Params, Size
from jax.experimental import stax
from jax.experimental.optimizers import OptimizerState, rmsprop_momentum

from .. import pg, td
from ..agent import IAgent
from ..memory import OnlineBuffer, Trajectory


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


class A2C(IAgent):
    """Synchronous on-policy, online actor-critic algorithm
    with n-step advantage baseline.
    See:
    Mnih V., 2016, https://arxiv.org/pdf/1602.01783.pdf and
    Sutton R., Barto. G., 2018, http://incompleteideas.net/book/RLbook2020.pdf

    This algorithm implements Mnih, V. 2016, that is,
    we ran the policy for T steps, where T = batch_size << len_episode
    and use the collected samples to update.
    """

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
        self.memory = OnlineBuffer(1, hparams.seed)
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
    def loss(
        params: Params,
        trajectory: Trajectory,
    ) -> Loss:
        """We run the policy π for T timesteps,
        to calculate the advantage"""
        #  inference
        logits, values = A2C.network.apply(params, trajectory.observations)
        #   Critic loss (Bellman regression)
        advantages = td.advantages(trajectory, values[:-1], values[1:])
        critic_loss = jnp.mean(jnp.square(advantages))  #  bellman mse
        #  Actor loss (Policy gradient loss)
        actor_loss = pg.a2c_softmax_loss(logits, advantages)
        entropy = pg.softmax_entropy(logits)
        return critic_loss + actor_loss - A2C.hparams.beta * entropy

    @pure
    def sgd_step(
        iteration: int,
        opt_state: OptimizerState,
        transition: Trajectory,
    ) -> Tuple[Loss, OptimizerState]:
        params = A2C.optimiser.params(opt_state)
        backward = jax.value_and_grad(A2C.loss, argnums=2)
        error, grads = backward(A2C.network, params, transition)
        return error, A2C.optimiser.update(iteration, grads, opt_state)

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
        """Selects an action using a softmax policy"""
        params = self.optimiser.params(self._opt_state)
        logits, _ = self.network.apply(params, timestep.observation)
        action = jax.random.categorical(self.rng, logits).squeeze()  # on-policy action
        return int(action)

    def update(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> None:

        loss = None
        if self.memory.full():
            transition = self.memory.sample(self.hparams.batch_size)
            loss, self.opt_state = self.sgd_step(
                self.network, self.optimiser, transition
            )
        return loss

    def log(
        self,
        timestep: TimeStep,
        action: Action,
        new_timestep: TimeStep,
        loss: Loss = None,
        log_frequency: int = 1,
    ):
        wandb.log({"Iteration": float(self._iteration)})
        wandb.log({"Action": int(action)})
        wandb.log({"Reward": float(new_timestep.reward)})
        if self._iteration % 100 == 0:
            wandb.log({"Observation": wandb.Image(timestep.observation)})
        if loss is not None:
            wandb.log({"Loss": float(loss)})
        return
