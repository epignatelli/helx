from functools import partial
from typing import NamedTuple, Tuple

import dm_env
import jax
import jax.numpy as jnp
import wandb
from dm_env import specs
from jax.experimental import stax
from jax.experimental.optimizers import OptimizerState, rmsprop_momentum

from ...jax import pure
from ...nn.module import Module, module
from ...optimise.optimisers import Optimiser
from ...typing import Loss, Params, Size
from .. import td
from ..buffer import ReplayBuffer, Transition
from ..base import Agent


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
                stax.Softmax,
            ),  #  actor
            stax.serial(
                stax.Dense(1),
            ),  # critic
        ),
    )


class A2C(Agent):
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
    ):
        # public:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.rng = jax.random.PRNGKey(hparams.seed)
        self.buffer = ReplayBuffer(1, hparams.seed)
        network = Cnn(action_spec.num_values)
        optimiser = Optimiser(
            *rmsprop_momentum(
                step_size=hparams.learning_rate,
                gamma=hparams.squared_gradient_momentum,
                momentum=hparams.gradient_momentum,
                eps=hparams.min_squared_gradient,
            )
        )
        super().__init__(network, optimiser, hparams)

        # private:
        _, params = self.network.init(self.rng, (-1, *obs_spec.shape))
        self._opt_state = self.optimiser.init(params)

    @partial(pure, static_argnums=(1,))
    def preprocess(x, size: Size):
        luminance_mask = jnp.array([0.2126, 0.7152, 0.0722]).reshape(1, 1, 1, 3)
        y = jnp.sum(x * luminance_mask, axis=-1).squeeze()
        target_shape = (*x.shape[:-3], *size)
        s = jax.image.resize(y, target_shape, method="bilinear")
        return s

    @pure
    def loss(
        params: Params,
        transition: Transition,
    ) -> Loss:
        """We run the policy π for T timesteps"""
        logits, v_0 = A2C.network.apply(params, transition.x_0)
        # Policy gradient loss (log prob of the policy)
        actor_loss = jax.nn.log_softmax(logits)
        # Policy entropy
        entropy = -jnp.sum(jnp.mean(logits) * jnp.log(logits))
        # Value loss (Regression on bellman target)
        target = td.nstep_return(transition, v_0)
        _, v_1 = A2C.network.apply(params, transition.x_1)
        critic_loss = jnp.sqrt(jnp.mean(jnp.square(target - v_1)))
        return critic_loss + actor_loss - A2C.hparams.beta * entropy

    @pure
    def sgd_step(
        iteration: int,
        opt_state: OptimizerState,
        transition: Transition,
    ) -> Tuple[Loss, OptimizerState]:
        params = A2C.optimiser.params(opt_state)
        backward = jax.value_and_grad(A2C.loss, argnums=2)
        error, grads = backward(A2C.network, params, transition)
        return error, A2C.optimiser.update(iteration, grads, opt_state)

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Selects an action using a softmax policy"""
        params = self.optimiser.params(self._opt_state)
        logits, _ = self.network.apply(params, timestep.observation)
        action = jax.random.categorical(self.rng, logits).squeeze()  # on-policy action
        return int(action)

    def update(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> None:
        timestep = timestep._replace(
            observation=self.preprocess(timestep.observation, (56, 56)),
        )
        new_timestep = new_timestep._replace(
            observation=self.preprocess(new_timestep.observation, (56, 56)),
        )
        self.buffer.add(timestep, action, new_timestep)

        loss = None
        if self.buffer.full():
            transition = self.buffer.sample(self.hparams.batch_size)
            loss, self.opt_state = self.sgd_step(
                self.network, self.optimiser, transition
            )
        return loss

    def log(self, reward: float, loss: Loss):
        wandb.log({"Iteration": float(self._iteration)})
        wandb.log({"Reward": reward})
        if loss is not None:
            wandb.log({"Loss": float(loss)})
        return
