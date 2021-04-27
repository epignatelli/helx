from functools import partial
from typing import NamedTuple, Tuple

import dm_env
import jax
import jax.numpy as jnp
from bsuite.baselines.base import Action, Agent
from dm_env import specs
from jax.experimental import stax
from jax.experimental.optimizers import OptimizerState, rmsprop_momentum

from ...methods import module, pure
from ...types import Logits, Loss, Module, Optimiser, Params, Value
from .. import td
from ..buffer import ReplayBuffer, Transition


class Hparams(NamedTuple):
    seed: int
    discount: float
    trace_decay: float
    n_steps: int


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
    with n-step advantage baseline"""

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        hparams: Hparams,
    ):
        # public:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.hparams = hparams
        self.rng = jax.random.PRNGKey(hparams.seed)
        self.buffer = ReplayBuffer(hparams.buffer_capacity, hparams.seed)
        self.network = Cnn(action_spec.num_values)
        self.optimiser = Optimiser(
            *rmsprop_momentum(
                step_size=hparams.learning_rate,
                gamma=hparams.squared_gradient_momentum,
                momentum=hparams.gradient_momentum,
                eps=hparams.min_squared_gradient,
            )
        )

        # private:
        self._iteration = 0
        _, params = self.network.init(self.rng, obs_spec.shape)
        self._opt_state = self.optimiser.init(params)

    @partial(pure, static_argnums=(0,))
    def loss(
        model: Module,
        params: Params,
        transition: Transition,
    ) -> Tuple[Loss, Tuple[Logits, Value]]:
        logits, v_0 = model.apply(params, transition.x_0)
        target = td.nstep_return(transition, v_0)
        _, v_1 = model.apply(params, transition.x_1)
        critic_loss = jnp.sqrt(jnp.mean(jnp.square(target - v_1)))
        actor_loss = jax.nn.log_softmax(logits)
        return critic_loss + actor_loss

    @partial(pure, static_argnums=(0, 1))
    def sgd_step(
        model: Module,
        optimiser: Optimiser,
        iteration: int,
        opt_state: OptimizerState,
        transition: Transition,
    ):
        params = optimiser.params(opt_state)
        backward = jax.value_and_grad(A2C.loss, argnums=2)
        error, grads = backward(model, params, transition)
        return error, optimiser.update(iteration, grads, opt_state)

    def select_action(self, timestep: dm_env.TimeStep) -> Action:
        """Selects an action using a softmax policy"""
        logits, _ = self._forward(self._state.params, timestep.observation)
        action = jax.random.categorical(self.rng, logits).squeeze()
        return int(action)

    def update(
        self, timestep: dm_env.TimeStep, action: Action, new_timestep: dm_env.TimeStep
    ) -> None:
        self.buffer.add(timestep, action, new_timestep)
        transition = self.buffer.sample(self.hparams.batch_size)
        loss, opt_state = self.sgd_step(self.network, self.optimiser, transition)
        return loss, opt_state
