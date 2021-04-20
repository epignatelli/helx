from functools import partial
from typing import NamedTuple, Tuple

import dm_env
import jax
import jax.numpy as jnp
from bsuite.baselines.base import Action, Agent
from dm_env import specs
from helx.methods import module, pure
from helx.rl import ReplayBuffer
from helx.types import Module, Optimiser, Params, Transition
from jax.experimental import stax
from jax.experimental.optimizers import OptimizerState

from . import td

Logits = jnp.ndarray
Value = jnp.ndarray
Loss = float


class Hparams(NamedTuple):
    seed: int
    discount: float
    trace_decay: float
    n_steps: int


@module
def MLP(n_actions: int, hidden_size: int) -> Module:
    return stax.serial(
        stax.Flatten,
        stax.Dense(hidden_size),
        stax.Relu(),
        stax.FanOut(2),
        stax.parallel(
            stax.serial(
                stax.Dense(n_actions),
                stax.Softmax(),
            ),  #  actor
            stax.serial(
                stax.Dense(n_actions),
            ),  # critic
        ),
    )


class A2C(Agent):
    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        network: Module,
        optimiser: Optimiser,
        hparams: Hparams,
    ):
        # public:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.hparams = hparams
        self.rng = jax.random.PRNGKey(hparams.seed)
        self.buffer = ReplayBuffer(hparams.buffer_capacity, hparams.seed)
        self.network = network
        self.optimiser = optimiser

        # private:
        self._iteration = 0

    @partial(pure, static_argnums=(0, 1))
    def loss(
        model: Module,
        hparams: Hparams,
        params: Params,
        trajectory: Transition,
    ) -> Tuple[Loss, Tuple[Logits, Value]]:
        logits, values = model.apply(params, trajectory.s)
        returns = td.g_lambda(
            trajectory.r, values, hparams.discount, hparams.trace_decay
        )
        td = returns

    @partial(pure, static_argnums=(0, 1))
    def sgd_step(
        model: Module,
        optimiser: Optimiser,
        hparams: Hparams,
        opt_state: OptimizerState,
        transition: ,
    ):
        params = optimiser.params(opt_state)
        error, grads = jax.value_and_grad(A2C.loss, argnums=2)(
            model, hparams, params, transition
        )
        opt_state = optimiser.update(0, grads, opt_state)
        return error, opt_state

    def select_action(self, timestep: dm_env.TimeStep) -> Action:
        logits, _ = self._forward(self._state.params, timestep.observation)
        action = jax.random.categorical(self.rng, logits).squeeze()
        return int(action)

    def update(
        self, timestep: dm_env.TimeStep, action: Action, new_timestep: dm_env.TimeStep
    ) -> None:
        self.buffer.add(timestep, action, new_timestep)

        if self._iteration < self.hparams.min_buffer_size:
            return

        trajectory = self.buffer.sample(self.hparams.batch_size)
        loss, opt_state = self.sgd_step(self.network, self.optimiser, trajectory)
        return loss, opt_state
