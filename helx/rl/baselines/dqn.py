from functools import partial
from typing import NamedTuple

import dm_env
import jax
import jax.numpy as jnp
from bsuite.baselines import base
from dm_env import specs
from jax.experimental.optimizers import OptimizerState, rmsprop_momentum
from jax.experimental.stax import Conv, Dense, Flatten, Relu, serial

from ...methods import module, pure
from ...types import Module, Optimiser, Params, Shape, Transition
from ..buffer import ReplayBuffer


class HParams(NamedTuple):
    batch_size: Shape = 32
    replay_memory_size: int = 1000000
    agent_history_len: int = 4
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    action_repeat: int = 4
    update_frequency: int = 4
    learning_rate: float = 0.00025
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01
    initial_exploration: float = 1.0
    final_exploration: float = 0.01
    final_exploration_frame: int = 1000000
    replay_start: int = 50000
    no_op_max: int = 30


class DqnParams(NamedTuple):
    online: Params
    target: Params


@module
def Cnn(n_actions: int) -> Module:
    return serial(
        Conv(32, (8, 8), (4, 4), "VALID"),
        Relu,
        Conv(64, (4, 4), (2, 2), "VALID"),
        Relu,
        Conv(64, (3, 3), (1, 1), "VALID"),
        Relu,
        Flatten,
        Dense(512),
        Relu,
        Dense(n_actions),
    )


class Dqn(base.Agent):
    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        in_shape: Shape,
        hparams: HParams,
        seed: int = 0,
    ):
        # public:
        self.action_spec = action_spec
        self.obs_spec = obs_spec
        self.hparams = hparams
        self.epsilon = hparams.initial_exploration
        self.replay_buffer = ReplayBuffer(hparams.replay_memory_size)
        self.network = Cnn(action_spec.num_values)
        self.rng = jax.random.PRNGKey(seed)
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
        _, online_params = self.network.init(self.rng, (-1, *in_shape))
        _, target_params = self.network.init(self.rng, (-1, *in_shape))
        self._params = DqnParams(online_params, target_params)
        self._opt_state = self.optimiser.init(online_params)

    @pure
    def preprocess(x, size=(84, 84)):
        # get luminance
        luminance_mask = jnp.array([0.2126, 0.7152, 0.0722]).reshape(1, 1, 1, 3)
        y = jnp.sum(x * luminance_mask, axis=-1).squeeze()
        target_shape = (*x.shape[:-3], *size)
        s = jax.image.resize(y, target_shape, method="bilinear")
        return s

    @partial(pure, static_argnums=(0,))
    def loss(
        model: Module,
        params_online: Params,
        params_target: Params,
        transition: Transition,
    ) -> jnp.ndarray:
        q_target = model.apply(params_target, transition.x_0)
        q_behaviour = model.apply(params_online, transition.x_0)
        # get the q target
        y = transition.r_1 + transition.gamma * jnp.max(q_target, axis=1)
        y = y.reshape(-1, 1)
        return jnp.mean(jnp.power((y - q_behaviour), 2))

    @partial(pure, static_argnums=(0, 1))
    def sgd_step(
        model: Module,
        optimiser: Optimiser,
        iteration: int,
        opt_state: OptimizerState,
        params: Params,
        transition: Transition,
    ):
        backward = jax.value_and_grad(Dqn.loss, argnums=1)
        loss, gradients = backward(model, params.online, params.target, transition)
        return loss, optimiser.update(iteration, gradients, opt_state)

    def anneal_epsilon(self):
        x0, y0 = (self.hparams.replay_start, self.hparams.initial_exploration)
        x1, y1 = (
            self.hparams.final_exploration_frame,
            self.hparams.final_exploration,
        )
        x = self._iteration
        y = ((y1 - y0) * (x - x0) / (x1 - x0)) + y0
        return y

    def select_action(
        self,
        timestep: dm_env.TimeStep,
    ) -> base.Action:
        """Policy function: maps the current observation/state to an action
        following an epsilon-greedy policy
        """
        # return random action with epsilon probability
        if jax.random.uniform(self.rng, (1,)) < self.epsilon:
            return jax.random.randint(self.rng, (1,), 0, self.n_actions)

        state = timestep.observation[None, ...]  # batching
        q_values = self.forward(self._online_params, state)
        # e-greedy
        action = jnp.argmax(q_values, axis=-1)
        return action

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ) -> float:
        # increment iteration
        self._iteration += 1

        # env has called reset(), do nothing
        if timestep.first():
            return

        # preprocess observations
        timestep = timestep._replace(observation=self.preprocess(timestep.observation))
        new_timestep = new_timestep._replace(
            observation=self.preprocess(new_timestep.observation)
        )

        # add experience to replay buffer
        self.replay_buffer.add(timestep, action, new_timestep)

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.replay_buffer) < self.hparams.replay_start:
            return

        # update the online parameters only every n interations
        if self._iteration % self.hparams.update_frequency:
            return

        # the exploration parameter is linearly interpolated to the end value
        self.epsilon = self.anneal_epsilon()

        # update the online parameters
        transition = self.replay_buffer.sample(self.hparams.batch_size)
        loss, self._opt_state = self.sgd_step(
            self.network,
            self.optimiser,
            self._iteration,
            self._opt_state,
            self._params,
            transition,
        )
        online_params = self.optimiser.params(self._opt_state)
        self._params = self._params._replace(online=online_params)

        # update the target network parameters every n step
        if self._iteration % self.hparams.target_network_update_frequency == 0:
            self._params = self._params._replace(target=online_params)
        return loss
