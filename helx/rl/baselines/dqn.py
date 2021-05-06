from typing import Callable, NamedTuple, Tuple

import dm_env
import jax
import jax.numpy as jnp
import wandb
from dm_env import Environment, TimeStep, specs
from jax.experimental.optimizers import OptimizerState, rmsprop_momentum
from jax.experimental.stax import Conv, Dense, Flatten, Relu, serial

from ...jax import pure
from ...nn.module import Module, module
from ...optimise.optimisers import Optimiser
from ...typing import Loss, Observation, Params, Shape
from .. import td
from ..buffer import OfflineBuffer, Transition
from ..agent import Agent


class HParams(NamedTuple):
    batch_size: Shape = 32
    replay_memory_size: int = 1000
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    learning_rate: float = 0.00025
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01
    initial_exploration: float = 1.0
    final_exploration: float = 0.01
    final_exploration_frame: int = 1000000
    replay_start: int = 1000
    hidden_size: int = 512


@module
def Cnn(n_actions: int, hidden_size) -> Module:
    return serial(
        Conv(32, (8, 8), (4, 4), "VALID"),
        Relu,
        Conv(64, (4, 4), (2, 2), "VALID"),
        Relu,
        Conv(64, (3, 3), (1, 1), "VALID"),
        Relu,
        Flatten,
        Dense(hidden_size),
        Relu,
        Dense(n_actions),
    )


class Dqn(Agent):
    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        hparams: HParams,
        seed: int = 0,
        preprocess: Callable[[Observation], Observation] = lambda x: x,
    ):
        # public:
        self.action_spec = action_spec
        self.obs_spec = obs_spec
        self.epsilon = hparams.initial_exploration
        self.replay_buffer = OfflineBuffer(hparams.replay_memory_size)
        self.rng = jax.random.PRNGKey(seed)
        self.preprocess = preprocess
        network = Cnn(action_spec.num_values, hidden_size=hparams.hidden_size)
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
        _, params_target = self.network.init(self.rng, (-1, *obs_spec.shape))
        _, params_online = self.network.init(self.rng, (-1, *obs_spec.shape))
        self._opt_state = self.optimiser.init(params_online)
        self._params_target = params_target

    @pure
    def loss(
        params_target: Params,
        params_online: Params,
        transition: Transition,
    ) -> Loss:
        q_online = Dqn.network.apply(params_online, transition.x_0)
        q_target = Dqn.network.apply(params_target, transition.x_1)
        # get the q target
        target = td.nstep_return(transition, jnp.max(q_target, axis=-1))
        delta = target - q_online
        return jnp.sqrt(jnp.mean(jnp.square(delta)))

    @pure
    def sgd_step(
        iteration: int,
        params_target: Params,
        opt_state: OptimizerState,
        transition: Transition,
    ) -> Tuple[Loss, OptimizerState]:
        params_online = Dqn.optimiser.params(opt_state)
        backward = jax.value_and_grad(Dqn.loss, argnums=1)
        error, gradients = backward(
            Dqn.network, params_target, params_online, transition
        )
        return error, Dqn.optimiser.update(iteration, gradients, opt_state)

    def anneal_epsilon(self):
        x0, y0 = (self.hparams.replay_start, self.hparams.initial_exploration)
        x1 = self.hparams.final_exploration_frame
        y1 = self.hparams.final_exploration
        x = self._iteration
        y = ((y1 - y0) * (x - x0) / (x1 - x0)) + y0
        return y

    def policy(
        self,
        timestep: dm_env.TimeStep,
    ) -> int:
        """Selects an action using an e-greedy policy"""
        # use random policy with epsilon probability
        if jax.random.uniform(self.rng, (1,)) < self.epsilon:
            return jax.random.randint(self.rng, (1,), 0, self.action_spec.num_values)

        # otherwise, use greedy policy
        state = timestep.observation[None, ...]  # batching
        q_values = self.forward(self._online_params, state)
        action = jnp.argmax(q_values)
        print(action)
        return int(action)

    def observe(self, env: Environment, timestep: TimeStep, action: int) -> Transition:
        #  get new MDP state
        new_timestep = env.step(action)
        #  store transition into the replay buffer
        self.replay_buffer.add(
            timestep, action, new_timestep, preprocess=self.preprocess
        )
        return new_timestep

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> float:
        """Dqn uses a replay buffer. The three inputs
        `timestep`, `action` and `new_timestep` are ignored."""
        # increment iteration
        self._iteration += 1

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.replay_buffer) < self.hparams.replay_start:
            return None

        # the exploration parameter is linearly interpolated to the end value
        self.epsilon = self.anneal_epsilon()

        # update the online parameters
        transition = self.replay_buffer.sample(self.hparams.batch_size)
        loss, self._opt_state = self.sgd_step(
            self.network,
            self.optimiser,
            self._iteration,
            self._params_target,
            self._opt_state,
            transition,
        )

        # update the target network parameters every `target_network_update_frequency` steps
        if self._iteration % self.hparams.target_network_update_frequency == 0:
            params = self.optimiser.params(self._opt_state)
            self._params_target = self._params._replace(target=params)
        return loss

    def log(self, reward: float, loss: Loss = None):
        wandb.log({"Iteration": float(self._iteration)})
        wandb.log({"Buffer size": len(self.replay_buffer)})
        wandb.log({"Epsilon": self.epsilon})
        wandb.log({"Reward": reward})
        if loss is not None:
            wandb.log({"Loss": float(loss)})
        return
