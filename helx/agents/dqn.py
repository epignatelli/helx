from __future__ import annotations

from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
import optax
import rlax
from chex import Array
from flax import linen as nn
from jax.random import KeyArray
from optax import GradientTransformation

import wandb

from ..agents.base import IAgent, IDeep, IStateless, IValueBased, SGDState
from ..mdp import Action, Episode, Transition
from ..memory import ReplayBuffer
from ..networks import AgentNetwork, EGreedyPolicy
from ..networks.modules import Identity
from ..spaces import Discrete
from .agent import Agent, Hparams


class DQNHparams(Hparams):
    # network
    hidden_size: int = 128
    # rl
    initial_exploration: float = 1.0
    final_exploration: float = 0.01
    final_exploration_frame: int = 1000000
    replay_start: int = 1000
    replay_memory_size: int = 1000
    update_frequency: int = 1
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    n_steps: int = 1
    # sgd
    batch_size: int = 32
    learning_rate: float = 0.00025
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.0


class DQNState(SGDState):
    """The state of a DQN agent."""

    params_target: nn.FrozenDict
    memory: ReplayBuffer


# Example implementation
class DQN(IDeep, IAgent):
    """A Deep Q-Network agent.
    This agent uses a neural network to approximate the Q-function.
    """

    def __init__(
        self,
        hparams: DQNHparams,
        representation_net: nn.Module,
        optimiser: GradientTransformation,
        seed: int = 0,
        **kwargs,
    ):
        """Initialises the agent.
        Args:
            network (AgentNetwork): The network to use for the agent.
            optimiser (GradientTransformation): The optimiser to use for the agent.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            key (KeyArray, optional): The JAX PRNG key. Defaults to jax.random.PRNGKey(0).
        """
        assert isinstance(hparams.action_space, Discrete)

        # static:
        self.hparams = hparams
        self.network = nn.Sequential(
            [representation_net, nn.Dense(hparams.action_space.n_bins)]
        )
        self.policy = EGreedyPolicy(
            hparams.replay_start,
            hparams.initial_exploration,
            hparams.final_exploration,
            hparams.final_exploration_frame,
        )
        self.optimiser = optimiser
        self.sgd_step = jax.jit(self.sgd_step)

        # dynamic:
        key = jax.random.PRNGKey(seed)
        params = self.network.init(key, hparams.obs_space.sample(key))
        opt_state = self.optimiser.init(self.params)
        self.state = DQNState(
            step=0,
            key=key,
            params=params,
            opt_state=opt_state,
            params_target=params.copy({}),
            memory=ReplayBuffer(hparams.replay_memory_size),
        )
    def sample_action(
        self, observation: Array, key: KeyArray, eval: bool = False, **kwargs
    ) -> Action:
        values = self.critic(observation)
        return self.policy(values, key)  # type: ignore

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        params_target: nn.FrozenDict,
    ) -> Tuple[Array, Any]:
        s_0, a_0, r_1, s_1, d = transition
        q_0 = self.network.critic(params, s_0)
        q_1 = self.network.critic(params_target, s_1)

        # ignoring type because flax modules can return anything
        # we should either strongly type the network or cast the output
        q_1 = jax.lax.stop_gradient(jnp.max(q_1))  # type: ignore
        td_error = r_1 + (1 - d) * self.hparams.discount * q_1 - q_0[a_0]  # type: ignore
        loss = jnp.mean(rlax.l2_loss(td_error))
        return loss, ()

    def sgd_step(
        self,
        state: SGDState,
        transition: Transition,
        params_target: nn.FrozenDict,
    ) -> Tuple[SGDState, Array, Any]:
        def _batched_loss(
            params: nn.FrozenDict,
            batched_transitions: Transition,
            *args,
        ) -> Tuple[Array, Any]:
            # in_axis for named arguments is not supported yet by jax.vmap
            # see https://github.com/google/jax/issues/7465
            in_axes = (None, 0) + (None,) * len(args)
            fun = jax.vmap(self.loss, in_axes=in_axes)
            loss, aux = fun(params, batched_transitions, *args)
            return loss.mean(), aux

        params = state.params
        grads, (loss, aux) = jax.value_and_grad(_batched_loss, has_aux=True)(
            params, transition, params_target
        )
        updates, opt_state = self.optimiser.update(grads, state.opt_state, params)
        params = optax.apply_updates(params, updates)

        new_state = SGDState(
            step=state.step + 1,
            key=jax.random.split(state.key, 2)[1],
            params=params,  # type: ignore
            opt_state=opt_state,
        )
        return new_state, loss, aux

    def update(self, episode: Episode, state: DQNState) -> Array:
        wandb.log({"Iteration": self.state.step})

        # update memory
        transitions: List[Transition] = episode.transitions()
        self.state.memory.add_range(transitions)
        wandb.log({"Buffer size": len(self.state.memory)})

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.state.memory) < self.hparams.replay_start:
            return jnp.asarray([])

        # update every `update_frequency` steps
        if self.state.step % self.hparams.update_frequency != 0:
            return jnp.asarray([])

        episode_batch: Transition = self.state.memory.sample(self.hparams.batch_size)

        sgd_state, loss, _ = self.sgd_step(
            self.state,
            episode_batch,
            self.params_target,
        )

        # update dqn state
        params_target = rlax.periodic_update(
            self.state.params,
            self.params_target,
            jnp.asarray(self.iteration),
            self.hparams.target_network_update_frequency,
        )
        self.state = self.state.replace(
            step=sgd_state.step,
            key=sgd_state.key,
            params=sgd_state.params,
            opt_state=sgd_state.opt_state,
            params_target=self.params_target,
            memory=self.state.memory,
        )

        wandb.log({"train/total_loss": loss.item()})
        wandb.log({"train/Return": jnp.sum(episode.r).item()})  # type: ignore
        return loss
