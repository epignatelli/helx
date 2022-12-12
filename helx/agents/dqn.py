# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
import rlax
import wandb
from chex import Array, Shape
from flax import linen as nn
from optax import GradientTransformation

from .agent import Agent
from ..mdp import Episode
from ..memory import ReplayBuffer


class DqnHParams(NamedTuple):
    # network
    input_shape: Shape
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
    min_squared_gradient: float = 0.01


class DQN(Agent):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    def __init__(
        self,
        network: nn.Module,
        optimiser: GradientTransformation,
        hparams: DqnHParams,
        seed: int,
    ):
        key = jax.random.PRNGKey(seed)
        key, k1 = jax.random.split(key)
        input_shape = hparams.input_shape
        params = network.init(k1, jnp.ones(input_shape))

        # const:
        self.key = key
        self.network = network
        self.optimiser = optimiser
        self.hparams = hparams
        self.memory = ReplayBuffer(hparams.replay_memory_size)
        self.iteration = 0
        self.params = params
        self.params_target = params.copy({})
        self.opt_state = optimiser.init(params)

    def epsilon(self, eval=False) -> Array:
        x0, y0 = (self.hparams.replay_start, self.hparams.initial_exploration)
        x1 = self.hparams.final_exploration_frame
        y1 = self.hparams.final_exploration
        x = self.iteration
        y = ((y1 - y0) * (x - x0) / (x1 - x0)) + y0
        eps = jnp.clip(
            y, self.hparams.final_exploration, self.hparams.initial_exploration
        )
        return jax.lax.select(eval, 0.0, eps)

    def policy(self, observation: Array, eval=False) -> Tuple[Array, Array]:
        """Selects an action using an e-greedy policy"""
        q_values = self.network.apply(self.params, observation)  # type: ignore
        action, log_probs = distrax.EpsilonGreedy(
            q_values, self.epsilon(eval)  # type: ignore
        ).sample_and_log_prob(seed=self.new_key())
        return action, log_probs

    @partial(jax.value_and_grad, argnums=1, has_aux=True)
    def loss(self, params, transitions_batch, params_target):
        Q = jax.vmap(self.network.apply, in_axes=(None, 0))  # type: ignore
        s_0, a_0, r_1, s_1, d = transitions_batch
        q_1 = jax.lax.stop_gradient(jnp.max(Q(params_target, s_1), axis=-1))  # type: ignore
        q_0 = Q(params, s_0)
        td_error = r_1 + (1 - d) * self.hparams.discount * jnp.max(q_1) - q_0[a_0]
        loss = jnp.mean(rlax.l2_loss(td_error))
        return loss, ()

    def update(self, episode: Episode) -> Array:
        # increment iteration
        self.iteration += 1
        wandb.log({"Iteration": self.iteration})

        # update memory
        transitions = episode.sars()
        self.memory.add_range(transitions)
        wandb.log({"Buffer size": len(self.memory)})

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.memory) < self.hparams.replay_start:
            return jnp.asarray([])

        # update every `update_frequency` steps
        if self.iteration % self.hparams.update_frequency != 0:
            return jnp.asarray([])

        episode_batch = self.memory.sample(self.hparams.batch_size)
        (loss, _), grads = self.loss(
            self.params,
            episode_batch,
            self.params_target,
        )
        updates, opt_state = self.optimiser.update(grads, self.opt_state)

        # update dqn state
        self.opt_state = opt_state
        self.params = optax.apply_updates(self.params, updates)
        self.params_target = rlax.periodic_update(
            self.params,
            self.params_target,
            jnp.asarray(self.iteration),
            self.hparams.target_network_update_frequency,
        )

        wandb.log({"train/total_loss": loss.item()})
        wandb.log({"train/Return": jnp.sum(episode.r).item()})  # type: ignore
        return loss
