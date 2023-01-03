# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from functools import partial
from typing import Any, List, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
import rlax
from chex import Array, Shape
from flax import linen as nn
from optax import GradientTransformation

import wandb

from ..mdp import Episode, Transition
from ..memory import ReplayBuffer
from .agent import Agent, Hparams


class DQNhparams(Hparams):
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


class DQN(Agent[DQNhparams]):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    def __init__(
        self,
        network: nn.Module,
        optimiser: GradientTransformation,
        hparams: DQNhparams,
        seed: int,
    ):
        super().__init__(network, optimiser, hparams, seed)
        self.memory = ReplayBuffer[Transition](hparams.replay_memory_size)
        self.params_target = self.params.copy({})

    def sample_action(self, observation: Array, eval: bool = False) -> Array:
        return self.policy(self.params, observation, eval)[0]

    def policy(
        self, params: nn.FrozenDict, observation: Array, eval=False
    ) -> Tuple[Array, Array]:
        """Selects an action using an e-greedy policy

        Args:
            observation (Array): the current observation including the batch dimension
            eval (bool, optional): whether to use the evaluation policy. Defaults to False.
        Returns:
            Tuple[Array, Array]: the action and the log probability of the action"""
        q_values = jnp.asarray(self.network.apply(params, observation))
        distr = distrax.EpsilonGreedy(q_values, self.epsilon(eval).item())
        action, log_probs = distr.sample_and_log_prob(seed=self.new_key())
        return action, log_probs

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        params_target: nn.FrozenDict,
    ) -> Tuple[Array, Any]:
        s_0, a_0, r_1, s_1, d = transition
        q_0 = self.network.apply(params, s_0)
        q_1 = self.network.apply(params_target, s_1)

        q_1 = jax.lax.stop_gradient(q_1)
        td_error = r_1 + (1 - d) * self.hparams.discount * jnp.max(q_1) - q_0[a_0]
        loss = jnp.mean(rlax.l2_loss(td_error))
        return loss, ()

    def update(self, episode: Episode) -> Array:
        # increment iteration
        self.iteration += 1
        wandb.log({"Iteration": self.iteration})

        # update memory
        transitions: List[Transition] = episode.transitions()
        self.memory.add_range(transitions)
        wandb.log({"Buffer size": len(self.memory)})

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.memory) < self.hparams.replay_start:
            return jnp.asarray([])

        # update every `update_frequency` steps
        if self.iteration % self.hparams.update_frequency != 0:
            return jnp.asarray([])

        episode_batch: Transition = self.memory.sample(self.hparams.batch_size)

        params, opt_state, loss, aux = self.sgd_step(
            self.params,
            episode_batch,
            self.opt_state,
            self.params_target,
        )

        # update dqn state
        self.opt_state = opt_state
        self.params = params
        self.params_target = rlax.periodic_update(
            self.params,
            self.params_target,
            jnp.asarray(self.iteration),
            self.hparams.target_network_update_frequency,
        )

        wandb.log({"train/total_loss": loss.item()})
        wandb.log({"train/Return": jnp.sum(episode.r).item()})  # type: ignore
        return loss

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
