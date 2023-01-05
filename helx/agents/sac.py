# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import List, Tuple

import distrax
import jax
import jax.numpy as jnp
import rlax
import wandb
from chex import Array, PRNGKey, Shape
from flax import linen as nn
from jax.lax import stop_gradient
from optax import GradientTransformation

from ..mdp import Episode, Transition
from ..memory import ReplayBuffer
from ..networks import AgentNetwork, Temperature
from .agent import Agent, Hparams


class SACHparams(Hparams):
    # network
    input_shape: Shape
    dim_A: int
    tau: float = 0.005
    # rl
    replay_start: int = 1000
    replay_memory_size: int = 1000
    update_frequency: int = 1
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    n_steps: int = 1
    entropy_rewards: bool = False
    # sgd
    batch_size: int = 32
    learning_rate: float = 3e-4


class SAC(Agent[SACHparams]):
    """Implements Soft Actor-Critic with gaussian policy for continuous action spaces.
    Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep
        reinforcement learning with a stochastic actor."
        International conference on machine learning. PMLR, 2018.
        https://arxiv.org/abs/1801.01290 for the base SAC.
    Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications."
        arXiv preprint arXiv:1812.05905 (2018).
        https://arxiv.org/abs/1812.05905 for the tunable entropy temperature
    Haonanm Yu, et al. "Do you need reward entropy (in practice)?".
        arXiv preprint arXiv:2201.12434. 2022 Jan 28.
        https://arxiv.org/abs/2201.12434
    """

    def __init__(
        self,
        network: AgentNetwork,
        optimiser: GradientTransformation,
        hparams: SACHparams,
        seed: int,
    ):
        super().__init__(network, optimiser, hparams, seed)

        self.memory = ReplayBuffer(hparams.replay_memory_size)
        self.params_target: nn.FrozenDict = self.params.copy({})

    def policy(
        self,
        params: nn.FrozenDict,
        observation: Array,
        key: PRNGKey,
    ) -> Tuple[Array, Array]:
        """
        Returns a sampled action and the log probability of the action under the
        policy distribution.

        Args:
            params: actor parameters
            observation: the current observation to act on
            key: a jax random key

        Returns:
            action: the action and the log probability of the action
        """
        mu, logvar = self.network.actor(params, observation)
        noise = jax.random.normal(key, (self.hparams.dim_A,))
        action = jnp.tanh(mu + logvar * noise)
        logprob = distrax.Normal(mu, jnp.exp(logvar)).log_prob(action)  # type: ignore
        return action, logprob

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        keys: jax.random.KeyArray,
        params_critic_target: nn.FrozenDict,
    ):
        print("{} compiles".format(self.__class__.__name__))
        s_0, _, r_1, s_1, d = transition
        (params_actor, params_critic, params_temperature) = params

        # current estimates
        temperature = self.network.extra(params_temperature)
        alpha = stop_gradient(temperature)
        _, logprobs_a_0 = self.policy(params_actor, s_0, keys)
        _, logprobs_a_1 = self.policy(params_actor, s_1, keys)
        probs_a_0 = jnp.exp(logprobs_a_0)
        qA_0, qB_0 = self.network.critic(params_critic, s_0)
        qA_1, qB_1 = self.network.critic(params_critic_target, s_1)

        # augment reward with policy entropy
        policy_entropy = -jnp.sum(probs_a_0 * logprobs_a_0, axis=-1)  # type: ignore
        # SACLite: https://arxiv.org/abs/2201.12434
        policy_entropy = policy_entropy * self.hparams.entropy_rewards
        r_1 = r_1 + alpha * policy_entropy

        # actor loss
        q_0 = stop_gradient(jnp.min(jnp.stack([qA_0, qB_0], axis=0), axis=0))
        actor_loss = alpha * logprobs_a_0 - q_0

        # critic losses
        q_1 = jnp.min(jnp.stack([qA_1, qB_1], axis=0), axis=0)
        v_1 = q_1 - alpha * logprobs_a_1
        td_target = stop_gradient(r_1 + (1 - d) * self.hparams.discount * v_1)
        critic_loss = rlax.l2_loss(qA_0, td_target) + rlax.l2_loss(qB_0, td_target)  # type: ignore

        # temperature loss
        target_entropy = -self.hparams.dim_A
        logprobs_a_0 = stop_gradient(logprobs_a_0)
        temperature_loss = -(temperature * (logprobs_a_0 + target_entropy))

        # average over batches
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        temperature_loss = temperature_loss.mean()
        policy_entropy = policy_entropy.mean()
        loss = actor_loss + critic_loss + temperature_loss
        aux = (actor_loss, critic_loss, temperature_loss, policy_entropy, alpha)
        return loss, aux

    def update(self, episode: Episode) -> Array:
        # update iteration
        self.iteration += 1
        wandb.log({"Iteration": self.iteration})

        # update memory
        transitions: List[Transition] = episode.transitions()
        self.memory.add_range(transitions)  # makes self.loss recompile each time, why??
        wandb.log({"Buffer size": len(self.memory)})

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.memory) < self.hparams.replay_start:
            return jnp.asarray([])

        # check update period
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
        self.params_critic_target = jax.tree_map(
            lambda theta, theta_: theta * self.hparams.tau
            + (1 - self.hparams.tau) * theta_,
            self.params,
            self.params_target,
        )

        actor_loss, critic_loss, temperature_loss, policy_entropy, alpha = aux
        wandb.log({"train/total_loss": loss.item()})
        wandb.log({"train/actor_loss": actor_loss.item()})
        wandb.log({"train/critic_loss": critic_loss.item()})
        wandb.log({"train/temperature_loss": temperature_loss.item()})
        wandb.log({"train/policy_entropy": policy_entropy.item()})
        wandb.log({"train/alpha": alpha.item()})
        wandb.log({"train/Return": jnp.sum(episode.r).item()})  # type: ignore
        return loss
