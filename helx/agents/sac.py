# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
import rlax
import wandb
from chex import Array, PRNGKey, Shape
from flax import linen as nn
from jax.lax import stop_gradient
from optax import GradientTransformation

from ..mdp import Episode
from ..memory import ReplayBuffer
from .agent import Agent, Hparams


class SAChparams(Hparams):
    # network
    input_shape: Shape
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


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> Array:
        log_temperature = self.param(
            "log_temperature",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temperature)


class SAC(Agent):
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
        actor: nn.Module,
        critic: nn.Module,
        optimiser: GradientTransformation,
        hparams: SAChparams,
        seed: int,
    ):
        key = jax.random.PRNGKey(seed)
        key, k1, k2, k3 = jax.random.split(key, 4)
        temperature = Temperature()
        input_tensor = jnp.ones(hparams.input_shape)
        output_tensor, params_actor = actor.init_with_output(k1, input_tensor)
        params_critic = critic.init(k2, input_tensor)
        params_temperature = temperature.init(k3)

        self.key = key
        self.actor = actor
        self.critic = critic
        self.optimiser = optimiser
        self.temperature = temperature
        self.hparams: SAChparams = hparams
        self.memory = ReplayBuffer(hparams.replay_memory_size)
        self.iteration = 0
        self.dim_A = jnp.ndim(output_tensor)
        self.params = (params_actor, params_critic, params_temperature)
        self.opt_state = optimiser.init(self.params)
        self.params_critic_target = params_critic.copy({})
        super().__init__()

    def sample_action(
        self, observation: Array, eval: bool = False
    ) -> Tuple[Array, Array]:
        """Selects an action using a parameterise gaussian policy"""
        (params_actor, _, _) = self.params
        return self.policy(params_actor, observation, self.new_key())  # type: ignore

    def policy(
        self,
        params: nn.FrozenDict,
        observation: Array,
        key: PRNGKey,
        eval: bool = False,
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
        mu, logvar = self.actor.apply(params, observation)
        noise = jax.random.normal(key, (self.dim_A,))
        action = jnp.tanh(mu + logvar * noise)
        logprob = distrax.Normal(mu, jnp.exp(logvar)).log_prob(action)  # type: ignore
        return action, logprob

    def loss(self, params, transitions_batch, keys, params_critic_target):
        print("{} compiles".format(self.__class__.__name__))
        s_0, _, r_1, s_1, d = transitions_batch
        (params_actor, params_critic, params_temperature) = params

        _policy = jax.vmap(self.policy(p, o, k), in_axes=(None, 0, 0))  # type: ignore
        _value = jax.vmap(self.critic.apply, in_axes=(None, 0))  # type: ignore

        # current estimates
        temperature = self.temperature.apply(params_temperature)
        alpha = stop_gradient(temperature)
        _, logprobs_a_0 = _policy(params_actor, s_0, keys)
        _, logprobs_a_1 = _policy(params_actor, s_1, keys)
        probs_a_0 = jnp.exp(logprobs_a_0)
        qA_0, qB_0 = _value(params_critic, s_0)
        qA_1, qB_1 = _value(params_critic_target, s_1)

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
        target_entropy = -self.dim_A
        logprobs_a_0 = stop_gradient(logprobs_a_0)
        temperature_loss = -(temperature * (logprobs_a_0 + target_entropy))

        # average over batches
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        temperature_loss = temperature_loss.mean()
        policy_entropy = policy_entropy.mean()
        loss = actor_loss + critic_loss + temperature_loss
        aux = (actor_loss, critic_loss, temperature_loss, policy_entropy, alpha)
        return jnp.array(loss, dtype=float), aux

    def update(self, episode: Episode) -> Array:
        # update iteration
        self.iteration += 1
        wandb.log({"Iteration": self.iteration})

        # update memory
        transitions = episode.sars()
        self.memory.add_range(transitions)  # makes self.loss recompile each time, why??
        wandb.log({"Buffer size": len(self.memory)})

        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.memory) < self.hparams.replay_start:
            return jnp.asarray([])

        # check update period
        if self.iteration % self.hparams.update_frequency != 0:
            return jnp.asarray([])

        episode_batch = self.memory.sample(self.hparams.batch_size)
        keys = self.new_key(self.hparams.batch_size)
        (loss, aux), grads = self.loss(
            self.params,
            episode_batch,
            keys,
            self.params_critic_target,
        )
        updates, opt_state = self.optimiser.update(grads, self.opt_state)
        actor_loss, critic_loss, temperature_loss, policy_entropy, alpha = aux

        # update dqn state
        self.opt_state = opt_state
        self.params = optax.apply_updates(self.params, updates)
        self.params_critic_target = jax.tree_map(
            lambda theta, theta_: theta * self.hparams.tau
            + (1 - self.hparams.tau) * theta_,
            self.params[1],  # type: ignore
            self.params_critic_target,
        )

        wandb.log({"train/total_loss": loss.item()})
        wandb.log({"train/actor_loss": actor_loss.item()})
        wandb.log({"train/critic_loss": critic_loss.item()})
        wandb.log({"train/temperature_loss": temperature_loss.item()})
        wandb.log({"train/policy_entropy": policy_entropy.item()})
        wandb.log({"train/alpha": alpha.item()})
        wandb.log({"train/Return": jnp.sum(episode.r).item()})  # type: ignore
        return loss
