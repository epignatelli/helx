# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import rlax
from jax.lax import stop_gradient
from optax import GradientTransformation
import wandb

from helx.spaces import Discrete

from ..mdp import Transition
from .sac import SAC, SACHparams


class SACD(SAC):
    """Soft Actor-Critic agent with softmax policy for discrete action spaces
    and a twinned Q-network for value approximation with a separate target network
    updated by polyak averaging the online network parameters.
    See:
        1. Haarnoja T., 2018 (https://arxiv.org/abs/1801.01290)
            for the base SAC
        2. Haarnoja T., 2019 (https://arxiv.org/abs/1812.05905)
            for the tunable entropy temperature and the ablation of the value network
        3. Christodoulou P., 2019 (https://arxiv.org/abs/1910.07207)
            for the adaptation to discrete action spaces
    """

    def __init__(
        self,
        hparams: SACHparams,
        optimiser: GradientTransformation,
        seed: int,
        actor_representation_net: nn.Module,
        critic_representation_net: nn.Module,
    ):
        assert isinstance(hparams.action_space, Discrete)
        super().__init__(
            hparams,
            optimiser,
            seed,
            actor_representation_net,
            critic_representation_net,
        )

    def loss(
        self,
        params: nn.FrozenDict,
        transition: Transition,
        keys: jax.random.KeyArray,
        params_critic_target,
    ):
        s_0, _, r_1, s_1, d = transition
        batch_matmul = lambda a, b: jnp.sum(a * b, axis=-1, keepdims=True)

        # current estimates
        temperature = self.network.extra(params)
        alpha = stop_gradient(temperature)
        _, logprobs_a_0 = self.network.actor(params, s_0, keys)
        _, logprobs_a_1 = self.network.actor(params, s_1, keys)
        probs_a_0 = jnp.exp(logprobs_a_0)
        # critic is a double Q network
        qA_0, qB_0 = self.network.critic(params, s_0)  # type: ignore
        qA_1, qB_1 = self.network.critic(params_critic_target, s_1)  # type: ignore

        # augment reward with policy entropy
        policy_entropy = -jnp.sum(probs_a_0 * logprobs_a_0, axis=-1)
        # SACLite: https://arxiv.org/abs/2201.12434
        policy_entropy = policy_entropy * self.hparams.entropy_rewards
        r_1 = r_1 + alpha * policy_entropy

        # actor target: V(sₜ) = π(sₜ)ᵀ • [αlog(π(sₜ)) + Q(sₜ)]
        q_0 = stop_gradient(jnp.min(jnp.stack([qA_0, qB_0], axis=0), axis=0))
        actor_loss = batch_matmul(probs_a_0, alpha * logprobs_a_0 - q_0)

        # critic target: V(sₜ₊₁) = π(sₜ₊₁)ᵀ • [Q(sₜ₊₁) - αlog(π(sₜ₊₁))]
        probs_a_1 = jnp.exp(logprobs_a_1)
        q_1 = jnp.min(jnp.stack([qA_1, qB_1], axis=0), axis=0)
        # compared to SAC, which takes the expectation over the action
        # because it doesn't have the probs over all possible actions
        # here we can, so calculate the exact value of the state
        v_1 = batch_matmul(probs_a_1, (q_1 - alpha * logprobs_a_1))
        q_target = stop_gradient(r_1 + (1 - d) * self.hparams.discount * v_1)
        critic_loss = rlax.l2_loss(qA_0, q_target) + rlax.l2_loss(qB_0, q_target)

        # temperature target: π(sₜ)ᵀ • [- αlog(π(sₜ)) + H]
        target_entropy = -self.hparams.action_space.shape[0]
        logprobs_a_0 = stop_gradient(logprobs_a_0)
        probs_a_0 = stop_gradient(probs_a_0)
        temperature_loss = -(temperature * logprobs_a_0 + target_entropy)
        temperature_loss = batch_matmul(probs_a_0, temperature_loss)

        loss = actor_loss + critic_loss + temperature_loss
        aux = (actor_loss, critic_loss, temperature_loss, policy_entropy, alpha)
        return loss, aux
