# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import rlax
from jax.lax import stop_gradient

from .sac import SAC


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

    def policy(self, params_actor, observation, key):
        logits = self.actor.apply(params_actor, observation)
        logprobs = jax.nn.log_softmax(logits)
        action = jax.random.categorical(key, logits)  # type: ignore
        return action, logprobs

    @partial(jax.value_and_grad, argnums=1, has_aux=True)
    def loss(
        self,
        params,
        sarsd_batch,
        keys,
        params_critic_target,
    ):
        print(f"{self.__class__.__name__}.loss compiles")  # prints only if compiled
        s_0, _, r_1, s_1, d = sarsd_batch
        (params_actor, params_critic, params_temperature) = params

        _policy = jax.vmap(self.policy, in_axes=(None, 0, 0))  # type: ignore
        _value = jax.vmap(self.critic.apply, in_axes=(None, 0))  # type: ignore
        batch_matmul = lambda a, b: jnp.sum(a * b, axis=-1, keepdims=True)

        # current estimates
        temperature = self.temperature.apply(params_temperature)
        alpha = stop_gradient(temperature)
        _, logprobs_a_0 = _policy(params_actor, s_0, keys)
        _, logprobs_a_1 = _policy(params_actor, s_1, keys)
        probs_a_0 = jnp.exp(logprobs_a_0)
        probs_a_1 = jnp.exp(logprobs_a_1)
        qA_0, qB_0 = _value(params_critic, s_0)
        qA_1, qB_1 = _value(params_critic_target, s_1)

        # augment reward with policy entropy
        policy_entropy = -jnp.sum(probs_a_0 * logprobs_a_0, axis=-1)  # type: ignore
        # SACLite: https://arxiv.org/abs/2201.12434
        policy_entropy = policy_entropy * self.hparams.entropy_rewards
        r_1 = r_1 + alpha * policy_entropy

        # actor target: V(sₜ) = π(sₜ)ᵀ • [αlog(π(sₜ)) + Q(sₜ)]
        q_0 = stop_gradient(jnp.min(jnp.stack([qA_0, qB_0], axis=0), axis=0))
        actor_loss = batch_matmul(probs_a_0, alpha * logprobs_a_0 - q_0)  # type: ignore

        # critic target: V(sₜ₊₁) = π(sₜ₊₁)ᵀ • [Q(sₜ₊₁) - αlog(π(sₜ₊₁))]
        probs_a_1 = jnp.exp(logprobs_a_1)
        q_1 = jnp.min(jnp.stack([qA_1, qB_1], axis=0), axis=0)
        v_1 = batch_matmul(probs_a_1, (q_1 - alpha * logprobs_a_1))  # type: ignore
        q_target = stop_gradient(r_1 + (1 - d) * self.hparams.discount * v_1)
        critic_loss = rlax.l2_loss(qA_0, q_target) + rlax.l2_loss(qB_0, q_target)  # type: ignore

        # temperature target: π(sₜ)ᵀ • [- αlog(π(sₜ)) + H]
        target_entropy = -jnp.log(1 / self.dim_A)
        logprobs_a_0 = stop_gradient(logprobs_a_0)
        probs_a_0 = stop_gradient(probs_a_0)
        temperature_loss = -jnp.log(temperature) * (logprobs_a_0 + target_entropy)  # type: ignore
        temperature_loss = batch_matmul(probs_a_0, temperature_loss)

        # average over batches
        actor_loss = actor_loss.mean()  # type: ignore
        critic_loss = critic_loss.mean()
        temperature_loss = temperature_loss.mean()  # type: ignore
        policy_entropy = -jnp.sum(probs_a_0 * logprobs_a_0, axis=-1).mean()  # type: ignore
        loss = actor_loss + critic_loss + temperature_loss
        aux = (actor_loss, critic_loss, temperature_loss, policy_entropy, alpha)
        return loss, aux
