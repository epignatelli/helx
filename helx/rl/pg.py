from helx.typing import Array
import jax
import jax.numpy as jnp


def importance_sampling(logits_mu: Array, logits_pi: Array):
    ...


def softmax_entropy(logits):
    return -jnp.sum(jnp.mean(logits) * jnp.log(logits))


def a2c_softmax_loss(logits, advantages):
    return jax.nn.log_softmax(logits) * advantages


def trpo_softmax_loss(logits, advantages, hparams):
    raise NotImplementedError


def ppo_softmax_loss(logits, logits_old, advantages, hparams):
    r_t = logits / logits_old
    r_t_clipped = jnp.clip(r_t, 1 - hparams.epsilon, 1 + hparams.epsilon)
    return min(r_t, r_t_clipped) * advantages
