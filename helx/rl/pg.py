import jax
import jax.numpy as jnp


def softmax_entropy(logits):
    return -jnp.sum(jnp.mean(logits) * jnp.log(logits))


def a2c_softmax_loss(logits, advantages):
    return jax.nn.log_softmax(logits) * advantages
