# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Common missing modules to the jax.stax library"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal, zeros


def Rnn(cell):
    """Layer construction function for an RNN unroll.
    Implements truncated backpropagation through time for the backward pass
    """

    def init(rng, input_shape):
        return cell.init(rng, input_shape)

    def apply(params, inputs, **kwargs):
        prev_state = kwargs.pop("prev_state", None)
        if prev_state is None:
            msg = (
                "Recurrent layers require apply_fun to be called with a prev_state "
                "argument. That is, instead of `apply_fun(params, inputs)`, "
                "call it like apply_fun(params, inputs, prev_state=prev_state)` "
                "where `prev_state` is the rnn hidden state."
            )
            raise ValueError(msg)
        prev_state, outputs = jax.lax.scan(
            lambda prev_state, inputs: cell.apply(
                params, inputs, prev_state=prev_state
            )[::-1],
            prev_state,  # None will be handled by the respective cell's apply_fun
            inputs,
        )
        return outputs, prev_state

    return (init, apply)


class LSTMState(NamedTuple):
    h: jnp.ndarray
    c: jnp.ndarray


def LSTMCell(
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    h_initial_state_fn=zeros,
    c_initial_state_fn=zeros,
):
    """Layer construction function for an LSTM cell.
    Formulation: Zaremba, W., 2015, https://arxiv.org/pdf/1409.2329.pdf"""

    def initial_state(rng):
        shape = (hidden_size,)
        k1, k2 = jax.random.split(rng)
        return LSTMState(h_initial_state_fn(k1, shape), c_initial_state_fn(k2, shape))

    def init(rng, input_shape):
        #  init params
        in_dim, out_dim = input_shape[-1] + hidden_size, 4 * hidden_size
        output_shape = input_shape[:-1] + (hidden_size,)
        k1, k2, k3 = jax.random.split(rng, 3)
        W, b = W_init(k1, (in_dim, out_dim)), b_init(k2, (out_dim,))
        hidden_state = initial_state(k3)
        return (output_shape, (output_shape, output_shape)), ((W, b), hidden_state)

    def apply(params, inputs, **kwargs):
        prev_state = kwargs.get("prev_state")  # type: ignore
        W, b = params
        xh = jnp.concatenate([inputs, prev_state.h], axis=-1)  # type: ignore
        gated = jnp.matmul(xh, W) + b
        i, f, o, g = jnp.split(gated, indices_or_sections=4, axis=-1)
        c = sigmoid(f) * prev_state.c + sigmoid(i) * jnp.tanh(g)  # type: ignore
        h = sigmoid(o) * jnp.tanh(c)
        return h, LSTMState(h, c)

    return (init, apply)


def LSTM(
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    h_initial_state=zeros,
    c_initial_state=zeros,
):
    return Rnn(
        LSTMCell(
            hidden_size,
            W_init,
            b_init,
            h_initial_state,
            c_initial_state,
        )
    )


def GRUCell(
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    initial_state_fn=zeros,
):
    """Layer construction function for an GRU cell.
    Formulation: Chung, J., 2014, https://arxiv.org/pdf/1412.3555v1.pdf"""

    def initial_state(rng):
        return initial_state_fn(rng, (hidden_size,))

    def init(rng, input_shape):
        in_dim, out_dim = input_shape[-1] + hidden_size, 3 * hidden_size
        output_shape = input_shape[:-1] + (hidden_size,)
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        W_i = W_init(k1, (in_dim, out_dim))
        W_h = W_init(k2, (in_dim, out_dim))
        b = b_init(k3, (out_dim,))
        hidden_state = initial_state(k4)
        return output_shape, ((W_i, W_h, b), hidden_state)

    def apply(params, inputs, **kwargs):
        prev_state = kwargs.get("prev_state")
        W_i, W_h, b = params
        W_hz, W_ha = jnp.split(W_h, indices_or_sections=(2 * hidden_size,), axis=-1)  # type: ignore
        b_z, b_a = jnp.split(b, indices_or_sections=(2 * hidden_size,), axis=-1)  # type: ignore

        gated = jnp.matmul(inputs, W_i)
        zr_x, a_x = jnp.split(gated, indices_or_sections=[2 * hidden_size], axis=-1)  # type: ignore
        zr_h = jnp.matmul(prev_state, W_hz)
        z, r = jnp.split(
            jax.nn.sigmoid(zr_x + zr_h + b_z), indices_or_sections=2, axis=-1
        )
        a_h = jnp.matmul(r * prev_state, W_ha)
        a = jnp.tanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))
        h = (1 - z) * prev_state + z * a
        return h, h

    return (init, apply)


def GRU(
    hidden_size,
    W_init=glorot_normal(),
    b_init=normal(),
    initial_state_fn=zeros,
):
    return Rnn(GRUCell(hidden_size, W_init, b_init, initial_state_fn))
