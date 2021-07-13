# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common networks."""
import collections
from typing import Any, Tuple

import dm_env
import haiku as hk
import jax.nn
import jax.numpy as jnp
from impala import sr as sr_models_lib
from jax._src.lax.lax import Array

NetOutput = collections.namedtuple("NetOutput", ["policy_logits", "value"])


class CatchNet(hk.RNNCore):
    """A simple neural network for catch."""

    def __init__(self, num_actions, name=None):
        super().__init__(name=name)
        self._num_actions = num_actions

    def initial_state(self, batch_size):
        if batch_size is None:
            shape = []
        else:
            shape = [batch_size]
        return jnp.zeros(shape)  # Dummy.

    def __call__(self, x: dm_env.TimeStep, state):
        torso_net = hk.Sequential(
            [hk.Flatten(), hk.Linear(128), jax.nn.relu, hk.Linear(64), jax.nn.relu]
        )
        torso_output = torso_net(x.observation)
        policy_logits = hk.Linear(self._num_actions)(torso_output)
        value = hk.Linear(1)(torso_output)
        value = jnp.squeeze(value, axis=-1)
        return NetOutput(policy_logits=policy_logits, value=value), state

    def unroll(self, x, state):
        """Unrolls more efficiently than dynamic_unroll."""
        out, _ = hk.BatchApply(self)(x, None)
        return out, state


class AtariShallowTorso(hk.Module):
    """Shallow torso for Atari, from the DQN paper."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        torso_net = hk.Sequential(
            [
                lambda x: x / 255.0,
                hk.Conv2D(32, kernel_shape=[8, 8], stride=[4, 4], padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(64, kernel_shape=[4, 4], stride=[2, 2], padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(64, kernel_shape=[3, 3], stride=[1, 1], padding="VALID"),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(512),
                jax.nn.relu,
            ]
        )
        return torso_net(x)


class ResidualBlock(hk.Module):
    """Residual block."""

    def __init__(self, num_channels, name=None):
        super().__init__(name=name)
        self._num_channels = num_channels

    def __call__(self, x):
        main_branch = hk.Sequential(
            [
                jax.nn.relu,
                hk.Conv2D(
                    self._num_channels,
                    kernel_shape=[3, 3],
                    stride=[1, 1],
                    padding="SAME",
                ),
                jax.nn.relu,
                hk.Conv2D(
                    self._num_channels,
                    kernel_shape=[3, 3],
                    stride=[1, 1],
                    padding="SAME",
                ),
            ]
        )
        return main_branch(x) + x


class AtariDeepTorso(hk.Module):
    """Deep torso for Atari, from the IMPALA paper."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        torso_out = x / 255.0
        for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
            conv = hk.Conv2D(
                num_channels, kernel_shape=[3, 3], stride=[1, 1], padding="SAME"
            )
            torso_out = conv(torso_out)
            torso_out = hk.max_pool(
                torso_out,
                window_shape=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding="SAME",
            )
            for j in range(num_blocks):
                block = ResidualBlock(num_channels, name="residual_{}_{}".format(i, j))
                torso_out = block(torso_out)

        torso_out = jax.nn.relu(torso_out)
        torso_out = hk.Flatten()(torso_out)
        torso_out = hk.Linear(256)(torso_out)
        torso_out = jax.nn.relu(torso_out)
        return torso_out


class AtariNet(hk.RNNCore):
    """Network for Atari."""

    def __init__(self, num_actions, use_resnet, use_lstm, name=None):
        super().__init__(name=name)
        self._num_actions = num_actions
        self._use_resnet = use_resnet
        self._use_lstm = use_lstm
        self._core = hk.ResetCore(hk.LSTM(256))

    def initial_state(self, batch_size):
        return self._core.initial_state(batch_size)

    def __call__(self, x: dm_env.TimeStep, state):
        x = jax.tree_map(lambda t: t[None, ...], x)
        return self.unroll(x, state)

    def unroll(self, x, state):
        """Unrolls more efficiently than dynamic_unroll."""
        if self._use_resnet:
            torso = AtariDeepTorso()
        else:
            torso = AtariShallowTorso()

        torso_output = hk.BatchApply(torso)(x.observation)
        if self._use_lstm:
            should_reset = jnp.equal(x.step_type, int(dm_env.StepType.FIRST))
            core_input = (torso_output, should_reset)
            core_output, state = hk.dynamic_unroll(self._core, core_input, state)
        else:
            core_output = torso_output
            # state passes through.

        return hk.BatchApply(self._head)(core_output), state

    def _head(self, core_output):
        policy_logits = hk.Linear(self._num_actions)(core_output)
        value = hk.Linear(1)(core_output)
        value = jnp.squeeze(value, axis=-1)
        return NetOutput(policy_logits=policy_logits, value=value)


class CatchConvNet(hk.RNNCore):
    def initial_state(self, batch_size):
        return ()

    def __call__(self, x: dm_env.TimeStep, state):
        return (
            hk.Sequential(
                [
                    hk.Conv2D(32, (2, 2), (1, 1)),
                    jax.nn.relu,
                    hk.Conv2D(64, (2, 2), (1, 1)),
                    jax.nn.relu,
                    hk.Flatten(),
                    hk.Linear(256),
                    jax.nn.relu,
                ]
            )(x.observation),
            state,
        )


class KeyToDoorConvNet(CatchConvNet):
    ...


class SyntheticReturns(hk.RNNCore):
    """A simple neural network for catch."""

    def __init__(self, name=None, **sr_config):
        super().__init__(name=name)
        hidden_units = 256
        core = hk.LSTM(hidden_units)
        self._sr_wrapper = hk.ResetCore(
            sr_models_lib.SyntheticReturnsCoreWrapper(core, **sr_config)
        )

    def initial_state(self, batch_size):
        return self._sr_wrapper.initial_state(batch_size)

    def __call__(self, inputs, state) -> sr_models_lib.SrOutput:
        features, timestep = inputs

        return_targets = timestep.reward
        should_reset = jnp.equal(timestep.step_type, int(dm_env.StepType.FIRST))

        core_inputs = ((features, return_targets), should_reset)
        # sr_state = jax.tree_map(lambda t: t[0], sr_state)
        output, state = hk.dynamic_unroll(self._sr_wrapper, core_inputs, state)
        return output, state


class PolicyNetwork(hk.RNNCore):
    def __init__(self, num_actions, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions

    def initial_state(self, batch_size):
        return ()

    def __call__(self, x: Array, state):
        hidden_units = 256
        features = hk.Sequential([hk.Linear(hidden_units), jax.nn.relu])(x)
        logits = hk.Linear(self.num_actions)(features)
        value = jnp.squeeze(hk.Linear(1)(features), axis=-1)
        return NetOutput(logits, value), state


class _SR(hk.RNNCore):
    def __init__(
        self,
        num_actions,
        vision_net_fn,
        name=None,
        **sr_config,
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.conv_net = vision_net_fn()
        self.sr_net = SyntheticReturns(name, **sr_config)
        self.policy_net = PolicyNetwork(num_actions)

    def initial_state(self, batch_size):
        return (
            self.conv_net.initial_state(batch_size),
            self.sr_net.initial_state(batch_size),
            self.policy_net.initial_state(batch_size),
        )

    def unroll(
        self, timestep: dm_env.TimeStep, state: Any
    ) -> Tuple[sr_models_lib.SrOutput, Any]:
        c_state, sr_state, p_state = state

        #  features net
        features, c_state = hk.BatchApply(self.conv_net)(timestep, c_state)

        #  sr net
        sr_inputs = (features, timestep)
        sr_output, sr_state = self.sr_net(sr_inputs, sr_state)

        #  policy net
        core_output, p_state = hk.BatchApply(self.policy_net)(sr_output.output, p_state)

        #  wrap up and return
        outputs = sr_output._replace(output=core_output)
        state = (c_state, sr_state, p_state)
        return outputs, state

    def __call__(self, timestep: dm_env.TimeStep, state: Any):
        timestep = jax.tree_map(lambda t: t[None, ...], timestep)
        return self.unroll(timestep, state)


class SrNet(_SR):
    def __init__(
        self,
        num_actions,
        vision_net_fn,
        capacity,
        memory_size=256,
        alpha=0.3,
        beta=1.0,
    ):
        #  make sure the rnn computes the sr model
        sr_config = {
            "memory_size": memory_size,
            "capacity": capacity,  #  max_ep_steps
            "alpha": alpha,
            "beta": beta,
            "name": "sr",
            # "apply_core_to_input": False,
        }
        super().__init__(num_actions, vision_net_fn, **sr_config)


class ImpalaNet(_SR):
    def __init__(self, num_actions, vision_net_fn, **ignored_kwargs):
        #  make sure the rnn skips the sr model
        sr_config = {
            "memory_size": 256,
            "capacity": 140,  # 0
            "alpha": 0.0,
            "beta": 1.0,
            "loss_func": lambda x, y: 0.0,
            # "apply_core_to_input": True,
            "name": "sr_ablated",
        }
        super().__init__(num_actions, vision_net_fn, **sr_config)
