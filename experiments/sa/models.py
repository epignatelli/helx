import collections
from typing import Any, Tuple

import dm_env
import haiku as hk
from haiku._src.basic import Linear
from haiku._src.conv import Conv2D
from jax._src.lax.lax import Array
import jax.nn
import jax.numpy as jnp
import sr.models as sr_models_lib

NetOutput = collections.namedtuple("NetOutput", ["policy_logits", "value"])


class ConvNet(hk.Module):
    def __call__(self, x: dm_env.TimeStep):
        return hk.Sequential(
            [
                Conv2D(32, (2, 2), (1, 1)),
                jax.nn.relu,
                Conv2D(64, (2, 2), (1, 1)),
                jax.nn.relu,
                hk.Linear(256),
                jax.nn.relu,
            ]
        )(x.observation)


class SyntheticReturns(hk.RNNCore):
    """A simple neural network for catch."""

    def __init__(self, num_actions, name=None, **sr_config):
        super().__init__(name=name)
        self._num_actions = num_actions
        self._sr_config = sr_config

    def initial_state(self, batch_size):
        if batch_size is None:
            shape = []
        else:
            shape = [batch_size]
        return jnp.zeros(shape)  # Dummy.

    def __call__(self, x: dm_env.TimeStep, rl_state):
        net = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(256),
                jax.nn.relu,
            ]
        )
        #  get the features
        features = net(x.observation)

        #  get the rl outputs
        policy_logits = hk.Linear(self._num_actions)(features)
        value = jnp.squeeze(hk.Linear(1)(features), axis=-1)
        rl_output = NetOutput(policy_logits=policy_logits, value=value)

        #  get the sr outputs
        should_reset = jnp.equal(x.step_type[:-1], int(dm_env.StepType.FIRST))
        sr_output, sr_state = hk.ResetCore(
            sr_models_lib.SyntheticReturnsCoreWrapper(hk.LSTM(128), **self._sr_config)
        )(features, should_reset)

        #  wrap up an return
        outputs = (rl_output, sr_output)
        states = (rl_state, sr_state)
        return outputs, states

    def unroll(self, x, state):
        """Unrolls more efficiently than dynamic_unroll."""
        out, _ = hk.BatchApply(self)(x, None)
        return out, state


class PolicyNetwork(hk.Module):
    def __init__(self, num_actions, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions

    def __call__(self, x: Array):
        features = hk.Sequential([Linear(256), jax.nn.relu])(x)
        logits = hk.Linear(self.num_actions)(features)
        value = jnp.squeeze(hk.Linear(1)(features), axis=-1)
        return (logits, value)


class SrNet(hk.RNNCore):
    def __init__(self, num_actions, name=None, **sr_config):
        self.num_actions = num_actions
        self.sr_config = sr_config
        sr_config.update({"apply_core_to_input": False})
        super().__init__(name)

    def __call__(self, x: dm_env.TimeStep, state: Any):
        features = ConvNet()(x.observation)


class ImpalaNet(_SyntheticReturns):
    def __init__(self, num_actions, name=None, **sr_config):
        sr_config.update({"apply_core_to_input": True})
        super().__init__(
            num_actions,
            name,
        )
