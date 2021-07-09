"""A stateless agent interface."""
import collections
import functools
from typing import Any, Callable, Optional, Tuple

import dm_env
import haiku as hk
from examples.impala import util
import jax
import jax.numpy as jnp
import numpy as np
import impala.agent as impala_agent_lib
from models import SyntheticReturnsCoreWrapper as SR

AgentOutput = collections.namedtuple(
    "AgentOutput", ["policy_logits", "values", "action"]
)

Action = int
Nest = Any
NetFactory = Callable[[int], hk.RNNCore]


class Agent(impala_agent_lib.Agent):
    """A stateless agent interface."""

    def __init__(self, num_actions: int, obs_spec: Nest, net_factory: NetFactory):
        """Constructs an Agent object.

        Args:
          num_actions: Number of possible actions for the agent. Assumes a flat,
            discrete, 0-indexed action space.
          obs_spec: The observation spec of the environment.
          net_factory: A function from num_actions to a Haiku module representing
            the agent. This module should have an initial_state() function and an
            unroll function.
        """
        #  rl algorithm
        self._obs_spec = obs_spec
        rl_net_factory = functools.partial(net_factory, num_actions)
        # Instantiate two hk.transforms() - one for getting the initial state of the
        # agent, another for actually initializing and running the agent.
        _, self._rl_initial_state_apply_fn = hk.without_apply_rng(
            hk.transform(lambda batch_size: rl_net_factory().initial_state(batch_size))
        )

        self._rl_init_fn, self._rl_apply_fn = hk.without_apply_rng(
            hk.transform(lambda obs, state: rl_net_factory().unroll(obs, state))
        )

        #  sr algorithm
        memory_size = 128
        capacity = 300
        hidden_layers = (128, 128)
        alpha = 0.3
        beta = 1.0
        sr_core = hk.LSTM(memory_size)
        sr_net_factory = hk.ResetCore(
            SR(sr_core, memory_size, capacity, hidden_layers, alpha, beta)
        )
        # Instantiate two hk.transforms() - one for getting the initial state of the
        # agent, another for actually initializing and running the agent.
        _, self._sr_initial_state_apply_fn = hk.without_apply_rng(
            hk.transform(lambda batch_size: sr_net_factory().initial_state(batch_size))
        )

        self._sr_init_fn, self._sr_apply_fn = hk.without_apply_rng(
            hk.transform(lambda obs, state: sr_net_factory().unroll(obs, state))
        )

    @functools.partial(jax.jit, static_argnums=0)
    def initial_params(self, rng_key):
        """Initializes the agent params given the RNG key."""
        dummy_inputs = jax.tree_map(
            lambda t: np.zeros(t.shape, t.dtype), self._obs_spec
        )
        dummy_inputs = util.preprocess_step(dm_env.restart(dummy_inputs))
        dummy_inputs = jax.tree_map(lambda t: t[None, None, ...], dummy_inputs)
        return self._init_fn(rng_key, dummy_inputs, self.initial_state(1))

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def initial_state(self, batch_size: Optional[int]):
        """Returns agent initial state."""
        # We expect that generating the initial_state does not require parameters.
        return self._initial_state_apply_fn(None, batch_size)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng_key,
        params: hk.Params,
        timestep: dm_env.TimeStep,
        state: Nest,
    ) -> Tuple[AgentOutput, Nest]:
        """For a given single-step, unbatched timestep, output the chosen action."""
        # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.
        timestep = jax.tree_map(lambda t: t[None, None, ...], timestep)
        state = jax.tree_map(lambda t: t[None, ...], state)

        net_out, next_state = self._apply_fn(params, timestep, state)
        # Remove the padding from above.
        net_out = jax.tree_map(lambda t: jnp.squeeze(t, axis=(0, 1)), net_out)
        next_state = jax.tree_map(lambda t: jnp.squeeze(t, axis=0), next_state)
        # Sample an action and return.
        action = hk.multinomial(rng_key, net_out.policy_logits, num_samples=1)
        action = jnp.squeeze(action, axis=-1)
        return AgentOutput(net_out.policy_logits, net_out.value, action), next_state

    def unroll(
        self,
        params: hk.Params,
        trajectory: dm_env.TimeStep,
        state: Nest,
    ) -> AgentOutput:
        """Unroll the agent along trajectory."""
        net_out, _ = self._apply_fn(params, trajectory, state)
        return AgentOutput(net_out.policy_logits, net_out.value, action=[])
