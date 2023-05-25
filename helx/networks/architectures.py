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


from __future__ import annotations

from typing import Any, Tuple, Dict

import flax.linen as nn
import jax.numpy as jnp
from chex import Array, PyTreeDef
from jax.random import KeyArray

from ..mdp import Action
from .modules import Identity


class AgentNetwork(nn.Module):
    """Defines the network architecture of an agent, and can be used as it is.
    Args:
        embedding_torso (nn.Module): A Flax module that defines the state
            representation network. If the representation is shared between the actor
            and critic, then this module should be an instance of `Identity`, and
            the input of actor_head and critic_head should be the raw observation.
        actor_head (nn.Module): A Flax module that defines the actor network.
            The signature of this module should be `f(Array, KeyArray)`.
        critic_head (nn.Module): A Flax module that defines the critic network.
            The signature of this module should be `f(Array)`.
        state_transition_head (nn.Module): A Flax module that defines the
            state-transition dynamics network. Used for model-based RL.
            The signature of this module should be `f(Array, Action)`.
        reward_head (nn.Module): A Flax module that defines the reward network.
            Used for model-based RL.
            The signature of this module should be `f(Array, Action)`.
        custom_head (nn.Module): A Flax module that computes custom, unstructured data
            (e.g. a log of the agent's actions, additional rewards, goals).
            The signature of this module should be `f(Array, Action)`.
    """

    embedding_torso: nn.Module = Identity()
    actor_head: nn.Module | None = None
    critic_head: nn.Module | None = None
    state_transition_head: nn.Module | None = None
    reward_head: nn.Module | None = None
    custom_head: nn.Module | None = None

    @nn.compact
    def __call__(
        self, observation: Array, action: Action, key: KeyArray
    ) -> Tuple[Array, Array, Array, Array, Array | PyTreeDef, Array]:
        representation = self.embedding_torso(observation)

        actor_out = jnp.empty((0,))
        if self.actor_head is not None:
            actor_out = self.actor_head(representation, key)

        critic_out = jnp.empty((0,))
        if self.critic_head is not None:
            critic_out = self.critic_head(representation)

        state_transition_out = jnp.empty((0,))
        if self.state_transition_head is not None:
            state_transition_out = self.state_transition_head(representation, action)

        reward_out = jnp.empty((0,))
        if self.reward_head is not None:
            reward_out = self.reward_head(representation, action)

        extra_out = jnp.empty((0,))
        if self.custom_head is not None:
            extra_out = self.custom_head(observation, action)

        return (
            actor_out,
            critic_out,
            state_transition_out,
            reward_out,
            extra_out,
            representation,
        )

    def _submodule_params(
        self, params: nn.FrozenDict, key: str
    ) -> Dict[str, Any] | nn.FrozenDict:
        """Returns the parameters of a sub-module."""
        state = params
        out = {}
        for primal_key in state:
            if key in state[primal_key]:
                out[primal_key] = state[primal_key][key]

        return out

    def embedding(self, params: nn.FrozenDict, observation: Array) -> Array:
        """Computes the state representation $\\phi(s)$.§
        Args:
            params (nn.FrozenDict): The set of all network parameters.
            observation (Array): The environment observation.
        Returns:
            Array: The state representation."""
        return jnp.asarray(
            self.embedding_torso.apply(
                {"params": params["params"]["embedding_torso"]},
                observation,
            )
        )

    def actor(
        self, params: nn.FrozenDict, state_embedding: Array, key: KeyArray
    ) -> Tuple[Action, Array]:
        """The actor function to evaluate the agent's policy $\\pi(s)$.
        Args:
            params (nn.FrozenDict): The agent's parameters.
            observation (Array): The environment observation.
            key (KeyArray): A random key for sampling from the policy
                distribution.

        Returns:
            Tuple[Action, Array]: The sampled action and the log probability of the
                action under the policy distribution."""
        if self.actor_head is None:
            raise ValueError("The network has no actor head")

        submodule_params = self._submodule_params(params, "actor_head")
        (action, log_probs), _ = self.actor_head.apply(
            submodule_params, state_embedding, key
        )
        return action, log_probs

    def critic(
        self, params: nn.FrozenDict, state_embedding: Array, *args, **kwargs
    ) -> Array | PyTreeDef:
        """The critic function to evaluate the agent's value function
        $V(s)$ or $Q(s) \\forall a in \\mathcal{A}
        Args:
            params (nn.FrozenDict): The agent's parameters.
            observation (Array): The environment observation.

        Returns:
            Array | PyTreeDef: The value of the state under the value function.
            Can be an Array representing the value, or a PyTreeDef, like a Tuple
            of two Arrays for a Double-Q network ."""
        if self.critic_head is None:
            raise ValueError("The critic head is not defined")

        submodule_params = self._submodule_params(params, "critic_head")
        values, _ = self.critic_head.apply(submodule_params, state_embedding)
        return values

    def state_transition(
        self, params: nn.FrozenDict, state_embedding: Array, action: Action
    ) -> Array:
        """The state transition function to evaluate the agent's dynamics model
            $p(s' | s, a)$.
        Args:
            observation (Array): The observation to condition onto.
        Returns:
            Array: The next state or the probability distribution for the next state over
                a set of states"""
        if self.state_transition_head is None:
            raise ValueError("The state transition net is not defined")

        submodule_params = self._submodule_params(params, "state_transition_head")
        out, _ = self.state_transition_head.apply(
            submodule_params, state_embedding, action
        )
        return out

    def reward(
        self, params: nn.FrozenDict, state_embedding: Array, action: Action
    ) -> Array:
        if self.reward_head is None:
            raise ValueError("The reward net is not defined")

        submodule_params = self._submodule_params(params, "reward_head")
        out, _ = self.reward_head.apply(submodule_params, state_embedding, action)
        return out

    def custom(
        self,
        params: nn.FrozenDict,
        observation: Array | None = None,
        action: Action | None = None,
    ) -> Array | PyTreeDef:
        if self.custom_head is None:
            raise ValueError("The custom net is not defined")

        submodule_params = self._submodule_params(params, "custom_head")
        out, _ = self.custom_head.apply(submodule_params, observation, action)
        return out
