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
        actor_net (nn.Module): A Flax module that defines the actor network.
            The signature of this module should be `f(Array, KeyArray)`.
        critic_net (nn.Module): A Flax module that defines the critic network.
            The signature of this module should be `f(Array)`.
        transition_net (nn.Module): A Flax module that defines the
            state-transition dynamics network. Used for model-based RL.
            The signature of this module should be `f(Array, Action)`.
        reward_net (nn.Module): A Flax module that defines the reward network.
            Used for model-based RL.
            The signature of this module should be `f(Array, Action)`.
        extra_net (nn.Module): A Flax module that computes custom, unstructured data
            (e.g. a log of the agent's actions, additional rewards, goals).
            The signature of this module should be `f(Array, Action)`.
        state_representation_net (nn.Module): A Flax module that defines the state
            representation network. If the representation is shared between the actor
            and critic, then this module should be an instance of `Identity`, and
            the input of actor_net and critic_net should be the raw observation.
    """

    actor_net: nn.Module | None = None
    critic_net: nn.Module | None = None
    transition_net: nn.Module | None = None
    reward_net: nn.Module | None = None
    extra_net: nn.Module | None = None
    shared_net: nn.Module = Identity()

    @nn.compact
    def __call__(
        self, observation: Array, action: Action, key: KeyArray
    ) -> Tuple[Array, Array, Array, Array, Array | PyTreeDef, Array]:
        representation = self.shared_net(observation)

        actor_out = jnp.empty((0,))
        if self.actor_net is not None:
            actor_out = self.actor_net(representation, key)

        critic_out = jnp.empty((0,))
        if self.critic_net is not None:
            critic_out = self.critic_net(representation)

        state_transition_out = jnp.empty((0,))
        if self.transition_net is not None:
            state_transition_out = self.transition_net(representation, action)

        reward_out = jnp.empty((0,))
        if self.reward_net is not None:
            reward_out = self.reward_net(representation, action)

        extra_out = jnp.empty((0,))
        if self.extra_net is not None:
            extra_out = self.extra_net(observation, action)

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

    def actor(
        self, params: nn.FrozenDict, observation: Array, key: KeyArray
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
        if self.actor_net is None:
            raise ValueError("Actor net not defined")

        observation = self.state_representation(params, observation)

        submodule_params = self._submodule_params(params, "actor_net")
        (action, log_probs), _ = self.actor_net.apply(
            submodule_params, observation, key, mutable="stats"
        )
        return action, log_probs

    def critic(
        self, params: nn.FrozenDict, observation: Array, *args, **kwargs
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
        if self.critic_net is None:
            raise ValueError("Critic net not defined")

        observation = self.state_representation(params, observation)

        submodule_params = self._submodule_params(params, "critic_net")
        values, _ = self.critic_net.apply(
            submodule_params, observation, mutable="stats"
        )
        return values

    def state_transition(
        self, params: nn.FrozenDict, observation: Array, action: Action
    ) -> Array:
        """The state transition function to evaluate the agent's dynamics model
            $p(s' | s, a)$.
        Args:
            observation (Array): The observation to condition onto.
        Returns:
            Array: The next state or the probability distribution for the next state over
                a set of states"""
        if self.transition_net is None:
            raise ValueError("State transition net not defined")

        observation = self.state_representation(params, observation)
        submodule_params = self._submodule_params(params, "transition_net")

        out, _ = self.transition_net.apply(
            submodule_params,
            observation,
            action,
            mutable="stats",
        )
        return out

    def reward(
        self, params: nn.FrozenDict, observation: Array, action: Action
    ) -> Array:
        if self.reward_net is None:
            raise ValueError("Reward net not defined")

        observation = self.state_representation(params, observation)
        submodule_params = self._submodule_params(params, "reward_net")

        out, _ = self.reward_net.apply(
            submodule_params,
            observation,
            action,
            mutable="stats",
        )
        return out

    def extra(
        self,
        params: nn.FrozenDict,
        observation: Array | None = None,
        action: Action | None = None,
    ) -> Array | PyTreeDef:
        if self.extra_net is None:
            raise ValueError("Extra net not defined")

        submodule_params = self._submodule_params(params, "extra_net")

        out, _ = self.extra_net.apply(
            submodule_params,
            observation,
            action,
            mutable="stats",
        )
        return out

    def state_representation(self, params: nn.FrozenDict, observation: Array) -> Array:
        if "shared_net" not in params["params"]:
            return observation
        return jnp.asarray(
            self.shared_net.apply(
                {"params": params["params"]["shared_net"]},
                observation,
            )
        )
