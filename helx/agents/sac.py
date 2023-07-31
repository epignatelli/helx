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


# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import linen as nn
from flax import struct
from flax.core.scope import VariableDict as Params
from jax import Array
from jax.random import KeyArray
from optax import GradientTransformation

from ..modules import Parallel, Split, Temperature
from ..mdp import StepType, Timestep
from ..memory import ReplayBuffer
from ..spaces import Continuous
from .agent import Agent, AgentState, HParams, Log


class SACHParams(HParams):
    # network
    action_space: Continuous
    tau: float = 0.005
    # rl
    replay_start: int = 1000
    replay_memory_size: int = 1000
    update_frequency: int = 1
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    n_steps: int = 1
    entropy_rewards: bool = False
    # sgd
    batch_size: int = 32
    learning_rate: float = 3e-4


class SACLog(Log):
    buffer_size: Array = jnp.asarray(0)
    critic_loss: Array = jnp.asarray(float("inf"))
    actor_loss: Array = jnp.asarray(float("inf"))
    temperature_loss: Array = jnp.asarray(float("inf"))
    total_loss: Array = jnp.asarray(float("inf"))
    policy_entropy: Array = jnp.asarray(float("inf"))
    alpha: Array = jnp.asarray(float("inf"))


class SACState(AgentState):
    params: Tuple[Params, Params, Params] = struct.field(pytree_node=True)
    buffer: ReplayBuffer = struct.field(pytree_node=True)
    log: SACLog = struct.field(pytree_node=True)


class SAC(Agent):
    """Implements Soft Actor-Critic with gaussian policy for continuous action spaces.
    Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep
        reinforcement learning with a stochastic actor."
        International conference on machine learning. PMLR, 2018.
        https://arxiv.org/abs/1801.01290 for the base SAC.
    Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications."
        arXiv preprint arXiv:1812.05905 (2018).
        https://arxiv.org/abs/1812.05905 for the tunable entropy temperature
    Haonanm Yu, et al. "Do you need reward entropy (in practice)?".
        arXiv preprint arXiv:2201.12434. 2022 Jan 28.
        https://arxiv.org/abs/2201.12434

    Args:
        network (AgentNetwork): the network to use for the agent.
            the `actor_net` is usually a `GaussianPolicy` for continuous action spaces.
            the `critic_net` is double Q network.
            the `extra_net` is usually a `Temperature` layer to automatically
            tune the entropy pull.
        optimiser (GradientTransformation): an optax optimiser to perform gradient descent.
        hparams (SACHparams): the hyperparameters for the agent.
        seed (int): the seed to use for the agent.
            Using the same seed will result in reproducible agent results.
    """

    hparams: SACHParams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)
    actor: nn.Module = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)
    temperature: nn.Module = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        hparams: SACHParams,
        optimiser: optax.GradientTransformation,
        actor_backbone: nn.Module,
        critic_backbone: nn.Module,
    ) -> SAC:
        action_ndim = hparams.action_space.ndim
        actor = nn.Sequential(
            [
                actor_backbone,
                Split(2),
                Parallel((nn.Dense(action_ndim), nn.Dense(action_ndim))),
            ]
        )
        critic = nn.Sequential(
            [
                Split(2),
                Parallel((critic_backbone, jtu.tree_map(lambda x: x, critic_backbone))),
                Parallel((nn.Dense(1), nn.Dense(1))),
            ]
        )
        temperature = Temperature()
        return SAC(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            temperature=temperature,
        )

    def init(self, key: KeyArray) -> SACState:
        """Initialises the agent state.

        Args:
            key (KeyArray): a JAX PRNG key.

        Returns:
            SACState: the initial agent state.
        """
        key, k1, k2, k3 = jax.random.split(key, num=4)
        params = (
            self.actor.init(k1),
            self.critic.init(k2),
            self.temperature.init(k3),
        )
        opt_state = self.optimiser.init(params)
        buffer = ReplayBuffer.create(
            self.hparams.obs_space,
            self.hparams.action_space,
            self.hparams.n_steps,
            self.hparams.replay_memory_size,
        )
        return SACState(
            iteration=jnp.asarray(0),
            opt_state=opt_state,
            log=SACLog(),
            params=params,
            buffer=buffer,
        )

    def sample_action(
        self, train_state: SACState, obs: Array, *, key: KeyArray, eval: bool = False
    ) -> Array:
        params_actor, _, _ = train_state.params
        mu, logvar = self.actor.apply(params_actor, obs)
        mu, logvar = jnp.asarray(mu), jnp.asarray(logvar)
        action = distrax.Normal(mu, logvar).sample(seed=key)
        return action

    def loss(
        self,
        params: Tuple[Params, Params, Params],
        transition: Timestep,
    ) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
        s_tm1 = transition.observation[:-1]
        s_t = transition.observation[1:]
        r_t = transition.reward[:-1][0]  # [0] because scalar
        terminal_tm1 = transition.step_type[:-1] != StepType.TERMINATION

        # params
        params_actor, params_critic, params_temperature = params

        # current estimates
        temperature = jnp.asarray(self.temperature.apply(params_temperature))
        alpha = jax.lax.stop_gradient(temperature)
        logprobs_a_tm1 = self.actor(params_actor, s_tm1)
        logprobs_a_t = self.actor(params_actor, s_t)
        probs_a_tm1 = jnp.exp(logprobs_a_tm1)

        # augment reward with policy entropy
        policy_entropy = -jnp.sum(probs_a_tm1 * logprobs_a_tm1, axis=-1)
        # SACLite: https://arxiv.org/abs/2201.12434
        policy_entropy = jnp.asarray(policy_entropy * self.hparams.entropy_rewards)
        r_t = r_t + alpha * policy_entropy

        # critic is a double Q network
        qA_tm1, qB_tm1 = self.critic(params_critic, s_tm1)
        qA_t, qB_t = self.critic(params_critic, s_t)

        # actor loss
        q_tm1 = jax.lax.stop_gradient(
            jnp.min(jnp.stack([qA_tm1, qB_tm1], axis=0), axis=0)
        )
        actor_loss = jnp.asarray(alpha * logprobs_a_tm1 - q_tm1)

        # critic losses
        q_t = jnp.min(jnp.stack([qA_t, qB_t], axis=0), axis=0)
        v_t = q_t - alpha * logprobs_a_t
        q_target = jax.lax.stop_gradient(
            r_t + terminal_tm1 * self.hparams.discount * v_t
        )
        critic_loss = jnp.asarray(
            optax.l2_loss(qA_tm1, q_target) + optax.l2_loss(qB_tm1, q_target)
        )

        # temperature loss
        target_entropy = -self.hparams.action_space.ndim
        logprobs_a_tm1 = jax.lax.stop_gradient(logprobs_a_tm1)
        temperature_loss = jnp.asarray(
            -(temperature * (logprobs_a_tm1 + target_entropy))
        )

        loss = actor_loss + critic_loss + temperature_loss
        aux = (actor_loss, critic_loss, temperature_loss, policy_entropy, alpha)
        return loss, aux

    def update(
        self,
        train_state: SACState,
        transition: Timestep,
        *,
        key: KeyArray,
    ) -> SACState:
        # update iteration
        iteration = jnp.asarray(train_state.iteration + 1, dtype=jnp.int32)

        # update memory
        buffer = train_state.buffer.add(transition)

        # update actor, critic and temperature
        def _sgd_step(params, opt_state):
            def _loss_fn(params, transition):
                loss, aux = jax.vmap(self.loss, in_axes=(None, 0))(params, transition)
                return jnp.mean(loss), aux

            transitions = buffer.sample(key, self.hparams.batch_size)
            (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                params, transitions
            )
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, aux

        cond = buffer.size() < self.hparams.replay_memory_size
        cond = jnp.logical_or(cond, iteration % self.hparams.update_frequency != 0)
        params, opt_state, loss, aux = jax.lax.cond(
            cond,
            lambda p, o: _sgd_step(p, o),
            lambda p, o: (p, o, jnp.asarray(float("inf"))),
            train_state.params,
            train_state.opt_state,
        )

        # update log
        log = SACLog(
            iteration=iteration,
            actor_loss=aux[0],
            critic_loss=aux[1],
            temperature_loss=aux[2],
            total_loss=loss,
            policy_entropy=aux[3],
            alpha=aux[4],
            step_type=transition.step_type[-1],
            returns=train_state.log.returns
            + jnp.sum(
                self.hparams.discount ** transition.t[:-1] * transition.reward[:-1]
            ),
            buffer_size=buffer.size(),
        )

        # update train_state
        train_state = train_state.replace(
            iteration=iteration,
            opt_state=opt_state,
            params=params,
            buffer=buffer,
            log=log,
        )
        return train_state
