# # Copyright 2023 The Helx Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #   http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.


# # pyright: reportPrivateImportUsage=false
# from __future__ import annotations

# from typing import Tuple

# import distrax
# import jax
# import jax.numpy as jnp
# import jax.tree_util as jtu
# import optax
# from flax import linen as nn
# from flax import struct
# from flax.core.scope import VariableDict as Params
# from jax import Array
# from jax.random import KeyArray
# from optax import GradientTransformation
# import rlax

# from helx.base.modules import Lambda, Parallel, Split, Temperature
# from helx.base.mdp import Timestep, StepType
# from helx.base.memory import ReplayBuffer
# from helx.base.spaces import Continuous
# from helx.base import losses
# from helx.envs.environment import Environment

# from .agent import Agent, HParams, Log as LogBase


# class SACHParams(HParams):
#     # rl
#     discount: float = 0.99
#     n_steps: int = 1
#     # sgd
#     batch_size: int = 32
#     learning_rate: float = 3e-4
#     # SAC
#     tau: float = 0.005
#     replay_start: int = 1000
#     replay_memory_size: int = 1000
#     update_frequency: int = 1
#     target_network_update_frequency: int = 10000
#     entropy_rewards: bool = False
#     n_actors: int = 1
#     lambda_: float = 0.0  # TD(0)


# class Log(LogBase):
#     buffer_size: Array = jnp.asarray(0)
#     critic_loss: Array = jnp.asarray(float("inf"))
#     actor_loss: Array = jnp.asarray(float("inf"))
#     temperature_loss: Array = jnp.asarray(float("inf"))
#     total_loss: Array = jnp.asarray(float("inf"))
#     policy_entropy: Array = jnp.asarray(float("inf"))
#     alpha: Array = jnp.asarray(float("inf"))


# class SAC(Agent):
#     """Implements Soft Actor-Critic with gaussian policy for continuous action spaces.
#     Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep
#         reinforcement learning with a stochastic actor."
#         International conference on machine learning. PMLR, 2018.
#         https://arxiv.org/abs/1801.01290 for the base SAC.
#     Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications."
#         arXiv preprint arXiv:1812.05905 (2018).
#         https://arxiv.org/abs/1812.05905 for the tunable entropy temperature
#     Haonanm Yu, et al. "Do you need reward entropy (in practice)?".
#         arXiv preprint arXiv:2201.12434. 2022 Jan 28.
#         https://arxiv.org/abs/2201.12434

#     Args:
#         network (AgentNetwork): the network to use for the agent.
#             the `actor_net` is usually a `GaussianPolicy` for continuous action spaces.
#             the `critic_net` is double Q network.
#             the `extra_net` is usually a `Temperature` layer to automatically
#             tune the entropy pull.
#         optimiser (GradientTransformation): an optax optimiser to perform gradient descent.
#         hparams (SACHparams): the hyperparameters for the agent.
#         seed (int): the seed to use for the agent.
#             Using the same seed will result in reproducible agent results.
#     """

#     hparams: SACHParams = struct.field(pytree_node=False)
#     optimiser: GradientTransformation = struct.field(pytree_node=False)
#     actor: nn.Module = struct.field(pytree_node=False)
#     critic: nn.Module = struct.field(pytree_node=False)
#     critic_target: nn.Module = struct.field(pytree_node=False)
#     temperature: nn.Module = struct.field(pytree_node=False)
#     params: Params = struct.field(pytree_node=True)
#     buffer: ReplayBuffer = struct.field(pytree_node=True)

#     @classmethod
#     def init(
#         cls,
#         env: Environment,
#         hparams: SACHParams,
#         optimiser: optax.GradientTransformation,
#         actor_backbone: nn.Module,
#         critic_backbone: nn.Module,
#         *,
#         key: KeyArray,
#     ) -> SAC:
#         action_size = hparams.action_space.size
#         reshape = Lambda(lambda x: x.reshape(hparams.action_space.shape))
#         actor = nn.Sequential(
#             [
#                 actor_backbone,
#                 Split(2),
#                 Parallel((nn.Dense(action_size), nn.Dense(action_size))),
#                 Parallel((reshape, jtu.tree_map(lambda x: x, reshape))),
#             ]
#         )
#         critic = nn.Sequential([critic_backbone, nn.Dense(1)])
#         critic_target = nn.Sequential([critic_backbone, nn.Dense(1)])
#         temperature = Temperature()

#         k1, k2, k3 = jax.random.split(key, num=3)
#         obs = env.observation_space.sample(key=key)
#         params = {
#             "actor": actor.init(k1, obs),
#             "critic": critic.init(k2, obs),
#             "critic_target": critic_target.init(k2, obs),
#             "temperature": temperature.init(k3),
#         }
#         item = env.reset(key)
#         buffer = ReplayBuffer.create(item, hparams.replay_memory_size, hparams.n_steps)
#         return SAC(
#             hparams=hparams,
#             optimiser=optimiser,
#             actor=actor,
#             critic=critic,
#             critic_target=critic_target,
#             temperature=temperature,
#             params=params,
#             buffer=buffer,
#         )

#     def policy(self, params: Params, observation: Array) -> distrax.Distribution:
#         logits = jnp.asarray(self.actor.apply(params, observation))
#         return distrax.Categorical(logits=logits)

#     def value_fn(self, params: Params, observation: Array) -> Array:
#         return jnp.asarray(self.critic.apply(params, observation))

#     def collect_experience(self, env: Environment, *, key: KeyArray) -> Timestep:
#         timestep = env.reset(key)  # this is a batch of timesteps of t=0
#         episodes = []
#         for t in range(self.hparams.n_steps):
#             _, key = jax.random.split(key)
#             action_distribution = self.policy(self.params, timestep.observation)
#             action, log_probs = action_distribution.sample_and_log_prob(
#                 seed=key, sample_shape=env.action_space.shape
#             )
#             timestep = timestep.replace(info={"log_probs": log_probs})
#             episodes.append(timestep)
#             timestep = env.step(key, timestep, action)  # get new timestep
#         episodes.append(timestep)  # add last timestep
#         # first axis is the number of actors, second axis is time
#         return jtu.tree_map(lambda x: jnp.stack(x, axis=0), episodes)

#     def loss(self, params: Params, transition: Timestep) -> Tuple[Array, Log]:
#         values = self.value_fn(params, transition.observation[-1])
#         values = values * jnp.logical_not(
#             jnp.array_equal(transition[-1].step_type, StepType.TERMINATION)
#         )
#         values_target = self.value_fn(
#             self.critic_target.params, transition.observation[-1]
#         )
#         values_target = values_target * jnp.logical_not(
#             jnp.array_equal(transition[-1].step_type, StepType.TERMINATION)
#         )
#         advantage = rlax.truncated_generalized_advantage_estimation(
#             transition.reward,
#             self.hparams.discount**transition.t,
#             self.hparams.lambda_,
#             values_target,
#         )
#         # critic loss
#         critic_loss = jnp.square(values - advantage)
#         # actor loss
#         action_distribution = self.policy(params, transition.observation[-1])
#         log_probs = action_distribution.log_prob(transition.action[-1])
#         alpha = self.temperature.alpha(params)
#         actor_loss = -jnp.asarray(log_probs) * (
#             jax.lax.stop_gradient(values)
#             - jax.lax.stop_gradient(alpha)
#         )
#         # temperature loss
#         temperature_loss =
#         # total loss
#         total_loss = critic_loss + actor_loss + temperature_loss
#         # policy entropy
#         policy_entropy = action_distribution.entropy()
#         return (
#             total_loss,
#             Log(
#                 critic_loss=jnp.mean(critic_loss),
#                 actor_loss=jnp.mean(actor_loss),
#                 temperature_loss=jnp.mean(temperature_loss),
#                 total_loss=jnp.mean(total_loss),
#                 policy_entropy=jnp.mean(policy_entropy),
#                 alpha=jnp.mean(alpha),
#             ),
#         )

#     def update(
#         self,
#         episodes: ReplayBuffer,
#         *,
#         key: KeyArray,
#     ) -> SAC:
#         # update buffer
#         buffer = self.buffer.add(episodes)

#         transitions = buffer.sample(key, self.hparams.batch_size)
#         (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
#             params, transitions
#         )

#         # update actor, critic and temperature
#         def _sgd_step(params, opt_state):
#             def _loss_fn(params, transition):
#                 loss, aux = jax.vmap(self.loss, in_axes=(None, 0))(params, transition)
#                 return jnp.mean(loss), aux

#             transitions = buffer.sample(key, self.hparams.batch_size)
#             (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
#                 params, transitions
#             )
#             updates, opt_state = self.optimiser.update(grads, opt_state)
#             params = optax.apply_updates(params, updates)
#             return params, opt_state, loss, aux

#         cond = buffer.size() < self.hparams.replay_memory_size
#         cond = jnp.logical_or(cond, iteration % self.hparams.update_frequency != 0)
#         params, opt_state, loss, aux = jax.lax.cond(
#             cond,
#             lambda p, o: _sgd_step(p, o),
#             lambda p, o: (p, o, jnp.asarray(float("inf"))),
#             train_state.params,
#             train_state.opt_state,
#         )

#         # update log
#         log = SACLog(
#             iteration=iteration,
#             actor_loss=aux[0],
#             critic_loss=aux[1],
#             temperature_loss=aux[2],
#             total_loss=loss,
#             policy_entropy=aux[3],
#             alpha=aux[4],
#             step_type=transition.step_type[-1],
#             returns=train_state.log.returns
#             + jnp.sum(
#                 self.hparams.discount ** transition.t[:-1] * transition.reward[:-1]
#             ),
#             buffer_size=buffer.size(),
#         )

#         # update train_state
#         train_state = train_state.replace(
#             iteration=iteration,
#             opt_state=opt_state,
#             params=params,
#             buffer=buffer,
#             log=log,
#         )
#         return train_state
