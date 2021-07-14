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
"""IMPALA learner class."""

import functools
import itertools
import queue
import threading
from typing import Dict, Tuple
import warnings

import dm_env
import haiku as hk
from impala import agent as agent_lib
from impala import util
import jax
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np
import optax
import rlax


# The IMPALA paper sums losses, rather than taking the mean.
# We wrap rlax to do so as well.
def policy_gradient_loss(logits, *args):
    """rlax.policy_gradient_loss, but with sum(loss) and [T, B, ...] inputs."""
    mean_per_batch = jax.vmap(rlax.policy_gradient_loss, in_axes=1)(logits, *args)
    total_loss_per_batch = mean_per_batch * logits.shape[0]
    return jnp.sum(total_loss_per_batch)


def entropy_loss(logits, *args):
    """rlax.entropy_loss, but with sum(loss) and [T, B, ...] inputs."""
    mean_per_batch = jax.vmap(rlax.entropy_loss, in_axes=1)(logits, *args)
    total_loss_per_batch = mean_per_batch * logits.shape[0]
    return jnp.sum(total_loss_per_batch)


class Learner:
    """Manages state and performs updates for IMPALA learner."""

    def __init__(
        self,
        agent: agent_lib.Agent,
        rng_key,
        opt: optax.GradientTransformation,
        batch_size: int,
        discount_factor: float,
        frames_per_iter: int,
        max_abs_reward: float = 0,
        use_synthetic_returns=False,
        logger=None,
    ):
        if jax.device_count() > 1:
            warnings.warn(
                "Note: the impala example will only take advantage of a "
                "single accelerator."
            )

        self._agent = agent
        self._opt = opt
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._frames_per_iter = frames_per_iter
        self._max_abs_reward = max_abs_reward
        self._use_synthetic_returns = use_synthetic_returns

        # Data pipeline objects.
        self._done = False
        self._host_q = queue.Queue(maxsize=self._batch_size)
        self._device_q = queue.Queue(maxsize=1)

        # Prepare the parameters to be served to actors.
        params = agent.initial_params(rng_key)
        self._params_for_actor = (0, jax.device_get(params))

        # Set up logging.
        if logger is None:
            logger = util.NullLogger
        self._logger = util.WandbLogger("impala")

    def _loss(
        self,
        theta: hk.Params,
        trajectories: util.Transition,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute vtrace-based actor-critic loss."""
        initial_state = jax.tree_map(lambda t: t[0], trajectories.agent_state)
        sr_output = self._agent.unroll(theta, trajectories.timestep, initial_state)

        #  swap between sr and impala here
        use_sr = int(self._use_synthetic_returns)
        net_out = sr_output.output
        sr_output = sr_output._replace(
            augmented_return=sr_output.augmented_return * use_sr
        )
        sr_output = sr_output._replace(
            synthetic_return=sr_output.synthetic_return * use_sr
        )
        sr_output = sr_output._replace(sr_loss=sr_output.sr_loss * use_sr)

        v_t = net_out.values[1:]
        # Remove bootstrap timestep from non-timesteps.
        _, actor_out, _ = jax.tree_map(lambda t: t[:-1], trajectories)
        net_out = jax.tree_map(lambda t: t[:-1], net_out)
        v_tm1 = net_out.values

        # Get the discount, reward, step_type from the *next* timestep.
        timestep = jax.tree_map(lambda t: t[1:], trajectories.timestep)
        discounts = timestep.discount * self._discount_factor
        rewards = timestep.reward
        #  if using the sr model, reward is the compound reward
        if self._use_synthetic_returns:
            rewards = sr_output.augmented_return

        if self._max_abs_reward > 0:
            rewards = jnp.clip(rewards, -self._max_abs_reward, self._max_abs_reward)

        # The step is uninteresting if we transitioned LAST -> FIRST.
        # timestep corresponds to the *next* time step, so we filter for FIRST.
        mask = jnp.not_equal(timestep.step_type, int(dm_env.StepType.FIRST))
        mask = mask.astype(jnp.float32)

        rhos = rlax.categorical_importance_sampling_ratios(
            net_out.policy_logits, actor_out.policy_logits, actor_out.action
        )
        # vmap vtrace_td_error_and_advantage to take/return [T, B, ...].
        vtrace_td_error_and_advantage = jax.vmap(
            rlax.vtrace_td_error_and_advantage, in_axes=1, out_axes=1
        )

        vtrace_returns = vtrace_td_error_and_advantage(
            v_tm1, v_t, rewards, discounts, rhos
        )
        pg_advs = vtrace_returns.pg_advantage
        pg_loss = policy_gradient_loss(
            net_out.policy_logits, actor_out.action, pg_advs, mask
        )

        value_loss = 0.5 * jnp.sum(jnp.square(vtrace_returns.errors) * mask)
        ent_loss = entropy_loss(net_out.policy_logits, mask)

        sr_loss = jnp.sum(sr_output.sr_loss)
        sr_loss *= int(self._use_synthetic_returns)

        total_loss = pg_loss
        total_loss += 0.5 * value_loss
        total_loss += 0.01 * ent_loss
        total_loss += sr_loss

        logs = {}
        logs["mdp_return"] = jnp.sum(timestep.reward)
        logs["synthetic_return"] = jnp.sum(sr_output.synthetic_return)

        logs["pg_loss"] = pg_loss
        logs["value_loss"] = value_loss
        logs["entropy_loss"] = ent_loss
        logs["sr_loss"] = sr_loss
        logs["total_loss"] = total_loss
        return total_loss, logs

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, params, opt_state, batch: util.Transition):
        """The actual update function."""
        (_, logs), grads = jax.value_and_grad(self._loss, has_aux=True)(params, batch)

        grad_norm_unclipped = optimizers.l2_norm(grads)
        updates, updated_opt_state = self._opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        weight_norm_l2 = optimizers.l2_norm(params)
        weight_norm_l1 = util.l1_norm(params)
        logs.update(
            {
                "grad_norm_unclipped": grad_norm_unclipped,
                "weight_norm_l1": weight_norm_l1,
                "weight_norm_l2": weight_norm_l2,
            }
        )
        return params, updated_opt_state, logs

    def enqueue_traj(self, traj: util.Transition):
        """Enqueue trajectory."""
        self._host_q.put(traj)

    def params_for_actor(self) -> Tuple[int, hk.Params]:
        return self._params_for_actor

    def host_to_device_worker(self):
        """Elementary data pipeline."""
        batch = []
        while not self._done:
            # Try to get a batch. Skip the iteration if we couldn't.
            try:
                for _ in range(len(batch), self._batch_size):
                    # As long as possible while keeping learner_test time reasonable.
                    batch.append(self._host_q.get(timeout=10))
            except queue.Empty:
                continue

            assert len(batch) == self._batch_size
            # Prepare for consumption, then put batch onto device.
            stacked_batch = jax.tree_multimap(lambda *xs: np.stack(xs, axis=1), *batch)
            self._device_q.put(jax.device_put(stacked_batch))

            # Clean out the built-up batch.
            batch = []

    def run(self, max_iterations: int = -1):
        """Runs the learner for max_iterations updates."""
        # Start host-to-device transfer worker.
        transfer_thread = threading.Thread(target=self.host_to_device_worker)
        transfer_thread.start()

        (num_frames, params) = self._params_for_actor
        opt_state = self._opt.init(params)

        steps = range(max_iterations) if max_iterations != -1 else itertools.count()
        for _ in steps:
            batch = self._device_q.get()
            params, opt_state, logs = self.update(params, opt_state, batch)
            num_frames += self._frames_per_iter

            # Collect parameters to distribute to downstream actors.
            self._params_for_actor = (num_frames, jax.device_get(params))

            # Collect and write logs out.
            logs = jax.device_get(logs)
            logs = {k: float(v) for k, v in logs.items()}
            logs.update(
                {
                    "num_frames": int(num_frames),
                }
            )
            self._logger.write(logs)

        # Shut down.
        self._done = True
        self._logger.close()
        transfer_thread.join()
