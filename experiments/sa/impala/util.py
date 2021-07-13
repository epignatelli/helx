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
"""Util."""
import collections

import dm_env
import numpy as np
import tree
import wandb
from absl import logging
import jax
import jax.numpy as jnp


# Can represent either a single transition, a trajectory, or a batch of
# trajectories.
Transition = collections.namedtuple(
    "Transition", ["timestep", "agent_out", "agent_state"]
)


def _preprocess_none(t) -> np.ndarray:
    if t is None:
        return np.array(0.0, dtype=np.float32)
    else:
        return np.asarray(t)


def preprocess_step(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    if timestep.discount is None:
        timestep = timestep._replace(discount=1.0)
    return tree.map_structure(_preprocess_none, timestep)


class NullLogger:
    """Logger that does nothing."""

    def write(self, _):
        pass

    def close(self):
        pass


class AbslLogger:
    """Writes to logging.info."""

    def write(self, d):
        logging.info(d)

    def close(self):
        pass


class WandbLogger(AbslLogger):
    """Writes to logging.info."""

    def __init__(self, project=None):
        self.base = super()
        wandb.init(project=project or "sr")

    def write(self, d):
        try:
            wandb.log(d)
        except RuntimeError as e:
            self.base.write("Wandb failed: {}".format(repr(e)))
        self.base.write(d)

    def close(self):
        pass


def l1_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = jax.tree_flatten(tree)
    return jnp.abs(sum(jnp.vdot(x, x) for x in leaves))
