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
"""Single-process IMPALA wiring."""

from functools import partial
import threading
from typing import List, NamedTuple

import haiku as hk
import jax
import optax
from absl import app, flags


import environments
import impala.actor as actor_lib
import impala.agent as agent_lib
import impala.learner as learner_lib
import impala.util as util
import impala.haiku_nets as models_lib

flags.DEFINE_bool("DEBUG", False, "")
flags.DEFINE_integer("ACTION_REPEAT", 1, "")
flags.DEFINE_integer("BATCH_SIZE", 2, "")
flags.DEFINE_float("DISCOUNT_FACTOR", 0.99, "")
flags.DEFINE_integer("MAX_ENV_FRAMES", 20000, "")
flags.DEFINE_integer("NUM_ACTORS", 2, "")
flags.DEFINE_integer("UNROLL_LENGTH", 20, "")
flags.DEFINE_integer("SEED", 0, "")
flags.DEFINE_enum("MODEL", "Impala", ("Impala", "Sr", "Sa"), "")
flags.DEFINE_enum("EXPERIMENT", "Catch", ("Catch", "KeyToDoor"), "")


class MemoryCapacity(NamedTuple):
    Catch: int = 140


def run_actor(actor: actor_lib.Actor, stop_signal: List[bool]):
    """Runs an actor to produce num_trajectories trajectories."""
    while not stop_signal[0]:
        frame_count, params = actor.pull_params()
        actor.unroll_and_push(frame_count, params)


def main(_):
    FLAGS = flags.FLAGS

    DEBUG = FLAGS.DEBUG
    ACTION_REPEAT = FLAGS.ACTION_REPEAT
    BATCH_SIZE = FLAGS.BATCH_SIZE
    DISCOUNT_FACTOR = FLAGS.DISCOUNT_FACTOR
    MAX_ENV_FRAMES = FLAGS.MAX_ENV_FRAMES
    NUM_ACTORS = FLAGS.NUM_ACTORS
    UNROLL_LENGTH = FLAGS.UNROLL_LENGTH
    SEED = FLAGS.SEED
    FRAMES_PER_ITER = ACTION_REPEAT * BATCH_SIZE * UNROLL_LENGTH
    MODEL = FLAGS.MODEL
    EXPERIMENT = FLAGS.EXPERIMENT

    # A thunk that builds a new environment.
    # Substitute your environment here!
    build_env = getattr(environments, EXPERIMENT)

    # Construct the agent. We need a sample environment for its spec.
    seed = SEED
    env_for_spec = build_env(seed)
    num_actions = env_for_spec.action_spec().num_values

    #  get the experiment model: (impala, sr, new)
    def model(n):
        vision_net = getattr(models_lib, EXPERIMENT + "ConvNet")
        f = getattr(models_lib, MODEL + "Net")
        return f(n, vision_net_fn=vision_net)

    # model = lambda n: model(n, vision_net_fn=vision_net)
    agent = agent_lib.Agent(num_actions, env_for_spec.observation_spec(), model)

    # Logger
    project_name = "_".join([str(EXPERIMENT), str(MODEL)])
    logger = util.AbslLogger() if DEBUG else util.WandbLogger(project_name)

    # Construct the optimizer.
    max_updates = MAX_ENV_FRAMES / FRAMES_PER_ITER
    opt = optax.rmsprop(5e-3, decay=0.99, eps=1e-7)

    # Construct the learner.
    learner = learner_lib.Learner(
        agent,
        jax.random.PRNGKey(428),
        opt,
        BATCH_SIZE,
        DISCOUNT_FACTOR,
        FRAMES_PER_ITER,
        max_abs_reward=1.0,
        logger=logger,
        use_synthetic_returns=(MODEL.lower() != "impala"),
    )

    # Construct the actors on different threads.
    # stop_signal in a list so the reference is shared.
    actor_threads = []
    stop_signal = [False]
    for i in range(NUM_ACTORS):
        seed += 1
        actor = actor_lib.Actor(
            agent,
            build_env(SEED),
            UNROLL_LENGTH,
            learner,
            rng_seed=i,
            logger=logger,
        )
        args = (actor, stop_signal)
        actor_threads.append(threading.Thread(target=run_actor, args=args))

    # Start the actors and learner.
    for t in actor_threads:
        t.start()
    learner.run(int(max_updates))

    # Stop.
    stop_signal[0] = True
    for t in actor_threads:
        t.join()


if __name__ == "__main__":
    app.run(main)
