import helx.logging
import helx.rl
import helx.rl.experiment
import jax.numpy as jnp
from absl import app, flags
from gym_minigrid.wrappers import ImgObsWrapper

import agent as a_lib
import environment as e_lib


flags.DEFINE_integer("size", 8, "Size of the minigrid", 3)
flags.DEFINE_integer(
    "n_goals", 2, "Number of goals that will spawn in the environment in each episode"
)
flags.DEFINE_integer(
    "n_traps", 1, "Number of traps that will spawn in the environment in each episode"
)
flags.DEFINE_float("alpha", 0.1, "Step size parameter")
flags.DEFINE_float(
    "lamda", 0.9, "Trace decay paremterer for the Sarsa(λ) algorithm", 0, 1
)
flags.DEFINE_integer(
    "train_episodes", 1_000_000, "Max number of episodes to learn from"
)
flags.DEFINE_integer(
    "eval_episodes", 1_000_000, "Max number of episodes to evaluate on"
)
FLAGS = flags.FLAGS


def main(argv):
    env = e_lib.EmptyMultigoal(
        size=FLAGS.size, n_goals=FLAGS.n_goals, n_traps=FLAGS.n_traps
    )
    env = e_lib.SymbolicObsWrapper(env)
    env = ImgObsWrapper(env)
    env = helx.rl.environment.from_gym(env)

    n_features = jnp.prod(env.observation_spec().shape)
    logger = helx.logging.TerminalLogger()
    agent = a_lib.SarsaLambda(env, FLAGS.alpha, FLAGS.lamda, n_features, logger)

    helx.rl.experiment.run(env, agent, FLAGS.train_episodes)
    helx.rl.experiment.run(env, agent, FLAGS.eval_episodes, True)


if __name__ == "__main__":
    app.run(main)
