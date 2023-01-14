import gymnasium
import optax
from absl import app, flags, logging

import helx
from helx.networks import MLP

helx.flags.define_flags_from_hparams(helx.agents.SACHparams)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.info("Starting")

    # environment
    env = gymnasium.make("HalfCheetah-v4", max_episode_steps=100)
    env = helx.environment.make_from(env)

    # optimiser
    optimiser = optax.adam(learning_rate=FLAGS.learning_rate)

    # agent
    hparams = helx.flags.hparams_from_flags(
        helx.agents.SACHparams,
        obs_space=env.observation_space(),
        action_space=env.action_space(),
        replay_start=10,
        batch_size=2,
    )
    agent = helx.agents.SAC(
        hparams=hparams,
        optimiser=optimiser,
        seed=0,
        actor_representation_net=MLP(features=[128, 128]),
        critic_representation_net=MLP(features=[128, 128]),
    )

    helx.experiment.run(agent, env, 2)


if __name__ == "__main__":
    app.run(main)
