import helx
from helx.rl import baselines, environment
import wandb
from absl import flags


if __name__ == "__main__":
    flags.DEFINE_string(
        "env",
        "MiniGrid-Empty-6x6-v0",
        "Environment name",
    )
    flags.DEFINE_integer(
        "seed",
        0,
        "Random seed to control the experiment",
    )
    flags.DEFINE_integer(
        "num_episodes",
        10,
        "Number of episodes to train on",
    )
    flags.DEFINE_integer(
        "num_eval_episodes",
        10,
        "Number of episodes to train on",
    )
    flags.DEFINE_integer(
        "batch_size",
        32,
        "Number of training cases over which each stochastic gradient descent (SGD) update is computed",
    )
    flags.DEFINE_integer(
        "replay_memory_size",
        1000000,
        "SGD updates are sampled from this number of most recent frames",
    )
    flags.DEFINE_integer(
        "agent_history_length",
        4,
        "The number of most recent frames experienced by the agent that are given as  input to the Q network",
    )
    flags.DEFINE_integer(
        "target_network_update_frequency",
        10000,
        "The frequency (measured in the number of parameters update) with which the target network is updated (this corresponds to the parameter C from Algorithm 1)",
    )
    flags.DEFINE_float(
        "discount",
        0.99,
        "Discount factor gamma used in the Q-learning update",
    )
    flags.DEFINE_integer(
        "action_repeat",
        4,
        "Repeat each action selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4th input frame",
    )
    flags.DEFINE_integer(
        "update_frequency",
        4,
        "The number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates.",
    )
    flags.DEFINE_float(
        "learning_rate",
        0.00025,
        "The learning rate used by the RMSProp",
    )
    flags.DEFINE_float(
        "gradient_momentum",
        0.95,
        "Gradient momentum used by the RMSProp",
    )
    flags.DEFINE_float(
        "squared_gradient_momentum",
        0.95,
        "Squared gradient (denominator) momentum used by the RMSProp",
    )
    flags.DEFINE_float(
        "min_squared_gradient",
        0.01,
        "Constant added to the squared gradient in the denominator of the RMSProp update",
    )
    flags.DEFINE_float(
        "initial_exploration",
        1.0,
        "Initial value of ɛ in ɛ-greedy exploration",
    )
    flags.DEFINE_float(
        "final_exploration",
        0.01,
        "Final value of ɛ in ɛ-greedy exploration",
    )
    flags.DEFINE_integer(
        "final_exploration_frame",
        1000000,
        "The number of frames over which the initial value of ɛ is linearly annealed to its final value",
    )
    flags.DEFINE_integer(
        "replay_start",
        50000,
        "A uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory",
    )
    flags.DEFINE_integer(
        "no_op_max",
        30,
        'Maximum numer of "do nothing" actions to be performed by the agent at the start of an episode',
    )
    flags.DEFINE_integer(
        "hidden_size",
        512,
        "Dimension of last linear layer for value regression",
    )
    flags.DEFINE_integer(
        "n_steps",
        1,
        "Number of timesteps for multistep return",
    )

    FLAGS = flags.FLAGS
    hparams = baselines.dqn.HParams(**FLAGS)

    env = environment.make_minigrid("MiniGrid-Empty-6x6-v0")
    hparams = baselines.dqn.HParams(
        replay_memory_size=FLAGS.replay_memory_size,
        replay_start=FLAGS.replay_start,
        batch_size=FLAGS.batch_size,
    )
    wandb.init()
    dqn = baselines.dqn.Dqn((56, 56, 3), env.action_spec().num_values, hparams)
    helx.rl.run.run(dqn, env, 1000000)
