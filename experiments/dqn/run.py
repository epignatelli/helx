import helx
from helx.rl import baselines, environment
import wandb


if __name__ == "__main__":

    env = environment.make_minigrid("MiniGrid-Empty-6x6-v0")
    hparams = baselines.dqn.HParams(
        replay_memory_size=5000, replay_start=5000, batch_size=32
    )
    wandb.init()
    dqn = baselines.dqn.Dqn((56, 56, 3), env.action_spec().num_values, hparams)
    helx.rl.run.run(dqn, env, 1000000)
