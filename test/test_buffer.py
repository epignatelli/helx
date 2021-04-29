from helx.rl.buffer import ReplayBuffer, Transition
from gym_minigrid.wrappers import *
from bsuite.utils.gym_wrapper import DMEnvFromGym


def make(name):
    env = gym.make(name)
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
    return env


def test_buffer():
    buffer = ReplayBuffer(10, 1)
    env = make("MiniGrid-Empty-8x8-v0")
    num_episodes = 10

    for episode in range(num_episodes):
        print(
            "Starting episode number {}/{}\t\t\t".format(episode, num_episodes - 1),
            end="\r",
        )
        # initialise environment
        timestep = env.reset()
        while not timestep.last():
            # policy
            action = 1
            # step environment
            new_timestep = env.step(action)
            # update
            buffer.add(timestep, action, new_timestep)
            # prepare next
            timestep = new_timestep

    transition_batched = buffer.sample(2)


if __name__ == "__main__":
    test_buffer()
