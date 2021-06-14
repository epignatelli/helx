from helx.rl.environment import make_minigrid, MultiprocessEnv


def test_multiprocess_env():
    env = make_minigrid("MiniGrid-Empty-5x5-v0")
    env = MultiprocessEnv(env, 3)
    print(env.reset())
    for i in range(100):
        print(env.step([0, 0]))


if __name__ == "__main__":
    test_multiprocess_env()
