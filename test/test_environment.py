import multiprocessing as mp
from helx.rl.environment import make_minigrid, MultiprocessEnv


def test_multiprocess_env():
    n = 32
    env = make_minigrid("MiniGrid-Empty-5x5-v0")
    env = MultiprocessEnv(env, n)
    env.reset()
    print("resetted env")
    for i in range(5):
        env.step((0,) * n)
        print("env stepped")
    # env.close()


def test_multiprocess_env_async():
    n = 3
    env = make_minigrid("MiniGrid-Empty-5x5-v0")
    env = MultiprocessEnv(env, n)
    m = mp.Manager()
    queue = m.Queue()
    print(env.reset())
    for i in range(5):
        print(env.step_async([0] * n, queue))
        print("Queue size is:", queue.qsize())
    env.close()


if __name__ == "__main__":
    test_multiprocess_env()
    # test_multiprocess_env_async()
