import logging
import time
import multiprocessing as mp
from helx.rl.environment import make_minigrid, MultiprocessEnv


def test_multiprocess_env():
    n = 32
    env = make_minigrid("MiniGrid-Empty-5x5-v0")
    env = MultiprocessEnv(env, n)
    env.reset()
    logging.debug("resetted env")
    for i in range(5):
        env.step((0,) * n)
        logging.debug("env stepped")


def test_multiprocess_env_async():
    n = 3
    env = make_minigrid("MiniGrid-Empty-5x5-v0")
    env = MultiprocessEnv(env, n)
    m = mp.Manager()
    buffer = m.Queue(10)
    logging.debug(type(buffer))
    env.reset()
    logging.debug("environment reset")
    for i in range(5):
        env.step_async([0] * n, buffer)
    #  give the time to the env to compute the step
    time.sleep(5)
    for i in range(10):
        buffer.get()
    print("Queue size is:", buffer.qsize())


if __name__ == "__main__":
    test_multiprocess_env()
    test_multiprocess_env_async()
