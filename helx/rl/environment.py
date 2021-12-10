import logging
import multiprocessing as mp
from copy import deepcopy
from multiprocessing.connection import Connection
from typing import Sequence

import dm_env
import gym
import jax
import jax.numpy as jnp
from bsuite.utils.gym_wrapper import DMEnvFromGym
from gym_minigrid.wrappers import *
from helx.image import greyscale, imresize
from helx.random import PRNGSequence
from helx.typing import Size


def make(name):
    env = gym.make(name)
    env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
    return env


def make_minigrid(name):
    env = gym.make(name)
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
    env.render = env.gym_env.render
    return env


def preprocess_atari(x):
    """Preprocessing function from
    Mnih, V., 2015, https://www.nature.com/articles/nature14236
    """
    # depthwise max pooling to remove flickering
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (2, 1, 1, 1), (1, 1, 1, 1), "SAME"
    )
    return greyscale(imresize(x, (84, 84)))


def preprocess_minigrid(x, size: Size = (56, 56)):
    """Refer to the minigrid implementation at:
    https://github.com/maximecb/gym-minigrid
    """
    return imresize(x / 255, size=size, channel_first=False)


def actor(server: Connection, client: Connection, env: dm_env.Environment):
    def _step(env, a: int):
        timestep = env.step(a)
        if timestep.last():
            timestep = env.reset()
        return timestep

    def _step_async(env, a: int, buffer: mp.Queue):
        timestep = env.step(a)
        buffer.put(timestep)
        print(buffer.qsize())
        if timestep.last():
            timestep = env.reset()
        return

    #  close copy of server connection from client process
    #  see: https://stackoverflow.com/q/8594909/6655465
    server.close()
    #  switch case command
    try:
        while True:
            cmd, data = client.recv()
            if cmd == "step":
                client.send(_step(env, data))
            elif cmd == "step_async":
                client.send(_step_async(env, *data))
            elif cmd == "reset":
                client.send(env.reset())
            elif cmd == "render":
                client.send(env.render(data))
            elif cmd == "close":
                client.send(env.close())
                break
            else:
                raise NotImplementedError("Command {} is not implemented".format(cmd))
    except KeyboardInterrupt:
        logging.info("SubprocVecEnv actor: got KeyboardInterrupt")
    finally:
        env.close()


class MultiprocessEnv(dm_env.Environment):
    """
    This class is inspired by openai's SubprocEnv.
    https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py

    An environment that allows concurrent interactions to improve experience collection throughput, as used in:
    https://arxiv.org/abs/1602.01783, https://arxiv.org/abs/1802.01561 and https://arxiv.org/abs/1707.06347
    The class runs multiple subproceses and communicates with them via pipes.
    """

    def __init__(
        self,
        env: dm_env.Environment,
        n_actors: int,
        context: str = "spawn",
        seed: int = 0,
    ):
        assert isinstance(
            env, dm_env.Environment
        ), "The environment to parallelise but be a dm_env.Environment, got {}".format(
            type(env)
        )
        #  public:
        self.n_actors: int = n_actors
        self.clients: Sequence[Connection] = None
        self.servers: Sequence[Connection] = None
        self.envs: Sequence[dm_env.Environment] = [
            deepcopy(env) for _ in range(n_actors)
        ]
        self.processes = []

        #  setup parallel workers
        rng = PRNGSequence(seed)
        ctx = mp.get_context(context)
        pipes = zip(*[ctx.Pipe() for _ in range(self.n_actors)])
        self.clients, self.servers = pipes
        for server, client, env in zip(self.servers, self.clients, self.envs):
            env.gym_env.seed(int(next(rng)[0]))
            self.processes.append(
                ctx.Process(
                    target=actor,
                    args=(server, client, env),
                    daemon=True,
                )
            )

        for i, p in enumerate(self.processes):
            logging.info("Starting actor {} on process {}".format(i, p))
            p.start()
            #  close copy of client connection from server process
            #  see: https://stackoverflow.com/q/8594909/6655465
            self.clients[i].close()

    def __del__(self):
        self.close()
        for p in self.processes:
            p.join()

    def reward_spec(self):
        return self.envs[0].reward_spec()

    def observation_spec(self):
        return self.envs[0].observation_spec()

    def action_spec(self):
        return self.envs[0].action_spec()

    def discount_spec(self):
        return self.envs[0].discount_spec()

    def reset(self) -> None:
        for server in self.servers:
            server.send(("reset", None))
        return self._receive()

    def step(self, actions: Sequence[int]) -> dm_env.TimeStep:
        self._check_actions(actions)
        for a, server in zip(actions, self.servers):
            server.send(("step", a))
        return self._receive()

    def step_async(self, actions: Sequence[int], queue: mp.Queue) -> None:
        self._check_actions(actions)
        for a, server in zip(actions, self.servers):
            server.send(("step_async", (a, queue)))
        return

    def close(self) -> None:
        for server in self.servers:
            server.send(("close", None))
        return self._receive()

    def render(self, mode: str = "rgb_array"):
        for server in self.servers:
            server.send(("render", mode))
        return self._receive()

    def is_waiting(self):
        return any(server.poll() for server in self.servers)

    def _receive(self):
        return [server.recv() for server in self.servers]

    def _check_actions(self, actions: Sequence[int]):
        assert (
            len(actions) == self.n_actors
        ), "The number of actions must be equal to the number of parallel environments.\
            \nReceived {} actions for {} environments. ".format(
            len(actions), self.n_actors
        )
