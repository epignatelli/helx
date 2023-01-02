from typing import Any
from .base import IEnvironment


class FromIvyGymEnv(IEnvironment):
    """Static class to convert between Ivy Gym environments and helx environments."""

    def __init__(self, env: Any):
        # TODO (epignatelli): Implement this
        raise NotImplementedError()
