from typing import Any
from .base import IEnvironment


class FromMujocoEnv(IEnvironment):
    """Static class to convert between mujoco environments and helx environments."""

    def __init__(self, env: Any):
        # TODO (epignatelli): Implement this
        raise NotImplementedError()
