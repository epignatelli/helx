from typing import Any
from .base import Environment


class FromMujocoEnv(Environment):
    """Static class to convert between mujoco environments and helx environments."""

    def __init__(self, env: Any):
        # TODO (epignatelli): Implement this
        raise NotImplementedError()
