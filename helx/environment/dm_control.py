from typing import Any
from .base import IEnvironment


class FromDMControlEnv(IEnvironment):
    """Static class to convert between dm_control environments and helx environments."""

    def __init__(self, env: Any):
        # TODO (epignatelli): Implement this
        raise NotImplementedError()
