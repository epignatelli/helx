from typing import Deque, Callable, NamedTuple

from ..typing import factory

class SchedulerState(NamedTuple):
    value: float
    metric_history: Deque[float]


class Scheduler(NamedTuple):
    step: Callable[[float, SchedulerState], SchedulerState]
    state: SchedulerState


def scheduler(fun):
    return factory(fun, Scheduler)


