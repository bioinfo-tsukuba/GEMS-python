"""Importable objects used by persistence tests, including fresh processes."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from gems_python.common.class_dumper import auto_dataclass
from tests.simulators.dataframe_comparison import TimedState
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment


class Choice(Enum):
    FIRST = 1
    SECOND = 2


@dataclass(frozen=True, slots=True)
class Leaf:
    choice: Choice
    ranges: tuple[tuple[int, int], ...]


@auto_dataclass
class Bundle:
    values: dict[Any, Any]
    path: Path
    absent: Any = None
    progress: int = field(default=0, init=False)


class ConfiguredState(TimedState):
    constructor_calls = 0

    def __init__(self, duration, machine_type, first_start, threshold):
        type(self).constructor_calls += 1
        super().__init__(duration, machine_type, first_start)
        self.threshold = threshold
        self.counter = 0

    def transition_function(self, df):
        self.counter += 1
        return "ConfiguredState"


class ConfiguredExperiment(Experiment):
    constructor_calls = 0

    def __init__(self, threshold, backend):
        type(self).constructor_calls += 1
        super().__init__(
            "configured", [ConfiguredState(3, 0, 7, threshold)], "ConfiguredState",
            experiment_uuid="configured", dataframe_backend=backend,
        )
        self.threshold = threshold
