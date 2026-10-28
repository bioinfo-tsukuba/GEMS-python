"""Synthetic reaction-condition search scenario and baselines."""

from itertools import product
from pathlib import Path
import random

from gems_python.common import dataframes
from gems_python.common.dataframes import DataFrame

from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task, TaskGroup,
)
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    Experiment, Experiments, State,
)


CONDITIONS = tuple(product((40, 60, 80), (0.25, 0.50, 0.75), (30, 60, 90)))
BEST_POSSIBLE_YIELD = 1.0


def synthetic_yield(condition: tuple[int, float, int]) -> float:
    """Bounded quadratic response with a known optimum at (60, 0.5, 60)."""
    temperature, concentration, duration = condition
    return max(
        0.0,
        1.0
        - ((temperature - 60) / 40) ** 2
        - ((concentration - 0.5) / 0.5) ** 2
        - ((duration - 60) / 120) ** 2,
    )


def candidate_order(strategy: str, seed: int = 0) -> tuple:
    candidates = list(CONDITIONS)
    if strategy == "random":
        random.Random(seed).shuffle(candidates)
    elif strategy != "grid":
        raise ValueError(f"Unknown search strategy: {strategy}")
    return tuple(candidates)


def best_so_far(values: list[float]) -> list[float]:
    best = 0.0
    result = []
    for value in values:
        best = max(best, value)
        result.append(best)
    return result


class ReactionState(State):
    def __init__(self, candidates: tuple):
        super().__init__()
        self.candidates = candidates

    def task_generator(self, df: DataFrame) -> TaskGroup:
        condition = self.candidates[dataframes.row_count(df) % len(self.candidates)]
        last_time = 0 if dataframes.row_count(df) == 0 else int(dataframes.last_value(df, "time"))
        return TaskGroup(
            optimal_start_time=last_time,
            penalty_type=LinearPenalty(1),
            tasks=[
                Task(condition[2] + 10, "react and measure", optimal_machine_type=0),
            ],
        )

    def transition_function(self, df: DataFrame) -> str:
        return "ReactionState"


class ReactionObserver:
    def __init__(self, candidates: tuple):
        self.candidates = candidates

    def __call__(
        self, state, history: DataFrame, group, task, event_time: int
    ) -> DataFrame:
        condition = self.candidates[dataframes.row_count(history)]
        temperature, concentration, duration = condition
        return dataframes.make_frame({
            "time": [event_time],
            "temperature": [temperature],
            "concentration": [concentration],
            "duration": [duration],
            "yield": [synthetic_yield(condition)],
            "task_group_id": [group.task_group_id],
            "task_id": [task.task_id],
        }, backend=dataframes.backend_of(history))


def build_experiments(directory: Path, candidates: tuple, dataframe_backend: str = "auto") -> Experiments:
    machines = MachineList([
        Machine(machine_type=0, machine_id="reactor"),
    ])
    experiment = Experiment(
        experiment_name="SyntheticReactionSearch",
        states=[ReactionState(candidates)],
        current_state_name="ReactionState",
        dataframe_backend=dataframe_backend,
        experiment_uuid="synthetic-reaction-search",
    )
    return Experiments(
        experiments=[experiment], machine_list=machines, parent_dir_path=directory
    )
