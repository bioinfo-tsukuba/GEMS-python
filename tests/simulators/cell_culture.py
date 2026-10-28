"""Synthetic, deterministic passage and observation scenario."""

import math
from pathlib import Path

import polars as pl

from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task, TaskGroup,
)
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    Experiment, Experiments, State,
)


SEED_CONFLUENCE = 0.12
GROWTH_RATE_PER_HOUR = 0.035
PASSAGE_THRESHOLD = 0.30
OBSERVATION_INTERVAL_MINUTES = 360


def logistic_growth(initial: float, hours: float) -> float:
    """Exact logistic curve with unit carrying capacity."""
    return 1 / (1 + (1 / initial - 1) * math.exp(-GROWTH_RATE_PER_HOUR * hours))


class PassageState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        start = 0 if df.is_empty() else int(df["time"][-1])
        return TaskGroup(
            optimal_start_time=start,
            penalty_type=LinearPenalty(10),
            tasks=[Task(20, "passage", optimal_machine_type=0)],
        )

    def transition_function(self, df: pl.DataFrame) -> str:
        return "ObservationState"


class ObservationState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        start = int(df["time"][-1]) + OBSERVATION_INTERVAL_MINUTES
        return TaskGroup(
            optimal_start_time=start,
            penalty_type=LinearPenalty(10),
            tasks=[Task(10, "observe", optimal_machine_type=1)],
        )

    def transition_function(self, df: pl.DataFrame) -> str:
        if float(df["confluence"][-1]) >= PASSAGE_THRESHOLD:
            return "PassageState"
        return "ObservationState"


def observe(state, history: pl.DataFrame, group, task, event_time: int) -> pl.DataFrame:
    if isinstance(state, PassageState):
        value = SEED_CONFLUENCE
        event = "passage"
    elif isinstance(state, ObservationState):
        last_time = int(history["time"][-1])
        last_value = float(history["confluence"][-1])
        value = logistic_growth(last_value, (event_time - last_time) / 60)
        event = "observation"
    else:
        raise ValueError(f"Unsupported state: {state.state_name}")
    return pl.DataFrame({
        "time": [event_time],
        "confluence": [value],
        "event": [event],
        "task_group_id": [group.task_group_id],
        "task_id": [task.task_id],
    })


def build_experiments(directory: Path) -> Experiments:
    machines = MachineList([
        Machine(machine_type=0, machine_id="bench"),
        Machine(machine_type=1, machine_id="imager"),
    ])
    experiment = Experiment(
        experiment_name="SyntheticCellCulture",
        states=[PassageState(), ObservationState()],
        current_state_name="PassageState",
    )
    return Experiments(
        experiments=[experiment], machine_list=machines, parent_dir_path=directory
    )
