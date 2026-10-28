"""Synthetic, deterministic passage and observation scenario."""

import math
from pathlib import Path

from gems_python.common import dataframes
from gems_python.common.dataframes import DataFrame

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
    def task_generator(self, df: DataFrame) -> TaskGroup:
        start = 0 if dataframes.row_count(df) == 0 else int(dataframes.last_value(df, "time"))
        return TaskGroup(
            optimal_start_time=start,
            penalty_type=LinearPenalty(10),
            tasks=[Task(20, "passage", optimal_machine_type=0)],
        )

    def transition_function(self, df: DataFrame) -> str:
        return "ObservationState"


class ObservationState(State):
    def task_generator(self, df: DataFrame) -> TaskGroup:
        start = int(dataframes.last_value(df, "time")) + OBSERVATION_INTERVAL_MINUTES
        return TaskGroup(
            optimal_start_time=start,
            penalty_type=LinearPenalty(10),
            tasks=[Task(10, "observe", optimal_machine_type=1)],
        )

    def transition_function(self, df: DataFrame) -> str:
        if float(dataframes.last_value(df, "confluence")) >= PASSAGE_THRESHOLD:
            return "PassageState"
        return "ObservationState"


def observe(state, history: DataFrame, group, task, event_time: int) -> DataFrame:
    if isinstance(state, PassageState):
        value = SEED_CONFLUENCE
        event = "passage"
    elif isinstance(state, ObservationState):
        last_time = int(dataframes.last_value(history, "time"))
        last_value = float(dataframes.last_value(history, "confluence"))
        value = logistic_growth(last_value, (event_time - last_time) / 60)
        event = "observation"
    else:
        raise ValueError(f"Unsupported state: {state.state_name}")
    return dataframes.make_frame({
        "time": [event_time],
        "confluence": [value],
        "event": [event],
        "task_group_id": [group.task_group_id],
        "task_id": [task.task_id],
    }, backend=dataframes.backend_of(history))


def build_experiments(directory: Path, dataframe_backend: str = "auto") -> Experiments:
    machines = MachineList([
        Machine(machine_type=0, machine_id="bench"),
        Machine(machine_type=1, machine_id="imager"),
    ])
    experiment = Experiment(
        experiment_name="SyntheticCellCulture",
        states=[PassageState(), ObservationState()],
        current_state_name="PassageState",
        dataframe_backend=dataframe_backend,
        experiment_uuid="synthetic-cell-culture",
    )
    return Experiments(
        experiments=[experiment], machine_list=machines, parent_dir_path=directory
    )
