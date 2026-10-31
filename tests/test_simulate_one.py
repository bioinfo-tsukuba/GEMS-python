from __future__ import annotations

from datetime import datetime
from typing import List

import polars as pl

from gems_python.one_machine_problem_interval_task.transition_manager import (
    State,
    Experiment,
    Experiments,
)
from gems_python.one_machine_problem_interval_task.task_info import (
    Task,
    TaskGroup,
)
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import (
    LinearPenalty,
)

# Inspect class metadata for debugging
print("Experiments repr:", Experiments)
print("Experiments module:", getattr(Experiments, "__module__", None))
print("Experiments bases:", getattr(Experiments, "__mro__", None))
print("Has __annotations__:", hasattr(Experiments, "__annotations__"))
print("Has __init__:", hasattr(Experiments, "__init__"))


# ---- Three States ----
class InitState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # Example: 15 minutes of setup
        tg = TaskGroup(
            optimal_start_time=0,
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(
                    processing_time=15,
                    interval=0,
                    experiment_operation="Init: setup",
                )
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # Transition to MeasureState when init_ok=True is observed
        if "init_ok" in df.columns and df.filter(pl.col("init_ok") == True).height > 0:
            return "MeasureState"
        return "InitState"

    def dummy_output(
        self, df: pl.DataFrame, task_group_id: int, task_id: int
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [datetime.now().isoformat()],
                "task_group_id": [task_group_id],
                "task_id": [task_id],
                "init_ok": [True],
                "state": [self.state_name],
            }
        )


class MeasureState(State):
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # Example: 30 minutes of measurement
        tg = TaskGroup(
            optimal_start_time=30,  # Example target time near the end of setup
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(
                    processing_time=30,
                    interval=0,
                    experiment_operation="Measure: read value",
                )
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # Transition to FinishState when measurement >= threshold is observed
        if "measurement" in df.columns and df.filter(pl.col("measurement") >= self.threshold).height > 0:
            return "FinishState"
        return "MeasureState"

    def dummy_output(
        self, df: pl.DataFrame, task_group_id: int, task_id: int
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [datetime.now().isoformat()],
                "task_group_id": [task_group_id],
                "task_id": [task_id],
                "measurement": [self.threshold + 0.5],
                "state": [self.state_name],
            }
        )


class FinishState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # Example: 5 minutes of finalization
        tg = TaskGroup(
            optimal_start_time=70,
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(
                    processing_time=5,
                    interval=0,
                    experiment_operation="Finish: finalise",
                )
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # Represent the final state by remaining here
        return "FinishState"

    def dummy_output(
        self, df: pl.DataFrame, task_group_id: int, task_id: int
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [datetime.now().isoformat()],
                "task_group_id": [task_group_id],
                "task_id": [task_id],
                "finished": [True],
                "state": [self.state_name],
            }
        )


# ---- Build the experiment ----
def build_experiment() -> Experiment:
    states: List[State] = [InitState(), MeasureState(threshold=1.0), FinishState()]
    exp = Experiment(
        experiment_name="DemoOneMachineExperiment",
        states=states,
        current_state_name="InitState",
        shared_variable_history=pl.DataFrame(),
    )
    return exp


def build_experiments() -> Experiments:
    exps = Experiments(
        experiments=[build_experiment()],
        parent_dir_path="experiments_dir_demo_one_machine",
    )
    return exps


# ---- Execution example ----
if __name__ == "__main__":
    exps = build_experiments()

    # Use existing methods to generate Gantt charts and schedules
    # exps.experiments[0].show_experiment_directed_graph(save_path="demo_graph_one_machine.png")

    # Use save_each_step=True to save each step
    exps.simulate(max_steps=3, save_each_step=True)
