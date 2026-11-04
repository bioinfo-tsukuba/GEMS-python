from __future__ import annotations
from gems_python.multi_machine_problem_interval_task.transition_manager import State, Experiment, Experiments
from datetime import datetime
from typing import List
import polars as pl

# Existing implementation
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task, TaskGroup, TaskStatus,
)
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty

# Import your State / Experiment / Experiments classes
# from your_module import State, Experiment, Experiments

print("Experiments repr:", Experiments)
print("Experiments module:", getattr(Experiments, "__module__", None))
print("Experiments bases:", getattr(Experiments, "__mro__", None))
print("Has __annotations__:", hasattr(Experiments, "__annotations__"))
print("Has __init__:", hasattr(Experiments, "__init__"))


# ---- Three States ----
class InitState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # Example: 15 minutes of setup on a general-purpose machine (machine_type=0)
        tg = TaskGroup(
            optimal_start_time=0,
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(processing_time=15, interval=0,
                     experiment_operation="Init: setup", optimal_machine_type=0)
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # Transition to MeasureState when init_ok=True is observed
        if "init_ok" in df.columns and df.filter(pl.col("init_ok") == True).height > 0:
            return "MeasureState"
        return "InitState"

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        return pl.DataFrame({
            "time": [datetime.now().isoformat()],
            "task_group_id": [task_group_id],
            "task_id": [task_id],
            "init_ok": [True],
            "state": [self.state_name],
        })


class MeasureState(State):
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # Example: 30 minutes on a reader (machine_type=1)
        tg = TaskGroup(
            optimal_start_time=30,  # Example target time near the end of setup
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(processing_time=30, interval=0,
                     experiment_operation="Measure: read value", optimal_machine_type=1)
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # Transition to FinishState when measurement >= threshold is observed
        if "measurement" in df.columns and df.filter(pl.col("measurement") >= self.threshold).height > 0:
            return "FinishState"
        return "MeasureState"

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        # Return a value above the threshold; use a lower value to test repeated measurement
        return pl.DataFrame({
            "time": [datetime.now().isoformat()],
            "task_group_id": [task_group_id],
            "task_id": [task_id],
            "measurement": [self.threshold + 0.5],
            "state": [self.state_name],
        })


class FinishState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # Example: 5 minutes of finalization on a general-purpose machine (machine_type=0)
        tg = TaskGroup(
            optimal_start_time=70,
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(processing_time=5, interval=0,
                     experiment_operation="Finish: finalise", optimal_machine_type=0)
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # Represent the final state by remaining here
        return "FinishState"

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        return pl.DataFrame({
            "time": [datetime.now().isoformat()],
            "task_group_id": [task_group_id],
            "task_id": [task_id],
            "finished": [True],
            "state": [self.state_name],
        })


# ---- Build the experiment ----
def build_experiment() -> Experiment:
    states: List[State] = [InitState(), MeasureState(threshold=1.0), FinishState()]
    exp = Experiment(
        experiment_name="DemoExperiment",
        states=states,
        current_state_name="InitState",
        shared_variable_history=pl.DataFrame(),
    )
    return exp


def build_experiments() -> Experiments:
    ml = MachineList()
    ml.add_machine(Machine(machine_type=0, description="General bench"))
    ml.add_machine(Machine(machine_type=1, description="Reader"))

    exps = Experiments(
        experiments=[build_experiment()],
        parent_dir_path="experiments_dir_demo",
        machine_list=ml,
    )
    return exps


# ---- Execution example ----
if __name__ == "__main__":
    exps = build_experiments()

    # Use existing methods to generate Gantt charts and schedules
    # exps.experiments[0].show_experiment_directed_graph(save_path="demo_graph.png")

    # Run 10 steps with dummy results and save each step
    logs = exps.simulate(max_steps=10, save_each_step=True)
    for i, r in enumerate(logs, 1):
        print(f"[step {i}] {r}")

    # Use True to save each step
    # exps.simulate(max_steps=3, save_each_step=True)
