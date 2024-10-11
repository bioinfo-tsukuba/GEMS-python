
from pathlib import Path
from gems_python.one_machine_problem_interval_task.transition_manager import Experiment, Experiments, State
from gems_python.common.class_dumper import auto_dataclass as dataclass
import polars as pl
from gems_python.one_machine_problem_interval_task.task_info import Task, TaskGroup
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import NonePenalty


class MinimumState(State):
    def __init__(self):
        super().__init__()
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        return  TaskGroup(
            optimal_start_time=0,
            penalty_type=NonePenalty(),
            tasks=[
                Task(processing_time=2, interval=0, experiment_operation="A"),
                Task(processing_time=3, interval=15, experiment_operation="B"),
                Task(processing_time=4, interval=20, experiment_operation="C")
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        # return the state name
        return "MinimumState"
    

def gen_minimum_experiment(experiment_name = "minimum_experiment") -> Experiment:
    return Experiment(
        experiment_name=experiment_name,
        states=[
            MinimumState()
        ],
        current_state_name="MinimumState",
        shared_variable_history=pl.DataFrame()
    )

def gen_minimum_experiments(temp_dir: str, experiment_name = "minimum_experiment") -> Experiments:
    # Path is volatile
    return Experiments(
        experiments=[
            gen_minimum_experiment(experiment_name=experiment_name)
        ],
        parent_dir_path = Path(temp_dir)
    )