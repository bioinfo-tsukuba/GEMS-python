
import json
from pathlib import Path
from gems_python.one_machine_problem_interval_task.transition_manager import Experiment, Experiments, State
from gems_python.common.class_dumper import auto_dataclass as dataclass
import polars as pl
from gems_python.one_machine_problem_interval_task.task_info import Task, TaskGroup
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import NonePenalty


@dataclass
class MinimumState(State):
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
    

    def to_dict(self):
        print("ans;casclmac")
        return {'state_name': self.state_name}

    @classmethod
    def from_dict(cls, data):
        return cls(state_name=data['state_name'])
    
    def to_json(self):
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)
    

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