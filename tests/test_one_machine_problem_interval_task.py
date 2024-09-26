from dataclasses import dataclass
from pathlib import Path
import unittest
import polars as pl
import inspect
import tempfile
import os

from gems_python.one_machine_problem_interval_task.penalty.penalty_class import NonePenalty
from gems_python.one_machine_problem_interval_task.task_info import Task, TaskGroup
from gems_python.one_machine_problem_interval_task.transition_manager import Experiments, Experiment, State

separate_line_length = 50

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
    

def gen_minimum_experiment(experiment_name = "minimum_experiment"):
    return Experiment(
        experiment_name=experiment_name,
        states=[
            MinimumState()
        ],
        current_state_name="MinimumState",
        shared_variable_history=pl.DataFrame()
    )

def gen_minimum_experiments(temp_dir: str, experiment_name = "minimum_experiment"):
    # Path is volatile
    return Experiments(
        experiments=[
            gen_minimum_experiment(experiment_name=experiment_name)
        ],
        parent_dir_path = Path(temp_dir)
    )



class TestClass(unittest.TestCase):
    def setUp(self):
        pass

    def test_task_group(self):
        task_group_1 = TaskGroup(
            optimal_start_time=0,
            penalty_type=NonePenalty(),
            tasks=[
                Task(processing_time=2, interval=0, experiment_operation="A"),
                Task(processing_time=3, interval=15, experiment_operation="B"),
                Task(processing_time=4, interval=20, experiment_operation="C")
            ]
        )

        task_group_2 = TaskGroup(
            optimal_start_time=2,
            penalty_type=NonePenalty(),
            tasks=[
                Task(processing_time=2, interval=0, experiment_operation="A"),
                Task(processing_time=3, interval=1, experiment_operation="B"),
                Task(processing_time=4, interval=2, experiment_operation="C")
            ]
        )

        task_groups = [task_group_1, task_group_2]


        text = "task_groups_bef"
        print(f"*"*int((separate_line_length-len(text))/2) + text + f"*"*int((separate_line_length-len(text))/2))
        for task_group in task_groups:
            print(task_group)
        print(f"*"*separate_line_length)

        task_groups = TaskGroup.set_task_group_ids(task_groups)


        text = "task_groups_aft"
        print(f"*"*int((separate_line_length-len(text))/2) + text + f"*"*int((separate_line_length-len(text))/2))
        for task_group in task_groups:
            print(task_group)

        print(f"*"*separate_line_length)

        task_groups = TaskGroup.schedule_task_groups(task_groups, reference_time=0)

        print(f"*"*int((separate_line_length-len(text))/2) + text + f"*"*int((separate_line_length-len(text))/2))
        for task_group in task_groups:
            print(task_group)
        print(f"*"*separate_line_length)

        for task_id in range(len(task_groups[0].tasks)):
            input(f"press enter to complete task {task_id}")
            task_groups = TaskGroup.complete_task(task_groups, group_id=0, task_id=task_id)

        print(f"{task_groups[0].to_json()=}")
        
        with tempfile.TemporaryDirectory() as dname:
            print(dname)                 
            lab: Experiments = gen_minimum_experiments(temp_dir=dname)
            print(lab)

            lab.save_all()

            input("press enter to continue")



        # experiment = Experiment(
        #     experiment_name="test", 
        #     states=[],
        #     current_state_name="",
        #     shared_variable_history=pl.DataFrame(),
        # )

        