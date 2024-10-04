import unittest
import polars as pl
import inspect
import tempfile
import os

from gems_python.one_machine_problem_interval_task.penalty.penalty_class import NonePenalty
from gems_python.one_machine_problem_interval_task.task_info import Task, TaskGroup
from gems_python.one_machine_problem_interval_task.transition_manager import Experiments
from tests.experiment_samples.minimum import gen_minimum_experiments

separate_line_length = 50

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

        print(f"{inspect.getfile(TaskGroup)=}")
        print(f"{inspect.getfile(task_group_1.__class__)=}")

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

        j = task_groups[0].to_json()
        task_group = TaskGroup.from_json(j)
        print(f"{task_group=}")



    def test_experiment_save(self):
        with tempfile.TemporaryDirectory() as dname:
            print(dname)                 
            lab: Experiments = gen_minimum_experiments(temp_dir="volatile")
            print(f"{lab=}")
            
            dict_dumped = lab.to_dict()
            json_dumped = lab.to_json()
            print(f"{json_dumped=}")

            # Save to file
            with open(os.path.join(dname, "experiment.json"), "w") as f:
                f.write(json_dumped)

            # # Load from file
            # lab2 = Experiments.from_dict(dict_dumped)
            # print(f"{lab2=}")
            # lab2 = Experiments.from_json(json_dumped)
            # print(f"{lab2=}")

            # input("press enter to continue")

    def test_experiment_loop(self):
        with tempfile.TemporaryDirectory() as dname:
            lab = gen_minimum_experiments(temp_dir=dname)
            print(lab)
            lab.execute_scheduling()
            print(lab)

            earliest_task, earliest_group_id = TaskGroup.get_ealiest_task_in_task_groups(lab.task_groups)
            print(f"{earliest_task=}")
            print(f"all {lab.task_groups=}")

            for i in range(10):
                print("*"*separate_line_length, f"iteration {i}", "*"*separate_line_length)
                task_groups_before = lab.task_groups.copy()
                for task_group in task_groups_before:
                    print("#"* (separate_line_length//2), "Task groups before", "#"* (separate_line_length//2))
                    print(f"{task_group.task_group_id=}")
                    for task in task_group.tasks:
                        print(f"{task=}")
                shared_variable_history = pl.DataFrame()

                earliest_task_group, earliest_task = lab.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
                    task_group_id=earliest_group_id,
                    task_id=earliest_task.task_id,
                    new_result_of_experiment=shared_variable_history
                )

                earliest_group_id = earliest_task_group.task_group_id

           

                for task_group in lab.task_groups:
                    print("#"* (separate_line_length//2), "Task groups after", "#"* (separate_line_length//2))
                    print(f"{task_group.task_group_id=}")
                    for task in task_group.tasks:
                        print(f"{task=}")

            


        