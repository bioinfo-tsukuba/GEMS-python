from datetime import datetime
import unittest
import polars as pl
import inspect
import tempfile
import os

from gems_python.one_machine_problem_interval_task.penalty.penalty_class import CyclicalRestPenaltyWithLinear, NonePenalty, CyclicalRestPenalty
from gems_python.one_machine_problem_interval_task.task_info import Task, TaskGroup
from gems_python.one_machine_problem_interval_task.transition_manager import Experiments
from tests.experiment_samples_one_interval_task.minimum import gen_minimum_experiments

separate_line_length = 50
UNIX_TIME_2024_11_14_00_00_00_JP_MIN = 1731510000//60

class TestSchedule(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_cyclical_rest_penalty(self):
        task_group_1 = TaskGroup(
            optimal_start_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN,
            penalty_type=CyclicalRestPenaltyWithLinear(
                cycle_start_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN,
                cycle_duration=60 * 24,
                rest_time_ranges=[(0, 10*60), (16*60, 24*60)],
                penalty_coefficient=10,
            ),
            tasks=[
                Task(processing_time=2, interval=0, experiment_operation="test_cyclical_rest_penaltyA"),
                Task(processing_time=3, interval=15, experiment_operation="test_cyclical_rest_penaltyB"),
                Task(processing_time=4, interval=20, experiment_operation="test_cyclical_rest_penaltyC")
            ]
        )

        task_groups = [task_group_1]
        schedule = TaskGroup.schedule_task_groups_simulated_annealing(task_groups, reference_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN)
        print(f"{schedule=}")

        scheduled_time = schedule[0].tasks[0].scheduled_time
        schedule_JP = datetime.fromtimestamp(scheduled_time*60).astimezone()
        print(f"{schedule_JP=}")


        self.assertLess(datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 10*60)*60).astimezone(), datetime.fromtimestamp(scheduled_time*60).astimezone())
        self.assertLess(datetime.fromtimestamp(scheduled_time*60).astimezone(), datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 16*60)*60).astimezone())

    
    def test_cyclical_rest_penalty2(self):
        task_group_1 = TaskGroup(
            optimal_start_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN - 10*60*24,
            penalty_type=CyclicalRestPenaltyWithLinear(
                cycle_start_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN,
                cycle_duration=60 * 24,
                rest_time_ranges=[(0, 10*60), (16*60, 24*60)],
                penalty_coefficient=10,
            ),
            tasks=[
                Task(processing_time=2, interval=0, experiment_operation="test_cyclical_rest_penalty2A"),
                Task(processing_time=3, interval=15, experiment_operation="test_cyclical_rest_penalty2B"),
                Task(processing_time=4, interval=20, experiment_operation="test_cyclical_rest_penalty2C")
            ]
        )

        task_groups = [task_group_1]
        schedule = TaskGroup.schedule_task_groups_simulated_annealing(task_groups, reference_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN)
        print(f"{schedule=}")

        scheduled_time = schedule[0].tasks[0].scheduled_time
        schedule_JP = datetime.fromtimestamp(scheduled_time*60).astimezone()
        print(f"{schedule_JP=}")


        self.assertLess(datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 10*60)*60).astimezone(), datetime.fromtimestamp(scheduled_time*60).astimezone())
        self.assertLess(datetime.fromtimestamp(scheduled_time*60).astimezone(), datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 16*60)*60).astimezone())


    
    def test_cyclical_rest_penalty2_(self):
        task_group_1 = TaskGroup(
            optimal_start_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN - 10*60*24,
            penalty_type=CyclicalRestPenaltyWithLinear(
                cycle_start_time=24*60-9*60,
                cycle_duration=60 * 24,
                rest_time_ranges=[(0, 10*60), (16*60, 24*60)],
                penalty_coefficient=10,
            ),
            tasks=[
                Task(processing_time=2, interval=0, experiment_operation="test_cyclical_rest_penalty2_A"),
                Task(processing_time=3, interval=15, experiment_operation="test_cyclical_rest_penalty2_B"),
                Task(processing_time=4, interval=20, experiment_operation="test_cyclical_rest_penalty2_C")
            ]
        )

        task_groups = [task_group_1]
        schedule = TaskGroup.schedule_task_groups_simulated_annealing(task_groups, reference_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN)
        print(f"{schedule=}")

        scheduled_time = schedule[0].tasks[0].scheduled_time
        schedule_JP = datetime.fromtimestamp(scheduled_time*60).astimezone()
        print(f"{schedule_JP=}")


        self.assertLess(datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 10*60)*60).astimezone(), datetime.fromtimestamp(scheduled_time*60).astimezone())
        self.assertLess(datetime.fromtimestamp(scheduled_time*60).astimezone(), datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 16*60)*60).astimezone())


    def test_cyclical_rest_penalty3(self):
        HEK_PROCESSING_TIME = {
            "HEKExpire": 0,
            "HEKPassage": 60, # < 60 min
            "HEKGetImage": 10, # < 10 min
            "HEKSampling": 40, # < 40 min
            "HEKWaiting": 0
        }
        optimal_time = UNIX_TIME_2024_11_14_00_00_00_JP_MIN
        optimal_start_time = int(optimal_time-HEK_PROCESSING_TIME["HEKSampling"]-HEK_PROCESSING_TIME["HEKGetImage"])
        task_group1 = TaskGroup(
            optimal_start_time=optimal_start_time,
            penalty_type=CyclicalRestPenaltyWithLinear(
                cycle_start_time = UNIX_TIME_2024_11_14_00_00_00_JP_MIN,
                cycle_duration=60 * 24,
                rest_time_ranges=[(0, 10*60), (16*60, 24*60)],
                penalty_coefficient = 1000
            ),
            # cycle_start_time: int
            # cycle_duration: int
            # rest_time_ranges: List[Tuple[int, int]]
            # penalty_coefficient: int
            tasks=[
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKGetImage"],
                    interval=1*60,
                    experiment_operation="test_cyclical_rest_penalty3HEKGetImage"
                ),
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKSampling"],
                    experiment_operation="HEKSampling"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                )
            ]
        )

        task_groups = [task_group1]
        schedule = TaskGroup.schedule_task_groups_simulated_annealing(task_groups, reference_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN)
        print(f"{schedule=}")

        scheduled_time = schedule[0].tasks[0].scheduled_time
        schedule_JP = datetime.fromtimestamp(scheduled_time*60).astimezone()
        print(f"{schedule_JP=}")

        self.assertLess(datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 10*60)*60).astimezone(), datetime.fromtimestamp(scheduled_time*60).astimezone())
        self.assertLess(datetime.fromtimestamp(scheduled_time*60).astimezone(), datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 16*60)*60).astimezone())

        

    def test_cyclical_rest_penalty4(self):
        HEK_PROCESSING_TIME = {
            "HEKExpire": 0,
            "HEKPassage": 60, # < 60 min
            "HEKGetImage": 10, # < 10 min
            "HEKSampling": 40, # < 40 min
            "HEKWaiting": 0
        }
        optimal_time = UNIX_TIME_2024_11_14_00_00_00_JP_MIN
        optimal_start_time = int(optimal_time-HEK_PROCESSING_TIME["HEKSampling"]-HEK_PROCESSING_TIME["HEKGetImage"])
        task_group1 = TaskGroup(
            optimal_start_time=optimal_start_time,
            penalty_type=CyclicalRestPenaltyWithLinear(
                cycle_start_time = UNIX_TIME_2024_11_14_00_00_00_JP_MIN - 10*60*24,
                cycle_duration=60 * 24,
                rest_time_ranges=[(0, 10*60), (16*60, 24*60)],
                penalty_coefficient = 1000
            ),
            # cycle_start_time: int
            # cycle_duration: int
            # rest_time_ranges: List[Tuple[int, int]]
            # penalty_coefficient: int
            tasks=[
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKGetImage"],
                    interval=1*60,
                    experiment_operation="test_cyclical_rest_penalty4HEKGetImage"
                ),
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKSampling"],
                    experiment_operation="HEKSampling"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                )
            ]
        )

        task_groups = [task_group1]
        schedule = TaskGroup.schedule_task_groups_simulated_annealing(task_groups, reference_time=UNIX_TIME_2024_11_14_00_00_00_JP_MIN)
        print(f"{schedule=}")

        scheduled_time = schedule[0].tasks[0].scheduled_time
        schedule_JP = datetime.fromtimestamp(scheduled_time*60).astimezone()
        print(f"{schedule_JP=}")


        self.assertLess(datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 10*60)*60).astimezone(), datetime.fromtimestamp(scheduled_time*60).astimezone())
        self.assertLess(datetime.fromtimestamp(scheduled_time*60).astimezone(), datetime.fromtimestamp((UNIX_TIME_2024_11_14_00_00_00_JP_MIN + 16*60)*60).astimezone())

        






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

        # task_groups = TaskGroup.schedule_task_groups(task_groups, reference_time=0)
        task_groups = TaskGroup.schedule_task_groups_simulated_annealing(task_groups, reference_time=0)
        TaskGroup.generate_gantt_chart(task_groups)


        print(f"*"*int((separate_line_length-len(text))/2) + text + f"*"*int((separate_line_length-len(text))/2))
        for task_group in task_groups:
            print(task_group)
        print(f"*"*separate_line_length)

        for task_id in range(len(task_groups[0].tasks)):
            input(f"press enter to complete task {task_id}")
            task_groups = TaskGroup.complete_task(task_groups, group_id=0, task_id=task_id)
            TaskGroup.generate_gantt_chart(task_groups)

        
        for task_id in range(len(task_groups[1].tasks)):
            input(f"press enter to complete task {task_id}")
            task_groups = TaskGroup.complete_task(task_groups, group_id=1, task_id=task_id)
            TaskGroup.generate_gantt_chart(task_groups)

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


            # TODO: Stateの再構築ができないので、コメントアウト
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

            


        