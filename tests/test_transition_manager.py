from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Type
import unittest
import polars as pl
import inspect
from gems_python.transition_manager import Experiment, Experiments, LinearPenalty, NonePenalty, OneMachineTask, OneMachineTaskLocalInformation, PenaltyType, State, CyclicalRestPenaltyWithLinear, LinearWithRange

@dataclass
class MyState1(State):

    def transition_function(self, df: pl.DataFrame) -> str:
        # 簡単な実装例として、DataFrameの行数を返す
        return "MyState2"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        # 簡単な実装例として、タスク名、行数、最初の列のデータを返す
        task_name = "example_task"
        task_count = df.height
        pelalty_type = LinearPenalty(10)
        return OneMachineTaskLocalInformation(
            optimal_timing=0,
            processing_time=1,
            penalty_type=pelalty_type,
            experiment_operation=task_name,
        )
    

@dataclass
class MyState2(State):

    def transition_function(self, df: pl.DataFrame) -> str:
        # 簡単な実装例として、DataFrameの行数を返す
        return "MyState1"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        # 簡単な実装例として、タスク名、行数、最初の列のデータを返す
        task_name = "example_task"
        task_count = df.height
        pelalty_type = LinearWithRange(lower=-20, upper=40, lower_coefficient=10, upper_coefficient=20)
        return OneMachineTaskLocalInformation(
            optimal_timing=0,
            processing_time=1,
            penalty_type=pelalty_type,
            experiment_operation=task_name,
        )
    

@dataclass
class MyState__1(State):

    def transition_function(self, df: pl.DataFrame) -> str:
        # 簡単な実装例として、DataFrameの行数を返す
        return "MyState__1"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        # 簡単な実装例として、タスク名、行数、最初の列のデータを返す
        task_name = "example_task"
        task_count = df.height
        pelalty_type = LinearPenalty(10)
        return OneMachineTaskLocalInformation(
            optimal_timing=0,
            processing_time=1,
            penalty_type=pelalty_type,
            experiment_operation=task_name,
        )

@dataclass
class MyState__2(State):

    def transition_function(self, df: pl.DataFrame) -> str:
        # 簡単な実装例として、DataFrameの行数を返す
        return "MyState__2"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        # 簡単な実装例として、タスク名、行数、最初の列のデータを返す
        task_name = "example_task"
        task_count = df.height
        pelalty_type = NonePenalty()
        return OneMachineTaskLocalInformation(
            optimal_timing=0,
            processing_time=1,
            penalty_type=pelalty_type,
            experiment_operation=task_name,
        )

@dataclass
class MyExperiment1(Experiment):
    """
    TODO: 継承元のクラス名を表示できるのであれば行いたい
    ベースとなるプロトコル→今のプロトコル　のようにしたい
    """
@dataclass
class MyExperiment2(Experiment):
    """
    TODO: 継承元のクラス名を表示できるのであれば行いたい
    ベースとなるプロトコル→今のプロトコル　のようにしたい
    """


class TestTransitionManager(unittest.TestCase):
    def setUp(self):
        # テスト用のDataFrameを作成
        self.df = pl.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        self.state = MyState1()
        print(self.state)

        self.experiment = MyExperiment1(
            "test1",
            [MyState1(), MyState2()],
            self.state.state_name,
            self.df,
        )
        print(self.experiment)


        experiment2 = MyExperiment2(
            "test2",
            [MyState__1(), MyState__2()],
            "MyState__2",
            self.df,
        )

        e = Experiments(
            [self.experiment, experiment2],
            parent_dir_path=Path("tests/test_save_all")
            )
        e.save_all()


    def test_transition_function(self):
        result = self.state.transition_function(self.df)
        print(f"{result=}")

    def test_task_generator(self):
        task_name, task_count, penalty = self.state.task_generator(self.df)
        pena_json = penalty.to_json()
        print("pena", penalty.to_json())
        print("pena", PenaltyType.from_json(pena_json))
        self.assertEqual(task_name, "example_task")
        self.assertEqual(task_count, 3)
        print(penalty)



class TestScheduler(unittest.TestCase):
    def test_simulated_annealing_schedule_multiple_tasks(self):
        tasks = [
            OneMachineTask(optimal_timing=1, processing_time=5, penalty_type=LinearPenalty(penalty_coefficient=10), experiment_operation="op1", experiment_name="exp1", experiment_uuid="uuid1", task_id=1),
            OneMachineTask(optimal_timing=3, processing_time=3, penalty_type=LinearPenalty(penalty_coefficient=5), experiment_operation="op2", experiment_name="exp2", experiment_uuid="uuid2", task_id=2),
            OneMachineTask(optimal_timing=5, processing_time=8, penalty_type=LinearPenalty(penalty_coefficient=8), experiment_operation="op3", experiment_name="exp3", experiment_uuid="uuid3", task_id=3),
            OneMachineTask(optimal_timing=5, processing_time=8, penalty_type=LinearPenalty(penalty_coefficient=8), experiment_operation="op4", experiment_name="exp3", experiment_uuid="uuid4", task_id=4),
            OneMachineTask(optimal_timing=90, processing_time=8, penalty_type=LinearPenalty(penalty_coefficient=8), experiment_operation="op5", experiment_name="exp3", experiment_uuid="uuid5", task_id=5),
            OneMachineTask(optimal_timing=50, processing_time=8, penalty_type=LinearPenalty(penalty_coefficient=8), experiment_operation="op6", experiment_name="exp3", experiment_uuid="uuid6", task_id=6),
            OneMachineTask(optimal_timing=30, processing_time=8, penalty_type=LinearPenalty(penalty_coefficient=8), experiment_operation="op7", experiment_name="exp3", experiment_uuid="uuid7", task_id=7),
            OneMachineTask(optimal_timing=20, processing_time=8, penalty_type=LinearPenalty(penalty_coefficient=8), experiment_operation="op8", experiment_name="exp3", experiment_uuid="uuid8", task_id=8),
            OneMachineTask(optimal_timing=60, processing_time=8, penalty_type=LinearPenalty(penalty_coefficient=8), experiment_operation="op9", experiment_name="exp3", experiment_uuid="uuid9", task_id=9),
        ]

        scheduled_tasks = OneMachineTask.simulated_annealing_schedule(tasks)
        save_path = Path("tests")
        func_name = inspect.currentframe().f_code.co_name
        save_path = save_path / f"{func_name}.png"
        # OneMachineTask.vis()
        OneMachineTask.vis_with_diff(scheduled_tasks, save_path)

        for t in scheduled_tasks:
            print(t)


    
    def test_simulated_annealing_schedule_multiple_tasks_multiple_type(self):
        tasks = [
            OneMachineTask(optimal_timing=1, processing_time=5, penalty_type=LinearPenalty(penalty_coefficient=10), experiment_operation="op1", experiment_name="exp1", experiment_uuid="uuid1", task_id=1),
            OneMachineTask(optimal_timing=3, processing_time=3, penalty_type=CyclicalRestPenaltyWithLinear(cycle_start_time=0, cycle_duration=10, rest_time_ranges=[(0, 5), (10, 15)], penalty_coefficient=10), experiment_operation="op2", experiment_name="exp2", experiment_uuid="uuid2", task_id=2),
            OneMachineTask(optimal_timing=5, processing_time=8, penalty_type=LinearWithRange(lower=-20, upper=40, lower_coefficient=10, upper_coefficient=20), experiment_operation="op3", experiment_name="exp3", experiment_uuid="uuid3", task_id=3),
        ]

        scheduled_tasks = OneMachineTask.simulated_annealing_schedule(tasks)
        save_path = Path("tests")
        func_name = inspect.currentframe().f_code.co_name
        save_path = save_path / f"{func_name}.png"
        # OneMachineTask.vis()
        OneMachineTask.vis_with_diff(scheduled_tasks, save_path)


        for t in scheduled_tasks:
            print(t)


    def test_simulated_annealing_schedule_mono_task(self):
        tasks = [
            OneMachineTask(optimal_timing=5, processing_time=5, penalty_type=LinearPenalty(penalty_coefficient=10), experiment_operation="op1", experiment_name="exp1", experiment_uuid="uuid1", task_id=1),
        ]

        scheduled_tasks = OneMachineTask.simulated_annealing_schedule(tasks.copy())

        desired_tasks = [
            OneMachineTask(optimal_timing=5, processing_time=5, penalty_type=LinearPenalty(penalty_coefficient=10), experiment_operation="op1", experiment_name="exp1", experiment_uuid="uuid1", task_id=1, scheduled_timing = 5)
        ]

        assert(desired_tasks[0].scheduled_timing==scheduled_tasks[0].scheduled_timing)
        save_path = Path("tests")
        func_name = inspect.currentframe().f_code.co_name
        save_path = save_path / f"{func_name}.png"
        # OneMachineTask.vis()
        OneMachineTask.vis_with_diff(scheduled_tasks, save_path)

if __name__ == '__main__':
    unittest.main()
