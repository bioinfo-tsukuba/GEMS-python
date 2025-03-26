

from pathlib import Path
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, Experiments, State
from gems_python.common.class_dumper import auto_dataclass as dataclass
import polars as pl
from gems_python.multi_machine_problem_interval_task.task_info import TaskGroup, Task, Machine
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import NonePenalty


"""
   processing_time: int  # タスクの処理時間
   experiment_operation: str
   optimal_machine_type: int # タスクの最適なマシンタイプ
   interval: int = field(default=0)        # タスク間のインターバル、最初のタスクにはインターバルはない
   task_status: TaskStatus = TaskStatus.NOT_STARTED  # タスクのステータス
   allocated_machine_id: int = field(default=None)  # タスクが割り当てられたマシンのID
   scheduled_time: int = field(default=None)  # タスクの開始時刻
   task_id: int = field(default=None)
"""
class MinimumState(State):
   def task_generator(self, df: pl.DataFrame) -> TaskGroup:
       return  TaskGroup(
           optimal_start_time=0,
           penalty_type=NonePenalty(),
           tasks=[
               Task(processing_time=2, interval=0, experiment_operation="A", optimal_machine_type = 0),
               Task(processing_time=3, interval=15, experiment_operation="B", optimal_machine_type = 0),
               Task(processing_time=4, interval=20, experiment_operation="C", optimal_machine_type = 0)
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


class StandardState1(State):
   def task_generator(self, df: pl.DataFrame) -> TaskGroup:
       return  TaskGroup(
           optimal_start_time=0,
           penalty_type=NonePenalty(),
           tasks=[
               Task(processing_time=2, interval=0, experiment_operation="A", optimal_machine_type = 0),
               Task(processing_time=3, interval=15, experiment_operation="B", optimal_machine_type = 1),
               Task(processing_time=4, interval=20, experiment_operation="C", optimal_machine_type = 1),
           ]
       )
  
   def transition_function(self, df: pl.DataFrame) -> str:
       # return the state name
       return "StandardState2"
  


class StandardState2(State):
   def task_generator(self, df: pl.DataFrame) -> TaskGroup:
       return  TaskGroup(
           optimal_start_time=0,
           penalty_type=NonePenalty(),
           tasks=[
               Task(processing_time=2, interval=0, experiment_operation="D", optimal_machine_type = 0),
               Task(processing_time=3, interval=15, experiment_operation="E", optimal_machine_type = 0),
               Task(processing_time=4, interval=20, experiment_operation="F", optimal_machine_type = 0),
           ]
       )
  
   def transition_function(self, df: pl.DataFrame) -> str:
       # return the state name
       return "StandardState1"
  


def gen_standard_experiment(experiment_name = "standard_experiment") -> Experiment:
   return Experiment(
       experiment_name=experiment_name,
       states=[
           MinimumState(),
           StandardState1(),
           StandardState2()
       ],
       current_state_name="StandardState1",
       shared_variable_history=
       pl.DataFrame({
           "time": [0],
           "temperature": [0],
           "pressure": [0]
       })
   )


def gen_standard_experiments(temp_dir: str, experiment_name = "standard_experiment") -> Experiments:
   # Path is volatile
   lab = Experiments(
       experiments=[
           gen_standard_experiment(experiment_name=experiment_name)
       ],
       parent_dir_path = Path(temp_dir)
   )


   lab.machine_list.add_machine(Machine(machine_type = 0))
   lab.machine_list.add_machine(Machine(machine_type = 1))
   lab.machine_list.add_machine(Machine(machine_type = 2))


   return lab