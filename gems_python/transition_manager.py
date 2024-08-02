from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import json
from typing import Callable, Any, List, Optional, Tuple, Type, Union
import uuid
import polars as pl
from pathlib import Path

"""MODULE: Penalty
"""

# 最大ペナルティ値
PENALTY_MAXIMUM = 1000000  # 適切な値に置き換えてください

class PenaltyType(ABC):
    """
    抽象基底クラス:PenaltyType
    各種ペナルティタイプの基底クラスであり、calculate_penaltyメソッドを実装する必要があります。
    """

    @abstractmethod
    def calculate_penalty(self, scheduled_timing: int, optimal_timing: int) -> int:
        """
        ペナルティを計算する抽象メソッド
        :param scheduled_timing: スケジュールされたタイミング
        :param optimal_timing: 最適なタイミング
        :return: 計算されたペナルティ
        """
        pass

    def to_dict(self) -> dict:
        """
        Converts the penalty object to a dictionary.
        :return: Dictionary representation of the penalty object.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PenaltyType':
        """
        Creates a penalty object from a dictionary.
        :param data: Dictionary containing the penalty data.
        :return: Penalty object.
        """
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    def to_json(self) -> str:
        """
        Converts the penalty object to a JSON string.
        :return: JSON string representation of the penalty object.
        """
        data = self.to_dict()
        data["penalty_type"] = self.__class__.__name__
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'PenaltyType':
        """
        Creates a penalty object from a JSON string.
        :param json_str: JSON string containing the penalty data.
        :return: Penalty object.
        """
        data = json.loads(json_str)
        penalty_type = data.pop("penalty_type")

        try:
            penalty_class = globals()[penalty_type]
        except KeyError:
            raise ValueError(f"Invalid penalty type: {penalty_type}")
        
        return penalty_class.from_dict(data)

@dataclass
class NonePenalty(PenaltyType):
    """
    クラス：NonePenalty
    ペナルティなしのタイプ。常にペナルティは0です。
    """

    def calculate_penalty(self, scheduled_timing: int, optimal_timing: int) -> int:
        return 0

@dataclass
class LinearPenalty(PenaltyType):
    """
    Class: LinearPenalty
    Linear penalty type. Calculates penalty as abs(diff * penalty_coefficient).
    :param penalty_coefficient: Coefficient to multiply with the difference.
    """
    penalty_coefficient: int

    def calculate_penalty(self, scheduled_timing: int, optimal_timing: int) -> int:
        diff = scheduled_timing - optimal_timing
        return abs(diff * self.penalty_coefficient)


@dataclass
class LinearWithRange(PenaltyType):
    """
    Class: LinearWithRange
    Range-based linear penalty type. Returns 0 if within range, otherwise calculates penalty.
    :param lower: Lower bound of the range.
    :param lower_coefficient: Coefficient for penalties below the range.
    :param upper: Upper bound of the range.
    :param upper_coefficient: Coefficient for penalties above the range.
    """
    lower: int
    lower_coefficient: int
    upper: int
    upper_coefficient: int

    def calculate_penalty(self, scheduled_timing: int, optimal_timing: int) -> int:
        diff = scheduled_timing - optimal_timing
        if diff < self.lower:
            return (self.lower - diff) * self.lower_coefficient
        elif self.lower <= diff <= self.upper:
            return 0
        else:
            return (diff - self.upper) * self.upper_coefficient
        

@dataclass
class CyclicalRestPenalty(PenaltyType):
    """
    Class: CyclicalRestPenalty
    Penalty type for cyclical rest periods. Returns maximum penalty during rest periods.
    :param cycle_start_time: Start time of the cycle in minutes.
    :param cycle_duration: Length of the cycle in minutes.
    :param rest_time_ranges: List of (start, end) tuples defining rest periods within the cycle.

    Example:
    CyclicalRestPenalty(cycle_start_time=0, cycle_duration=60, rest_time_ranges=[(15, 30)])
    This example defines a 60-minute cycle starting from 0 minute, with a rest period from 15 to 30 minutes.
    """
    cycle_start_time: int
    cycle_duration: int
    rest_time_ranges: List[Tuple[int, int]]

    def calculate_penalty(self, scheduled_timing: int, optimal_timing: int) -> int:
        diff = scheduled_timing - self.cycle_start_time
        if diff < 0:
            # Scheduled time is before the cycle start
            return 0
        diff %= self.cycle_duration
        for start, end in self.rest_time_ranges:
            if start <= diff <= end:
                # Rest period
                return PENALTY_MAXIMUM
        # Thank you for your hard work.
        return 0
        

@dataclass
class CyclicalRestPenaltyWithLinear(PenaltyType):
    """
    Class: CyclicalRestPenaltyWithLinear
    Penalty type for cyclical rest periods with linear penalty outside rest periods.
    :param cycle_start_time: Start time of the cycle in minutes.
    :param cycle_duration: Length of the cycle in minutes.
    :param rest_time_ranges: List of (start, end) tuples defining rest periods within the cycle.
    :param penalty_coefficient: Coefficient for linear penalty outside rest periods.
    """
    cycle_start_time: int
    cycle_duration: int
    rest_time_ranges: List[Tuple[int, int]]
    penalty_coefficient: int

    def calculate_penalty(self, scheduled_timing: int, optimal_timing: int) -> int:
        diff = scheduled_timing - self.cycle_start_time
        if diff < 0:
            # Scheduled time is before the cycle start
            return 0
        diff %= self.cycle_duration
        for start, end in self.rest_time_ranges:
            if start <= diff <= end:
                return PENALTY_MAXIMUM
        # Thank you for your hard work.
        return abs(scheduled_timing - optimal_timing) * self.penalty_coefficient

class OneMachineTaskLocalInformation:
    """
    OneMachineTaskLocalInformation class.
    This class is used as a data class for the task.
    """
    optimal_timing: int
    processing_time: int
    penalty_type: Type[PenaltyType]
    experiment_operation: str

from simanneal import Annealer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
@dataclass
class OneMachineTask:
    """
    OneMachineTask class.
    This class is used as a data class for the task.
    """
    optimal_timing: int
    processing_time: int
    penalty_type: Type[PenaltyType]
    experiment_operation: str
    experiment_name: str
    experiment_uuid: str
    task_id: int = field(default=None)
    scheduled_timing: int = field(default=None)

    def to_dict(self) -> dict:
        """
        Converts the task object to a dictionary.
        :return: Dictionary representation of the task object.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'OneMachineTask':
        """
        Creates a task object from a dictionary.
        :param data: Dictionary containing the task data.
        :return: Task object.
        """
        return cls(**data)

    def to_json(self) -> str:
        """
        Converts the task object to a JSON string.
        :return: JSON string representation of the task object.
        """
        data = self.to_dict()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'OneMachineTask':
        """
        Creates a task object from a JSON string.
        :param json_str: JSON string containing the task data.
        :return: Task object.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def simulated_annealing_schedule(cls, tasks: List['OneMachineTask']) -> List['OneMachineTask']:
        """
        Simulated annealing scheduler.
        :param tasks: List of tasks to schedule.
        :return: Scheduled tasks.
        """
        print("SA_schedule:")
        class TaskAnnealer(Annealer):

            def __init__(self, state):
                super().__init__(state)
                self.step_count: int = 0

            def move(self):
                self.step_count += 1

                """
                random move or swap
                """
                temp = max(1, int(self.step_count/self.steps * (self.Tmax - self.Tmin) + self.Tmin))
                for i in range(min(temp, len(self.state))):
                  a = random.randint(0, len(self.state) - 1)
                  self.state[a].scheduled_timing += random.randint(-int(temp), int(temp))
                  self.state[a].scheduled_timing = max(0, self.state[a].scheduled_timing)

                if random.random() < 0.5:
                    a = random.randint(0, len(self.state) - 1)
                    b = random.randint(0, len(self.state) - 1)
                    self.state[a].scheduled_timing, self.state[b].scheduled_timing = self.state[b].scheduled_timing, self.state[a].scheduled_timing

            def energy(self):
                """Calculates the total penalty for the current state."""
                total_penalty = 0
                for task in self.state:
                    total_penalty += task.penalty_type.calculate_penalty(
                        task.scheduled_timing, task.optimal_timing
                    )

                # Overlapping penalty
                sorted_tasks = sorted(self.state, key=lambda x: x.scheduled_timing)
                for i in range(len(sorted_tasks) - 1):
                    task1 = sorted_tasks[i]
                    task2 = sorted_tasks[i + 1]
                    overlap = task1.scheduled_timing + task1.processing_time - task2.scheduled_timing
                    if overlap > 0:
                        total_penalty += overlap * 100000
                return total_penalty

        # Initialize the tasks with some initial schedule (e.g., their optimal timings)
        time = 0
        tasks = sorted(tasks, key=lambda x: x.optimal_timing + x.processing_time)
        for task in tasks:
            task.scheduled_timing = max(time, task.optimal_timing)
            time = task.scheduled_timing + task.processing_time

        # Create an instance of the annealer with the initial state
        annealer = TaskAnnealer(tasks)
        # Set the annealing parameters as needed
        annealer.steps = 100000
        annealer.Tmax = 25000.0
        annealer.Tmin = 1.0

        # Run the annealing process
        state, _ = annealer.anneal()

        return state


    @classmethod
    def vis(cls, tasks: List['OneMachineTask'], save_path: Path = None):
        """
        Visualizes the scheduled tasks as a Gantt chart.
            - tasks: List of scheduled tasks to visualize.
            - save_path: Path to save the visualization. If None, the visualization is shown.
        """
        fig, ax = plt.subplots()
        tasks = sorted(tasks, key=lambda x: x.scheduled_timing)

        # ガントチャートのデータを準備
        for task in tasks:
            start_time = task.scheduled_timing
            end_time = start_time + task.processing_time
            ax.barh(task.experiment_operation, end_time - start_time, left=start_time, edgecolor='black')

        # 軸のラベルを設定
        ax.set_xlabel('Time')
        ax.set_ylabel('Task')

        # time軸の最大値と最小値を設定
        max_time = max(task.scheduled_timing + task.processing_time for task in tasks)
        max_time = max(max(task.optimal_timing + task.processing_time for task in tasks), max_time) + 5
        min_time = min(task.scheduled_timing for task in tasks)
        min_time = min(min(task.optimal_timing for task in tasks), min_time) - 5
        ax.set_xlim(min_time, max_time)

        # タスクごとに異なる色を設定
        colors = plt.cm.get_cmap('tab10', len(tasks))
        for i, task in enumerate(tasks):
            start_time = task.scheduled_timing
            end_time = start_time + task.processing_time
            ax.barh(task.experiment_operation, end_time - start_time, left=start_time, color=colors(i), edgecolor='black')

        # 凡例を設定
        patches = [mpatches.Patch(color=colors(i), label=task.experiment_operation) for i, task in enumerate(tasks)]
        plt.legend(handles=patches)

        # グラフを表示
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()


    @classmethod
    def vis_with_diff(cls, tasks: List['OneMachineTask'], save_path: Path = None):
        """
        Visualizes the scheduled tasks as a Gantt chart.
            - tasks: List of scheduled tasks to visualize.
            - save_path: Path to save the visualization. If None, the visualization is shown.
        """
        fig, ax = plt.subplots()
        tasks = sorted(tasks, key=lambda x: x.scheduled_timing)

        # ガントチャートのデータを準備
        for task in tasks:
            start_time = task.scheduled_timing
            end_time = start_time + task.processing_time
            optimal_start = task.optimal_timing
            ax.barh(task.experiment_operation, end_time - start_time, left=start_time, edgecolor='black', alpha=0.7)

            # optimal_timingとscheduled_timingの差を表す線を追加
            if start_time != optimal_start:
                ax.plot([optimal_start, start_time], [task.experiment_operation, task.experiment_operation], 'r--', linewidth=2)

        # 軸のラベルを設定
        ax.set_xlabel('Time')
        ax.set_ylabel('Task')

        # time軸の最大値と最小値を設定
        max_time = max(task.scheduled_timing + task.processing_time for task in tasks)
        max_time = max(max(task.optimal_timing + task.processing_time for task in tasks), max_time) + 5
        min_time = min(task.scheduled_timing for task in tasks)
        min_time = min(min(task.optimal_timing for task in tasks), min_time) - 5
        ax.set_xlim(min_time, max_time)

        # タスクごとに異なる色を設定
        colors = plt.cm.get_cmap('tab10', len(tasks))
        for i, task in enumerate(tasks):
            start_time = task.scheduled_timing
            end_time = start_time + task.processing_time
            ax.barh(task.experiment_operation, end_time - start_time, left=start_time, color=colors(i), edgecolor='black')

        # 凡例を設定
        patches = [mpatches.Patch(color=colors(i), label=task.experiment_operation) for i, task in enumerate(tasks)]
        plt.legend(handles=patches)

        # グラフを表示
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

"""MODULE: State
"""


@dataclass
class State(ABC):
    """State class.
    This class is used as just an superclass for the State class.
    Defining "State", you should inherit this class and implement the methods.
    Note:
    When inheriting this class, you should implement the following methods:
    - transition_function
    - task_generator
    
    Also, you shoud define the following fields in the subclass, before instantiating the class:
    - state_name: str
    - state_index: int
    """
    state_name: str = field(init=False)
    state_index: int = field(init=False)

    @abstractmethod
    def transition_function(self, df: pl.DataFrame) -> int:
        pass

    @abstractmethod
    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        pass

"""MODULE: Experiment
"""

@dataclass
class Experiment:
    """
    Experiment class.

    Attributes:
        experiment_name (str): The name of the experiment.
        states (List[State]): The states of the experiment.
        current_state_index (int): The current state index of the experiment.
        shared_variable_history (pl.DataFrame): The shared variable history of the experiment.
        experiment_uuid (str): The unique identifier of the experiment.
    """
    experiment_name: str
    states: List[Type[State]]
    current_state_index: int
    shared_variable_history: pl.DataFrame  # mutability is required
    experiment_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_task: OneMachineTask = field(default=None)

    def __post_init__(self):
        """
        Initialize the experiment.
        """
        if self.current_task is None:
            self.current_task = self.generate_task_of_the_state()

    def show_experiment_name_and_state_names(self):
        print(f"Experiment name: {self.experiment_name}")
        print("State names:")
        for state in self.states:
            print(f"  - {state.state_name}")

    def show_current_state_name(self):
        current_state = self.states[self.current_state_index]
        print(f"Current state: {current_state.state_name}")

    def get_current_state_name(self) -> str:
        return self.states[self.current_state_index].state_name
    


    def execute_one_step(self) -> OneMachineTask:
        """
        Execute one step. Determine the next state index and generate a task.
        Note: The shared variable history is updated.
        Note: The current state index is updated.
        Note: The task is generated by the task generator of the next state.
        :return: Generated task.
        """
        # Determine the next state index
        try:
            next_state_index = self.determine_next_state_index()
        except Exception as err:
            raise RuntimeError(f"Error determining the next state index: {err}")

        # Update the current state index
        self.current_state_index = next_state_index
        
        # Generate a task
        try:
            task = self.generate_task_of_the_state()
        except Exception as err:
            raise RuntimeError(f"Error generating task: {err}")

        return task
    

    def generate_task_of_the_state(self) -> OneMachineTask:
        """
        Generate a task of the state.
        :return: Generated task.
        """
        # Generate a task
        state_index = self.current_state_index
        try:
            task_local_information = self.states[state_index].task_generator(self.shared_variable_history.clone())
            task = OneMachineTask(
                optimal_timing=task_local_information.optimal_timing,
                processing_time=task_local_information.processing_time,
                penalty_type=task_local_information.penalty_type,
                experiment_operation=task_local_information.experiment_operation,
                experiment_name=self.experiment_name,
                experiment_uuid=self.experiment_uuid,
            )
        except Exception as err:
            raise RuntimeError(f"Error generating task: {err}")

        return task
    
    def determine_next_state_index(self) -> int:
        """
        Determine the next state index.
        :return: Next state index.
        """
        # Determine the next state index
        try:
            next_state_index: int = self.states[self.current_state_index].transition_function(self.shared_variable_history)
        except Exception as err:
            raise RuntimeError(f"Error state transition: {err}")

        return next_state_index
 


"""MODULE: Experiments
"""

@dataclass
class Experiments:
    
    """
    Experiments class.
    This class is used as a data class for the experiments.
    """

    experiments: List[Experiment]
    parent_dir_path: Path
    # Automatically generated fields, not accept user input
    tasks: List[OneMachineTask] = field(default=None)

    def __post_init__(self):
        """
        Initialize the experiments.
        """
        if self.tasks is None:
            self.tasks = list()
            for experiment in self.experiments:
                self.tasks.append(experiment.current_task.copy())

            for index in range(len(self.experiments)):
                self.tasks[index].task_id = index

# テスト
def test_transition_manager():
    df = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    manager = TransitionManager(transition_function)
    result = manager.determine_next_state_index(df)
    print(f"Next state index: {result}")

if __name__ == "__main__":
    test_transition_manager()
