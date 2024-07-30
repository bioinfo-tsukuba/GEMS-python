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
        return abs(diff * self.penalty_coefficient)


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
    def task_generator(self, df: pl.DataFrame) -> Tuple[str, int, Type[PenaltyType]]:
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
        parent_dir_path (Path): The parent directory path of the experiment.

    Note:
        parent_dir_path is required to save the experiment data.
        Normally, parent_dir_path/uuid/ is used to save the experiment data.
    """
    experiment_name: str
    states: List[Type[State]]
    current_state_index: int
    shared_variable_history: pl.DataFrame  # mutability is required
    parent_dir_path: Path # The parent directory path of the experiment
    experiment_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_task: Tuple[str, int, Type[PenaltyType]] = field(default=None)

    def __post_init__(self):
        """
        Initialize the experiment.
        """
        # Create the parent directory if it does not exist
        self.parent_dir_path.mkdir(parents=True, exist_ok=True)
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
    


    def execute_one_step(self) -> Tuple[str, int, type[PenaltyType]]:
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
    

    def generate_task_of_the_state(self) -> Tuple[str, int, Type[PenaltyType]]:
        """
        Generate a task of the state.
        :return: Generated task.
        """
        # Generate a task
        state_index = self.current_state_index
        try:
            task = self.states[state_index].task_generator(self.shared_variable_history.clone(), self.experiment_name, self.experiment_uuid)
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
 

# テスト
def test_transition_manager():
    df = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    manager = TransitionManager(transition_function)
    result = manager.determine_next_state_index(df)
    print(f"Next state index: {result}")

if __name__ == "__main__":
    test_transition_manager()
