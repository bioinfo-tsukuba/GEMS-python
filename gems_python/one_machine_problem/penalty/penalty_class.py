from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
import json
from typing import List, Tuple

from .config import PENALTY_MAXIMUM

class PenaltyType(ABC):
    """
    抽象基底クラス:PenaltyType
    各種ペナルティタイプの基底クラスであり、calculate_penaltyメソッドを実装する必要があります。
    """

    @abstractmethod
    def calculate_penalty(self, scheduled_time: int, optimal_time: int) -> int:
        """
        ペナルティを計算する抽象メソッド
        :param scheduled_time: スケジュールされたタイミング
        :param optimal_time: 最適なタイミング
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

    def calculate_penalty(self, scheduled_time: int, optimal_time: int) -> int:
        return 0

@dataclass
class LinearPenalty(PenaltyType):
    """
    Class: LinearPenalty
    Linear penalty type. Calculates penalty as abs(diff * penalty_coefficient).
    :param penalty_coefficient: Coefficient to multiply with the difference.
    """
    penalty_coefficient: int

    def calculate_penalty(self, scheduled_time: int, optimal_time: int) -> int:
        diff = scheduled_time - optimal_time
        return abs(diff * self.penalty_coefficient)

@dataclass
class LinearWithRangePenalty(PenaltyType):
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

    def calculate_penalty(self, scheduled_time: int, optimal_time: int) -> int:
        diff = scheduled_time - optimal_time
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

    def calculate_penalty(self, scheduled_time: int, optimal_time: int) -> int:
        diff = scheduled_time - self.cycle_start_time
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

    def calculate_penalty(self, scheduled_time: int, optimal_time: int) -> int:
        diff = scheduled_time - self.cycle_start_time
        if diff < 0:
            # Scheduled time is before the cycle start
            return 0
        diff %= self.cycle_duration
        for start, end in self.rest_time_ranges:
            if start <= diff <= end:
                return PENALTY_MAXIMUM
        # Thank you for your hard work.
        return abs(scheduled_time - optimal_time) * self.penalty_coefficient