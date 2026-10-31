from gems_python.common.class_dumper import SerializationMixin
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from .config import PENALTY_MAXIMUM

class PenaltyType(SerializationMixin, ABC):
    """
    Abstract base class: PenaltyType
    Base class for penalty types; subclasses must implement calculate_penalty.
    """

    @abstractmethod
    def calculate_penalty(self, scheduled_time: int, optimal_time: int) -> int:
        """
        Abstract method for calculating a penalty
        :param scheduled_time: Scheduled time
        :param optimal_time: Optimal time
        :return: Calculated penalty
        """
        pass


@dataclass
class NonePenalty(PenaltyType):
    """
    Class: NonePenalty
    No-penalty type; the penalty is always 0.
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