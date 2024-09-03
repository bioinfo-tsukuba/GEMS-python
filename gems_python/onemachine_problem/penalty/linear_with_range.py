from dataclasses import dataclass
from .base import PenaltyType

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

    def calculate_penalty(self, scheduled_timing: int, optimal_time: int) -> int:
        diff = scheduled_timing - optimal_time
        if diff < self.lower:
            return (self.lower - diff) * self.lower_coefficient
        elif self.lower <= diff <= self.upper:
            return 0
        else:
            return (diff - self.upper) * self.upper_coefficient