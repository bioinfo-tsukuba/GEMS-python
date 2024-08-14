from dataclasses import dataclass
from .base import PenaltyType

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
