from dataclasses import dataclass

from gems_python.onemachine_problem.penalty.config import PENALTY_MAXIMUM
from .base import PenaltyType
from dataclasses import dataclass
from typing import List, Tuple


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
