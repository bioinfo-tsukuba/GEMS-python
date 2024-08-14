from dataclasses import dataclass
from .base import PenaltyType


@dataclass
class NonePenalty(PenaltyType):
    """
    クラス：NonePenalty
    ペナルティなしのタイプ。常にペナルティは0です。
    """

    def calculate_penalty(self, scheduled_timing: int, optimal_timing: int) -> int:
        return 0
