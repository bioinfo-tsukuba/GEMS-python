from dataclasses import dataclass
from .base import PenaltyType


@dataclass
class NonePenalty(PenaltyType):
    """
    クラス：NonePenalty
    ペナルティなしのタイプ。常にペナルティは0です。
    """

    def calculate_penalty(self, scheduled_time: int, optimal_time: int) -> int:
        return 0
