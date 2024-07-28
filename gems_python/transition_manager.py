from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Any, Optional, Tuple
import polars as pl

# TODO: PenaltyTypeを実装する
# class PenaltyType

class State(ABC):
    @abstractmethod
    def transition_function(self, df: pl.DataFrame) -> int:
        pass

    @abstractmethod
    def task_generator(self, df: pl.DataFrame) -> Tuple[str, int, Any]:
        pass

# テスト
def test_transition_manager():
    df = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    manager = TransitionManager(transition_function)
    result = manager.determine_next_state_index(df)
    print(f"Next state index: {result}")

if __name__ == "__main__":
    test_transition_manager()
