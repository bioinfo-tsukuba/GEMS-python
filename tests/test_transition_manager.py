from dataclasses import dataclass
from typing import Any, Tuple, Type
import unittest
import polars as pl
from gems_python.transition_manager import NonePenalty, PenaltyType, State

@dataclass
class MyState(State):

    state_name = "MyState"
    state_index = 0

    def transition_function(self, df: pl.DataFrame) -> int:
        # 簡単な実装例として、DataFrameの行数を返す
        return df.height

    def task_generator(self, df: pl.DataFrame) -> Tuple[str, int, Type[PenaltyType]]:
        # 簡単な実装例として、タスク名、行数、最初の列のデータを返す
        task_name = "example_task"
        task_count = df.height
        pelalty_type = NonePenalty()
        return (task_name, task_count, pelalty_type)

class TestTransitionManager(unittest.TestCase):
    def setUp(self):
        # テスト用のDataFrameを作成
        self.df = pl.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        self.state = MyState()
        print(self.state)

    def test_transition_function(self):
        result = self.state.transition_function(self.df)
        self.assertEqual(result, 3)  # 行数が3であることを確認

    def test_task_generator(self):
        task_name, task_count, penalty = self.state.task_generator(self.df)
        pena_json = penalty.to_json()
        print("pena", penalty.to_json())
        print("pena", PenaltyType.from_json(pena_json))
        self.assertEqual(task_name, "example_task")
        self.assertEqual(task_count, 3)
        print(penalty)

if __name__ == '__main__':
    unittest.main()
