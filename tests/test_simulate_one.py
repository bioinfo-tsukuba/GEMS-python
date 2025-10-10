from __future__ import annotations

from datetime import datetime
from typing import List

import polars as pl

from gems_python.one_machine_problem_interval_task.transition_manager import (
    State,
    Experiment,
    Experiments,
)
from gems_python.one_machine_problem_interval_task.task_info import (
    Task,
    TaskGroup,
)
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import (
    LinearPenalty,
)

# デバッグ用にクラスのメタ情報を確認
print("Experiments repr:", Experiments)
print("Experiments module:", getattr(Experiments, "__module__", None))
print("Experiments bases:", getattr(Experiments, "__mro__", None))
print("Has __annotations__:", hasattr(Experiments, "__annotations__"))
print("Has __init__:", hasattr(Experiments, "__init__"))


# ---- 3つの State ----
class InitState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # 例: 15分の準備作業
        tg = TaskGroup(
            optimal_start_time=0,
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(
                    processing_time=15,
                    interval=0,
                    experiment_operation="Init: setup",
                )
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # init_ok=True を観測したら MeasureState へ
        if "init_ok" in df.columns and df.filter(pl.col("init_ok") == True).height > 0:
            return "MeasureState"
        return "InitState"

    def dummy_output(
        self, df: pl.DataFrame, task_group_id: int, task_id: int
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [datetime.now().isoformat()],
                "task_group_id": [task_group_id],
                "task_id": [task_id],
                "init_ok": [True],
                "state": [self.state_name],
            }
        )


class MeasureState(State):
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # 例: 30分の測定
        tg = TaskGroup(
            optimal_start_time=30,  # 準備が終わる頃を狙う例
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(
                    processing_time=30,
                    interval=0,
                    experiment_operation="Measure: read value",
                )
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # measurement >= threshold を観測したら FinishState
        if "measurement" in df.columns and df.filter(pl.col("measurement") >= self.threshold).height > 0:
            return "FinishState"
        return "MeasureState"

    def dummy_output(
        self, df: pl.DataFrame, task_group_id: int, task_id: int
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [datetime.now().isoformat()],
                "task_group_id": [task_group_id],
                "task_id": [task_id],
                "measurement": [self.threshold + 0.5],
                "state": [self.state_name],
            }
        )


class FinishState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # 例: 5分の確定処理
        tg = TaskGroup(
            optimal_start_time=70,
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(
                    processing_time=5,
                    interval=0,
                    experiment_operation="Finish: finalise",
                )
            ],
        )
        return tg

    def transition_function(self, df: pl.DataFrame) -> str:
        # 終了状態のイメージ：ここに留まる
        return "FinishState"

    def dummy_output(
        self, df: pl.DataFrame, task_group_id: int, task_id: int
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [datetime.now().isoformat()],
                "task_group_id": [task_group_id],
                "task_id": [task_id],
                "finished": [True],
                "state": [self.state_name],
            }
        )


# ---- 実験の構築 ----
def build_experiment() -> Experiment:
    states: List[State] = [InitState(), MeasureState(threshold=1.0), FinishState()]
    exp = Experiment(
        experiment_name="DemoOneMachineExperiment",
        states=states,
        current_state_name="InitState",
        shared_variable_history=pl.DataFrame(),
    )
    return exp


def build_experiments() -> Experiments:
    exps = Experiments(
        experiments=[build_experiment()],
        parent_dir_path="experiments_dir_demo_one_machine",
    )
    return exps


# ---- 実行例 ----
if __name__ == "__main__":
    exps = build_experiments()

    # ガント・スケジュールなどは既存メソッドで生成可
    # exps.experiments[0].show_experiment_directed_graph(save_path="demo_graph_one_machine.png")

    # 保存も行いたい場合は save_each_step=True を指定
    exps.simulate(max_steps=3, save_each_step=True)
