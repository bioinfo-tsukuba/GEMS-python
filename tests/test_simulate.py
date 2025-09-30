from __future__ import annotations
from datetime import datetime
from typing import List
import polars as pl

# あなたの実装をインポート（ここは既存のまま）
from gems_python.multi_machine_problem_interval_task.task_info import Machine, MachineList, Task, TaskGroup
# ↑ Task/TaskGroup の API は環境差が出やすいので、下の make_one_task_group() の中だけ調整すればOK

from gems_python.multi_machine_problem_interval_task.transition_manager import State, Experiment, Experiments

# ========= ここから：差し替え用の TaskGroup 生成ヘルパ =========
def make_one_task_group(task_name: str, machine_type: int = 0, duration: int = 60) -> TaskGroup:
    """
    あなたの Task/TaskGroup 実装に合わせて、この関数だけ中身を調整してください。
    前提：TaskGroup は 1 個以上の Task を含み、後で
      - configure_task_group_settings(experiment_name, experiment_uuid)
      - is_completed(), task_group_id, experiment_uuid
      - 内部に Task（task_id を持つ）がある
    を満たす必要があります。
    例としてよくある API 形を想定したダミー実装を示します。
    """
    # --- 例：よくあるコンストラクタ形（あなたの実装に置き換え） ---
    # Task(duration=..., machine_type=..., name=...)
    t = Task(
        name=task_name,
        machine_type=machine_type,
        duration=duration,
    )
    tg = TaskGroup(tasks=[t])

    # 必要なら他のフィールドもここで初期化してください
    return tg
# ========= ここまで：差し替え用 =========


# ========= State 実装（dummy_output を含む） =========
class InitState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        return make_one_task_group("Init: setup", machine_type=0, duration=15)

    def transition_function(self, df: pl.DataFrame) -> str:
        # 直近の行で init_ok==True を見たら MeasureState へ
        if "init_ok" in df.columns and df.filter(pl.col("init_ok") == True).height > 0:
            return "MeasureState"
        return "InitState"

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        # 遷移関数が参照する列（init_ok）を入れる
        return pl.DataFrame({
            "time": [datetime.now().isoformat()],
            "task_group_id": [task_group_id],
            "task_id": [task_id],
            "init_ok": [True],
            "state": [self.state_name],
        })


class MeasureState(State):
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        return make_one_task_group("Measure: read value", machine_type=1, duration=30)

    def transition_function(self, df: pl.DataFrame) -> str:
        # 直近 measurement >= threshold を確認できたら FinishState
        if "measurement" in df.columns and df.filter(pl.col("measurement") >= self.threshold).height > 0:
            return "FinishState"
        return "MeasureState"

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        # 閾値を軽く超えるダミー値で「通す」
        # （敢えて試したいなら threshold-ε を入れて MeasureState に留まらせることも可能）
        return pl.DataFrame({
            "time": [datetime.now().isoformat()],
            "task_group_id": [task_group_id],
            "task_id": [task_id],
            "measurement": [self.threshold + 0.5],
            "state": [self.state_name],
        })


class FinishState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        return make_one_task_group("Finish: finalise", machine_type=0, duration=5)

    def transition_function(self, df: pl.DataFrame) -> str:
        # 終了状態のイメージ：ここに留まる
        return "FinishState"

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        return pl.DataFrame({
            "time": [datetime.now().isoformat()],
            "task_group_id": [task_group_id],
            "task_id": [task_id],
            "finished": [True],
            "state": [self.state_name],
        })
# ========= /State 実装 =========


# ========= 実験の組み立て =========
def build_experiment() -> Experiment:
    states: List[State] = [
        InitState(),
        MeasureState(threshold=1.0),
        FinishState(),
    ]

    # 初期 shared_variable_history は空で OK
    exp = Experiment(
        experiment_name="DemoExperiment",
        states=states,
        current_state_name="InitState",
        shared_variable_history=pl.DataFrame(),   # 必要なら初期行を入れてもよい
    )
    return exp


def build_experiments() -> Experiments:
    # マシン定義（あなたの環境の Machine/MachineList API に合わせて調整）
    ml = MachineList()
    ml.add_machine(Machine(machine_type=0, description="General bench"))
    ml.add_machine(Machine(machine_type=1, description="Reader"))

    exps = Experiments(
        experiments=[build_experiment()],
        parent_dir_path="experiments_dir_demo",
        machine_list=ml,
    )
    return exps
# ========= /組み立て =========


# ========= 実行例（シミュレーション） =========
if __name__ == "__main__":
    exps = build_experiments()

    # 現在の状態・遷移候補の可視化（任意）
    # exps.experiments[0].show_experiment_directed_graph(save_path="demo_graph.png")

    # 仮結果で 5 ステップ回す（ディスクへ保存しない）
    logs = exps.simulate(max_steps=5, save_each_step=False)

    for i, r in enumerate(logs, 1):
        print(f"[step {i}] {r}")

    # 結果をファイルにも残したい場合は True
    # exps.simulate(max_steps=3, save_each_step=True)
