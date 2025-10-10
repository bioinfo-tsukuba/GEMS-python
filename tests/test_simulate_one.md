# Overview / 概要
`tests/test_simulate_one.py` verifies the single-machine simulation features in `gems_python/one_machine_problem_interval_task`. It covers state transitions, task generation, dummy outputs for automated runs, and the behaviour of the scheduler.  
`tests/test_simulate_one.py` は、単一機器向けシミュレーション機能の動作を確認するサンプルです。状態遷移やタスク生成、ダミー出力による自動進行、スケジューリング挙動を確認できます。

# Prerequisites / 前提条件
- Run `poetry install` at the project root (`/data01/cab314/Project/GEMS-python`) to install dependencies including `polars`.  
  プロジェクトルート（`/data01/cab314/Project/GEMS-python`）で `poetry install` を実行し、`polars` を含む依存関係を整備してください。
- Execute the script through `poetry run` so that the expected Python interpreter is used.  
  `poetry run` を通じてスクリプトを実行し、想定する Python 環境を利用してください。
- `build_experiments` (`tests/test_simulate_one.py:106`) sets `parent_dir_path="experiments_dir_demo_one_machine"` and defaults to `save_each_step=False`, so no files are written unless you change the flag.  
  `build_experiments`（`tests/test_simulate_one.py:106`）では `parent_dir_path` に `experiments_dir_demo_one_machine` を指定し、`save_each_step=False` を既定としています。フラグを変更しない限りディスクへの保存は行われません。

# How to Run / 実行手順
1. Execute the commands below at the project root:  
   プロジェクトルートで次のコマンドを実行します。
   ```bash
   poetry install
   poetry run python tests/test_simulate_one.py
   ```
2. The console first prints simulated annealing scheduling logs, followed by per-step transition summaries such as `status`, `from_state`, and `to_state` in the `[step N] {...}` format.  
   コンソールにはシミュレーテッドアニーリングのログが表示され、その後に `[step N] {...}` 形式で状態遷移結果（`status`、`from_state`、`to_state` など）が続きます。
3. To persist results, change `save_each_step` to `True` as noted around `tests/test_simulate_one.py:124`. Directories like `experiments_dir_demo_one_machine/step_00000000` will then contain CSV schedules and experiment snapshots.  
   結果を保存したい場合は `tests/test_simulate_one.py:124` 付近のコメントを参考に `save_each_step=True` に設定してください。すると `experiments_dir_demo_one_machine/step_00000000` などが生成され、スケジュール CSV や実験情報が保存されます。

# Customising States and Tasks / 状態とタスクのカスタマイズ
- `InitState`, `MeasureState`, and `FinishState` (defined from line 18) inherit from `State` and provide `task_generator`, `transition_function`, and `dummy_output`.  
  `tests/test_simulate_one.py:18` 以降で `InitState`、`MeasureState`、`FinishState` を定義し、`task_generator`・`transition_function`・`dummy_output` を実装します。
- Each `TaskGroup` (e.g., line 31) models work on a single machine; adjust `penalty_type` and `optimal_start_time` to reflect scheduling constraints.  
  `TaskGroup`（例えば 31 行目）は単一機器での作業を表現します。`penalty_type` や `optimal_start_time` を調整して制約を与えてください。
- Ensure `dummy_output` returns the columns expected by `transition_function`, such as `measurement` in `MeasureState`; missing data prevents transitions.  
  `MeasureState` では `measurement` など遷移条件に必要な列を必ず `dummy_output` に含め、欠落がないようにしてください。列が不足すると遷移が進みません。

# Logs and Artifacts / ログと保存物
- Check `[step N]` lines to confirm transitions from `InitState` to `MeasureState` to `FinishState`. After reaching `FinishState`, the state remains there for subsequent steps.  
  `[step N]` 行で `InitState` → `MeasureState` → `FinishState` の遷移を確認してください。`FinishState` 到達後は同じ状態でループし続けます。
- When `save_each_step=True`, inspect `schedule.csv` and `experiments.json` under the generated directory to review scheduling results and shared variables.  
  `save_each_step=True` にした場合は、生成されたディレクトリ内の `schedule.csv` や `experiments.json` を確認し、スケジュール結果や共有変数履歴を把握できます。
- Remove unnecessary directories manually, or keep `save_each_step=False` to avoid persistent artifacts.  
  不要になった保存ディレクトリは手動で削除するか、保存を抑制したい場合は `save_each_step=False` のまま実行してください。

# Common Adjustments / よくある調整項目
- Shorten the run by lowering `simulate(max_steps=...)`.  
  `simulate(max_steps=...)` の値を小さくすると実行ステップ数を減らせます。
- If transitions halt, confirm that `dummy_output` returns a `pl.DataFrame` containing all required columns.  
  遷移が止まる場合は、`dummy_output` が必要な列を含む `pl.DataFrame` を返しているか確認してください。
- To experiment with penalties, modify the coefficient in `LinearPenalty` (line 24) or adjust task intervals to change scheduling pressure.  
  ペナルティを調整したい場合は `LinearPenalty`（24 行目）の係数やタスクの `interval` を変更して、スケジュールのずれに対する評価を試してください。
