# Overview / 概要
`tests/test_simulate_multi.py` demonstrates how to use the simulation utilities in `gems_python/multi_machine_problem_interval_task` for multi-machine workflows. The script showcases state transitions, task generation, dummy outputs for automated progression, and scheduling validation.  
`tests/test_simulate_multi.py` は、`gems_python/multi_machine_problem_interval_task` に含まれるシミュレーション機能の使い方を確認するサンプルです。複数機器を扱うワークフローを想定し、状態遷移やタスク生成、ダミー出力による自動進行、スケジューリング結果の確認までを一度に体験できます。

# Prerequisites / 前提条件
- Run `poetry install` at the project root (`/data01/cab314/Project/GEMS-python`) to prepare dependencies such as `polars`.  
  プロジェクト直下（`/data01/cab314/Project/GEMS-python`）で `poetry install` を実行し、`polars` を含む依存関係を整備してください。
- Always execute the script via `poetry run` so that the expected Python environment is used.  
  実行時は常に `poetry run` を利用し、プロジェクトが想定する Python 環境を使ってください。
- With `save_each_step=True`, logs are saved under the `parent_dir_path` configured in `tests/test_simulate_multi.py:75`; otherwise only console output is produced.  
  `save_each_step=True` を指定した場合、`tests/test_simulate_multi.py:75` で設定した `parent_dir_path` 配下にステップごとの保存ディレクトリが作成されます。未指定の場合は標準出力のみ生成されます。

# How to Run / 実行手順
1. Execute the following commands at the project root:  
   プロジェクトルートで次のコマンドを実行します。
   ```bash
   poetry install
   poetry run python tests/test_simulate_multi.py
   ```
2. The console prints simulated annealing scheduling logs followed by per-step transition results (fields such as `status`, `from_state`, `to_state`).  
   コンソールにはシミュレーテッドアニーリングのログと、`status` や `from_state` などを含む各ステップの遷移結果が出力されます。
3. When `save_each_step` remains `True`, directories like `experiments_dir_demo/step_00000000` are created automatically with CSV schedules, JSON experiment data, and generated graphs. Adjust the output path by editing `parent_dir_path`.  
   `save_each_step=True` のままにすると `experiments_dir_demo/step_00000000` などが自動生成され、スケジュール CSV や実験情報 JSON、状態遷移図などが保存されます。出力先を変えたい場合は `parent_dir_path` を変更してください。

# Customising States and Tasks / 状態とタスクのカスタマイズ
- Define `InitState`, `MeasureState`, and `FinishState` (from `tests/test_simulate_multi.py:19`). Each state inherits from `State` and implements `task_generator`, `transition_function`, and `dummy_output`.  
  `tests/test_simulate_multi.py:19` 以降で `InitState`、`MeasureState`、`FinishState` を定義し、`task_generator`・`transition_function`・`dummy_output` を実装します。
- `task_generator` must set `TaskGroup` fields consistent with the registered machines in `MachineList`. The sample registers two machines (`machine_type=0` and `machine_type=1`; see line 67).  
  `task_generator` が返す `TaskGroup` では、`MachineList` に登録した機器種別（例えば `machine_type=0` と `machine_type=1`）を指定する必要があります（`tests/test_simulate_multi.py:67` 参照）。
- Ensure that `dummy_output` mimics actual experiment results; missing columns or unexpected values will block correct state transitions.  
  `dummy_output` は実際の実験結果に近いデータ形式で返してください。必要な列が欠けたり値が異なると遷移が進みません。

# Logs and Artifacts / ログと保存物
- Console lines like `[step N] {...}` report each transition; verify progression and terminal behaviour.  
  `[step N] {...}` 形式のログで遷移状況を確認し、進捗や終了状態を把握します。
- Saved directories contain `schedule.csv`, `experiments.json`, and graphs under `experiments/`. Review them to inspect scheduling quality and state graphs.  
  保存ディレクトリには `schedule.csv` や `experiments.json`、さらに `experiments/` 配下のグラフが含まれます。スケジュールの妥当性や状態遷移図を確認できます。
- Delete unused directories manually, or run with `save_each_step=False` to avoid disk writes.  
  不要になった保存ディレクトリは手動で削除するか、`save_each_step=False` としてディスク書き込みを抑制してください。

# Common Adjustments / よくある調整項目
- Reduce simulation length by lowering `simulate(max_steps=...)`.  
  `simulate(max_steps=...)` の値を下げるとシミュレーションを短縮できます。
- If transitions stall, confirm that `dummy_output` returns a `pl.DataFrame` with all required columns.  
  遷移が止まる場合は、`dummy_output` が必要な列を含む `pl.DataFrame` を返しているか確認してください。
- Setting `Task.optimal_machine_type` to a machine not registered in `MachineList` causes scheduling errors; align machine registration with task definitions.  
  `MachineList` に存在しない機器種別を `Task.optimal_machine_type` に指定するとスケジューリングに失敗します。機器登録とタスク定義を一致させてください。
