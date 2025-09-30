# What is GEMS?

# How to Use

This guide outlines the necessary steps to define and run a cell culture experiment simulation using Rust. It focuses on setting up the experiment without providing direct code examples.

## Prerequisites

- poetry
- python >= 3.10

## setup

```shell
poetry install
```

## Test

```shell
python -m unittest
```

## シミュレーション機能の利用方法

セルカルチャー実験の状態遷移を事前に検証したい場合は、`Experiments.simulate` と `Experiments.simulate_one` を利用してダミー結果によるシミュレーションを行います。以下の手順に従って準備してください。

1. **各 State に `dummy_output` を実装する**  
   `State` を継承したクラスでは、既存の `task_generator` と `transition_function` に加えて `dummy_output` を実装します。返り値は `polars.DataFrame` で、カラム構成は遷移関数が参照する値（例: `confluence`, `measurement` など）を必ず含めます。シミュレーション時はこのデータが実験結果として扱われます。
2. **`Experiment` と `Experiments` を組み立てる**  
   実験ごとに `Experiment` を生成し、必要な `MachineList` を登録したうえで `Experiments` に渡します。実データ保存を行いたい場合は `parent_dir_path` を既存の保存ディレクトリに合わせて指定してください。
3. **シミュレーションを実行する**  
   簡易検証なら `simulate_one` で1ステップだけ進め、連続検証なら `simulate(max_steps=ステップ数)` を実行します。戻り値は各ステップの状態・タスク情報を含む辞書のリストです。保存を伴う検証を行う場合は `save_each_step=True` を指定し、実運用と同様にステップディレクトリが作成されることを確認してください。

最小構成のサンプルとして `tests/test_simulate.py` では 3 状態のダミー実験を、`tests/HEK_two_state_growth_sim.py` では HEK 細胞の２状態（Passage / Observation）を想定した成長シミュレーションを用意しています。初めて利用する場合はこれらをコピーしてカスタマイズするとスムーズです。

### 例: テストスクリプトの実行

```shell
# ダミー状態遷移サンプルの実行
poetry run python tests/test_simulate.py

# HEK 二状態フローのサンプル実行
poetry run python tests/HEK_two_state_growth_sim.py

# 既存のユニットテストをまとめて走らせる場合
poetry run python -m unittest discover -s tests -p "test_*.py"
```

各スクリプトは標準出力にステップごとの状態遷移ログを表示します。期待する遷移や推定値が得られているかを確認し、必要に応じて `dummy_output` の内容や閾値パラメータを調整してください。

## Steps to Define an Experiment

To define an experiment using the `interactive_ui.py` script with the updated directory and settings, follow these steps:

### 1. **Set Up the Project Directory**
- Ensure that your working directory has the following structure:
  ```
  sample_experiment/
  ├── experimental_setting/
  │   └── sample_setting.py
  └── mode/
      └── mode.txt
  ```

### 2. **Create Experiment Modules**
- The `ot2_setting.py` file should define at least one class representing an experiment and its state.
- Example structure in `ot2_setting.py`:
  ```python
  from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, Experiments, State

  class StandardState1(State):
      def task_generator(self, df):
          # Task generation logic here
          pass

      def transition_function(self, df):
          return "StandardState1"

  def gen_sample_experiment(experiment_name="sample_experiment"):
      return Experiment(
          experiment_name=experiment_name,
          states=[StandardState1()],
          current_state_name="StandardState1",
          shared_variable_history=
            pl.DataFrame({
                "time": [0],
                "temperature": [0],
                "pressure": [0]
            })
      )
  ```

### 3. **Configure Mode Settings**
- The `mode/mode.txt` file determines the current operational mode.
- Example `mode.txt`:
  ```
  add_experiments
  ```

### 4. **Add Experiments**
- Create a file named `mode_add_experiments.txt` in the `mode/` directory:
  ```
  sample_setting.gen_sample_experiment
  ```
- The script will automatically read this file and add the experiment. The file is deleted after successful loading.

### 5. **Run the Script**
- From your project root directory, run the script:
  ```bash
  python gems_python/multi_machine_problem_interval_task/interactive_ui.py
  ```

### 6. **Manage Modes**
- Available modes:
  - `module_load`: Reload all plugins.
  - `add_experiment`: Add a single experiment.
  - `delete_experiment`: Delete an experiment by UUID.
  - `show_experiments`: List all loaded experiments.
  - `add_machines`: Add machine configurations.
  - `delete_machines`: Delete specified machines.
  - `proceed`: Move to the next experiment step.
  - `stop`: Pause execution.
  - `exit`: Stop the program.

### 7. **Adding Machines**
- Create `mode_add_machines.txt` in the `mode/` directory:
  ```
  0,OT-2
  1,Human
  ```

### 8. **Deleting Machines**
- Specify machine IDs in `mode_delete_machines.txt`:
  ```
  0
  1
  ```

### 9. **Checking Available Modes**
- Set `mode.txt` to `help` to display all available modes:
  ```
  help
  ```

### 10. **Proceeding Through Experiments**
- Update `mode.txt` to `proceed` to move to the next state:
  ```
  proceed
  ```

### 11. **Stopping and Exiting**
- To pause execution, write `stop` in `mode.txt`.
- To terminate, write `exit` or `eof` in `mode.txt`.

---

By following these steps with the given directory and experiment settings, you can successfully define, manage, and execute OT-2 experiments using the provided interactive UI.
