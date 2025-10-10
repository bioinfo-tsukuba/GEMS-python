# GEMS Python 日本語クイックスタート

GEMS Python は、実験プロセスや生産タスクをステートマシンとして表現し、
シミュレーションおよびスケジューリングを行うための Python ツールキットです。
単一のリソースを想定したワークフローと、複数の機器を扱うワークフローの両方を
サポートし、プラグイン型のインターフェースで実験を柔軟に管理できます。

## コアパッケージ

- `gems_python.one_machine_problem_interval_task`  
  単一リソース前提のステート遷移モデルを提供し、シミュレーション補助機能や
  線形ペナルティモデルを備えています。
- `gems_python.multi_machine_problem_interval_task`  
  上記モデルを複数マシンへ拡張し、機器割り当てユーティリティと
  プラグインマネージャーを利用した対話型操作を提供します。

詳細な API リファレンスは `docs/` 以下の Sphinx ドキュメントで生成できます。

## セットアップ

Poetry を利用する場合（推奨）:

```bash
poetry install
poetry shell
```

pip を利用する場合:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows の場合: .venv\Scripts\activate
pip install -r requirements.txt
```

## クイックスタートの流れ

1. **単一マシン実験のシミュレーション**
   - `tests/test_simulate_one.py` を参照し、`InitState`・`MeasureState`・`FinishState`
     の三状態を定義する最小例を把握します。
   - 次のコマンドでシミュレーションを実行できます。
     ```bash
     python tests/test_simulate_one.py
     ```
     このスクリプトは `Experiments.simulate(max_steps=3, save_each_step=True)` を呼び出し、
     進捗スナップショットを出力します。

2. **複数マシンプラグインマネージャーの起動**
   - `examples/multi_machine_demo/` など任意の実験ディレクトリを用意し、
     保存済みステップと `experimental_setting/` 配下のプラグインモジュールを配置してください。
   - 以下のコマンドでインタラクティブループを開始します。
     ```bash
     python main.py
     ```
   - `mode/mode.txt` を編集すると `loop`、`add_experiment`、`delete_experiment` などの
     モードを切り替えられます。`PluginManager` は
     `experimental_setting/*.py` を自動的にスキャンし、新しい実験生成関数を登録します。

3. **実験ロジックの拡張**
   - 任意のモジュールで `State` を継承し、`task_generator` と
     `transition_function` を実装します。オフラインシミュレーションが必要な場合は
     `dummy_output` をオーバーライドします。
   - `TaskGroup` に関連付けるペナルティとして、線形ペナルティや休止周期ペナルティを
     選択し、スケジューリング挙動を調整します。

## ドキュメントの生成

Sphinx を利用した詳細ドキュメントは次のコマンドで生成できます。

```bash
sphinx-build -b html docs/source docs/build/html
```

生成後は `docs/build/html/index.html` をブラウザーで開いてください。
`quickstart` セクションが上記手順を再掲し、`autodoc` と `napoleon` が出力した
API リファレンスへリンクしています。

## 詳細例

ここでは、単一マシンのシミュレーション結果を活用しながら、複数マシン環境で実験を進める具体的な流れを紹介します。

1. **シミュレーション用の実験を作成**
   - `tests/test_simulate_one.py` を `examples/demo_states.py` にコピーし、状態遷移やペナルティ設定を自分のワークフローに合わせて調整します。
   - 次のコマンドでシミュレーション成果物を生成します。
     ```bash
     python examples/demo_states.py
     ```
     実行すると `experiments_dir_demo_one_machine/` のようなディレクトリにタスク履歴や共有変数の履歴が保存されます。

2. **プラグインとして取り込む**
   - `experimental_setting/` 配下に `demo_plugin.py` を作成し、次のように記述します。
     ```python
     from examples.demo_states import build_experiment

     def demo_plugin():
         return build_experiment()
     ```
   - `main.py` を再起動すると、`PluginManager` が新しいプラグインを自動で読み込みます。

3. **複数マシンのワークフローを操作**
   - `mode/mode.txt` を `add_experiment` に設定し、`mode/mode_add_experiment.txt` へ `demo_plugin.demo_plugin` を書き込んで実験を登録します。
   - モードを `loop` に切り替えると、保存済みの成果物を参照しながらスケジューラがステップを進めます。
   - コンソールに表示されるステップディレクトリを確認し、生成された `schedule.csv`、ガントチャート、スナップショット画像をレビューしてください。

## State と Experiment の設定手順

GEMS Python のコアは `State` クラスを継承した状態遷移と、`TaskGroup`・`Task` を組み合わせたスケジューリング定義で構成されます。以下の流れを押さえておくと実装がスムーズです。

1. **State サブクラスを実装**
   - `State` を継承し、次のメソッドを必ず実装してください。
     - `task_generator(self, df: pl.DataFrame) -> TaskGroup`  
       ペナルティやタスク、最適開始時刻などをまとめた `TaskGroup` を返します。
     - `transition_function(self, df: pl.DataFrame) -> str`  
       共有変数履歴を参照し、次に遷移すべきステート名を返します。
     - 必要に応じて `dummy_output` をオーバーライドし、オフラインシミュレーション用のダミー結果を出力します。
   - `TaskGroup` には `Task` を追加し、`processing_time`、`interval`、`experiment_operation` などを設定します。ペナルティとしては `LinearPenalty` 等を選択できます。

2. **Experiment を構築するファクトリ関数を用意**
   - 代表的には `build_experiment()` を定義し、以下を設定した `Experiment` を返します。
     - `experiment_name` — ログや保存ファイルに表示される識別名
     - `states` — 作成した State インスタンスのリスト（順序が遷移順になる想定）
     - `current_state_name` — 初期状態のクラス名
     - `shared_variable_history` — 初期状態では空の `pl.DataFrame()` を渡すことが多い
   - 複数実験を束ねる場合に備えて `build_experiments()` を用意し、`Experiments` インスタンスを返せるようにしておくと便利です。

3. **Experiments で管理**
   - `Experiments(experiments=[build_experiment()], parent_dir_path=...)` のように生成し、シミュレーションやマルチマシン管理を一括で行います。
   - ローカルテストでは `simulate(..., save_each_step=True)` を呼び、その後 `proceed_to_next_step()` を実行してプラグインマネージャーが参照するディレクトリ構造を出力します。

`tests/test_simulate_one.py` はこれら一連の構成要素をすべて含むテンプレートになっているため、新しいワークフローを作成する際の参考にしてください。

## リポジトリ構成

- `gems_python/one_machine_problem_interval_task/` — 単一マシン向けの遷移管理、タスク、ペナルティ実装
- `gems_python/multi_machine_problem_interval_task/` — 複数マシン向け拡張と CLI ユーティリティ
- `docs/` — Sphinx ソース (`index.rst`、`quickstart.rst`、API スタブ)
- `tests/` — シミュレーション例と回帰テスト
- `main.py` — 複数マシンプラグインマネージャーのエントリーポイント

## 次のアクション

- `tests/test_simulate_one.py` のステート定義を参考に、自身のワークフローへ合わせて `Experiments` を構築してください。
- `experimental_setting/` にプラグインを追加し、複数マシンマネージャーで新しい実験を登録してください。
- モジュールの docstring を拡充し、Sphinx ドキュメントを再生成することで最新の API を共有できます。
