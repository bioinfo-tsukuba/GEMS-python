# 当時未解決だった保存・復元の完成

当時、技術的に解決できなかった双方向の保存・復元を完成させる修正です。
Issue #19、main `3641e77817d8f4821b01026884be1a04543a7c1b` 起点の
`fix/previously-unresolved-roundtrip` で実装しました。

## 対象と保存内容

以下の3管理系で同じ保存・復元APIを使用できます。

- `one_machine_problem_simple_task`
- `one_machine_problem_interval_task`
- `multi_machine_problem_interval_task`

`Experiment` / `Experiments` / State / Task / TaskGroup / penalty / Machineの
具体的なクラスと属性を保存します。Stateの必須設定・動的な進捗、
`init=False`の属性、現在Stateの名前とindex、履歴のバックエンド・列順・型・値、
タスク時刻・ID・完了状態、機械割当、管理系のstep・reference timeが対象です。
同じTaskGroup等を複数箇所から参照している場合、その共有も復元します。

復元ではコンストラクターと`__post_init__`を実行しません。
UUID生成、スケジュール再計算、State/taskの再生成、保存先ディレクトリの作成を
読込時に行わず、保存時点の値を戻します。

## 使用例

```python
from pathlib import Path
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments

# labはスケジュール・実行途中のExperiments
json_text = lab.to_json()
restored = Experiments.from_json(json_text)

lab.to_pickle(Path("experiments.pkl"))
restored = Experiments.from_pickle(Path("experiments.pkl"))
restored = Experiments.from_pickle(dumped=lab.to_pickle())

dictionary = lab.to_dict()
restored = Experiments.from_dict(dictionary)
```

単純タスクの`Experiments`にも同じ6メソッドを追加しました。
`save_all()`は従来の個別実験・tasks出力に加え、
コレクション全体の`experiments.json`と`experiments.pkl`を保存します。
区間タスク管理系の`save_results()`も両形式を保存します。

Issue #24以降、共通保存処理は両形式を同じ保存世代へ書込み・同期し、
`snapshot_manifest.json`の切替えで確定します。reloadはmanifestが選んだ
世代とハッシュを検査し、書込み途中の世代を選びません。
ルートの2ファイルは互換用出力として残しますが、
manifestがある保存でこれらを直接編集してもreloadへ反映しません。
manifestなしの旧保存は従来の2ファイル方式で読みます。
詳しくは[保存世代と旧保存](experiment_lifecycle.md#保存世代と旧保存)を参照してください。

## reloadの自動選択

```python
# 区間タスク：最大の保存stepを自動選択
restored = lab.reload()
restored = lab.reload(step=3, source="json")
restored = lab.reload(step=3, source="pickle")

# 単純タスク：parent_dir_pathまたは指定ディレクトリから読込
restored = simple_lab.reload()
restored = simple_lab.reload(save_dir=Path("saved"), source="json")
```

| source | 動作 |
| --- | --- |
| `"auto"`（既定） | 両方を読む。型・値・DataFrame・共有参照が一致すればJSONを返す。有効なのが一方だけなら警告を出してその形式を返す |
| `"json"` | JSONだけ読む。失敗時はエラー |
| `"pickle"` | pickleだけ読む。失敗時はエラー |

両方が読めても内容が違う場合は、`auto`でエラーにします。
`source`を指定して選択してください。対話入力は行いません。
読込に失敗した場合も成功した場合も、呼出元のlabやstepは変更しません。
返却オブジェクトを変数に代入して使用してください。

## 保存形式と互換性

新しいJSONは`_gems_serialization: 1`、具体的な`module:Class`、
型付きコンテナ、共有参照を持ちます。`to_dict()`の出力形状も変更しています。
外部コードで旧辞書のトップレベル項目を直接参照している場合は変更が必要です。
保存・読込には公開APIを使用してください。DataFrameは既存のArrow IPC形式を使い、
空のPandas履歴のRangeIndex列について補足情報を保存します。

入れ子のdataclass、list / dict / tuple / set / frozenset、Enum、Path、UUID、
bytes、日付・時刻・期間、NumPyスカラーを保存できます。
型情報のない入れ子でも、新形式は保存された具体的な型を使用して復元します。

既存のクラス情報を含むState、識別可能なpenalty、Polars JSON・列辞書、
既存のArrow履歴を読み込めます。旧形式の型注釈があるtupleも復元します。
読込用辞書は変更しないので、同じ入力から繰り返し復元できます。

### 制約と意図的な判断

- 保存したクラスのモジュールを読込環境でも利用可能にしてください。
  関数内のローカルクラス、削除・改名されたクラスは復元できません。
  保存されるのは属性です。クラスの実行コード、観測関数、乱数生成器の外部状態や
  solverの内部状態は保存しません。今回の再開検証では同じ観測器・入力・solver設定を使います。
- 循環参照、ファイルハンドル、関数、未対応のオブジェクトは、項目のパスを示す
  `SerializationError`になります。共有参照は循環がなければ保存できます。
- 旧形式に具体的クラスの情報がない場合は、呼出APIや既存の型注釈で明示された
  クラスを使います。失われたExperiment派生型を自動で推定しません。
  Stateの必須設定や具体的penalty、進捗フィールドが欠けている場合は、
  デフォルト値で補って実行を再開せず、欠けた項目を示すエラーにします。
  例外として、ライフサイクル版がない旧multi-machine保存に限り、
  追加機能を無効にした初期記録・既定の再試行設定を補います。
  StateやTaskの進捗は作り直しません。
  旧形式にState indexがない場合は名前が一意に対応するときだけ復元し、
  複数のStateが同じ名前を持つ場合はエラーにします。
- 旧JSONには共有参照の情報がありません。その同一性を回復することはできません。
  旧pickleが正常で元の共有を保持している場合は`source="pickle"`で明示選択できます。
  この違いがあるJSON/pickleの組を`auto`で選択すると不一致エラーになります。
- 履歴は行順と名前付きの列をデータとする既存仕様です。Pandasの独自indexは
  保存対象ではありません。フレームはArrowで表現できる列型が対象です。
- JSONもpickleも、使用するクラスを読み込みます。信頼できる保存ファイルを
  読み込む前提です。
- UUIDを受け取る実験のIDは文字列に統一します。IPSサンプルの既定UUIDは
  インスタンスごとに生成し、指定された文字列IDは維持します。
- 今回の新形式は旧版のreaderで読めません。書き戻す前に旧ファイルを保管してください。

## 検証

追加の受入試験は`tests/test_persistence_roundtrip.py`です。
検証コマンド・環境・全テストの残存エラーは
[検証記録](benchmarks/persistence_roundtrip.md)を参照してください。
