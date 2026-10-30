# Experimentの終了・分割・保留

Issue #24。対象は`multi_machine_problem_interval_task`の仮想実験。
既存の`transition_function(history) -> str`は維持する。
State名や空のTask群だけで終了を推測しない。

## 終了と分割の宣言

Stateは任意の`lifecycle_action(history, context)`を実装できる。
Task群完了時、または空群の登録・State入場時に呼ぶ。既定の`None`は通常進行。
`context`にはExperiment UUID、永続Task群UUID、要求ID、論理時刻が入る。

```python
from gems_python.multi_machine_problem_interval_task.lifecycle import Finish, Split, ChildSpec

def lifecycle_action(self, history, context):
    if protocol_has_finished(history):
        return Finish("protocol complete")
    return None
```

終了では未完了の必要Taskがないことを確認し、Experimentを実行管理から削除する。
`lab.ended_experiments[uuid]`には履歴・終了理由・論理時刻・要求ID・初期条件と由来を残す。
物理試料の廃棄や履歴の消去は行わない。`EndRecord.history`はコピーを返す。
全実験終了は正常な`no_tasks`となる。

分割は`Experiments(..., enable_split=True)`で有効化する試験機能。
`Split(children)`または`lab.request_split(parent_uuid, request_id, children)`で要求する。
`ChildSpec`には子キー・名前・独立したState群・初期State名・初期観測、
初期条件の辞書・各条件の由来の辞書を必ず指定する。
量・単位・容器IDはその辞書に保存できる。科学的なコピー・換算・量保存の判断は
プロトコルで定義する。フレームワークは値を推測・自動配分しない。

要求だけでは子は活動せず、親を`SPLIT_PENDING`にする。
親の必要Taskが完了した後に
`lab.confirm_split(request_id, succeeded=True)`で確定する。
このAPIは装置操作を実行しない。仮想実行側が成功を確認して呼ぶ。
失敗は`confirm_split(..., succeeded=False, reason="...")`で記録し、
子を登録しない。訂正した仕様で再要求する場合は新しい要求IDを使う。

確定は親削除・全子登録・出自イベントを一括で変更し、計画対象をまとめて再計画する。
機械未登録の子も登録し、その子だけ装置待ちにする。
確定後の計画失敗で親を復活させない。
子UUIDは親UUID・要求ID・子キーから決定し、再送で増殖しない。
同じ要求IDに異なる仕様を渡すとエラーにする。
分割は2子以上。残余を続ける場合も子として指定する。

## 履歴と結果の再送

`shared_variable_history`はそのExperiment自身の初期観測と観測を保持する。
子には別途、分割時の親履歴スナップショットを保持する。
`history_for_state()`は親履歴→子の初期観測→子の観測を結合したコピーを返す。
Stateの遷移・Task生成・仮想観測へ同じ入力を渡す。
`inherited_history`もコピーを返す。兄弟・親の履歴を変更しない。
複数世代の分割でも継承履歴を重複してStateへ渡さない。

数値の`task_group_id`は一覧変更時に振り直される。
再送には永続的な`task_group_uuid`と`result_id`を使う。

```python
lab.start_task(task_group_uuid=group.task_group_uuid, task_id=task.task_id)
lab.record_task_result(group.task_group_uuid, task.task_id, result_id, result_frame)
```

同じ結果ID・Task・観測の再送は履歴やイベントを増やさない。
同じ結果IDに異なる内容を渡すとエラーにする。
結果ファイル経由でも`task_group_uuid`・`result_id`を追加できる。
旧数値ID APIは現時点の一覧への操作として残す。削除・振直しをまたぐ再送には永続IDが必要。

## 保留・探索再試行・STOP

最初のTask群全体と、後続群の生成時に入力を検査する。
必要な種類の機械が未登録なら、該当Experimentを`WAITING_MACHINE`にして警告する。
登録は保持し、その未開始群を探索から除く。他の実験は続ける。
`lab.add_machine(...)`で機械を登録すると再検査・再計画する。
機械が使用中でも、将来の有効な予約が得られる場合は装置待ちにしない。

`OPTIMAL`と`FEASIBLE`の計画を採用する。
`UNKNOWN`では警告し、未計画の開始を保留して探索上限を倍増する。

| 設定 | 既定 | 根拠 |
| --- | --- | --- |
| `solver_time_limit_seconds` | 5秒 | 従来設定を維持。呼出し単位の上書きも維持 |
| `unknown_policy` | `"retry"` | Issue #24の方針 |
| `max_unknown_retries` | 3回（初回を含まない） | 実装計画で指定 |
| `enable_split` | `False` | Issue #24の試験機能の方針 |

既定は5→10→20→40秒の最大4試行。実際の時間は構築・検査等も含む。
再試行で管理層の初回設定、登録、確定済み分割を変更しない。
`unknown_policy="stop"`なら初回UNKNOWNで警告とSTOP。
上限到達、INFEASIBLE、不正仕様・固定予約矛盾は診断付きErrorで全体STOP。
診断に対象Task群・機械・設定の再現入力を含める。
solverが矛盾箇所を特定できない場合はINFEASIBLE判定と入力を残し、原因を推測しない。
探索時間・構築・検査・反映の時間は構造化ログに記録し、科学的状態には混ぜない。

`STOPPED`では新しいTaskを開始しない。実行中操作を強制中断しない。
実行中Taskの結果は受領・記録し、次Stateへの進行と新Task生成は解除後まで保留する。
原因を訂正し、`lab.resume()`で明示的に再検査・再計画する。
正常な保存は読込み後に実行でき、STOP保存は読込み後もSTOPを維持する。
読込み自体では開始・再計画を行わない。

## 保存世代と旧保存

`save_snapshot`はJSON/pickleを同じ世代の`.snapshot_generations/<id>/`に書込み、
ファイル・ディレクトリを同期してから` snapshot_manifest.json`を原子的に切り替える。
読込みはmanifestを一度選び、その世代のハッシュを検査する。
書込み途中の世代は選ばず、既存の確定世代を維持する。
両形式が使えなければ理由を表示し、世代を混ぜて復元しない。
片方だけ有効なら既存の`source="auto"`と同じく警告してその形式を読む。

ルートの`experiments.json`／`experiments.pkl`も互換用出力として残す。
manifestがある場合、これらは読込みの正本ではない。
外部ツールがファイルを直接編集した結果を、manifest付き保存へ暗黙に反映しない。
manifestがない旧保存は従来の2ファイル方式で読む。
世代は再現・調査用に保持し、自動削除しない。
この変更は共通保存処理を使う3管理系に適用する。

汎用シリアライズ形式はversion 1を維持し、ライフサイクルにも別のversion 1を持つ。
ライフサイクル版がない旧保存だけに、無効な分割機能・空の記録・既定の再試行設定を補う。
新保存の欠けた進捗項目はエラーにする。StateやTaskの進捗を再生成しない。
保存クラスのコードは読込み環境に必要であり、solverの探索内部状態は保存しない。
再開する探索は残り回数と次の探索上限を引き継いだ新しい探索になる。

## 検証と対象外

受入試験は`test_experiment_lifecycle`、`test_experiment_scheduling_policy`、
`test_lifecycle_persistence`。既存の予約検査・DataFrame・保存・仮想実行試験も回す。
機能は単一プロセスの管理層を対象にする。
実機分割、不可逆操作のexactly-once保証、強制中断、物理的復旧、
複数プロセスからの同時書込みは対象外。
