# Issue #24 実装・検証記録

起点main `2140ab1919adb453122fd964cfdb3a7c216c74b9`。
実装ブランチ `feat/24-experiment-lifecycle`、要件はIssue #24、文書PRは#26。
ユーザー指定の作業計画に基づく。確定設定と使用APIは
[実装仕様](../experiment_lifecycle.md)に記載する。

## 環境・基準

Python 3.14.4、OR-Tools 9.15.6755、Pandas 3.1.0、Polars 1.44.2、
PyArrow 26.0.0。初回は`/tmp/gems-result-latency-venv/bin/python`を使用。
一時領域の消失後、`/tmp/gems-issue24-venv/bin/python3`へ主要依存を同じ版で復元し、
関連試験・比較元の全試験を再実行した。依存範囲は`pyproject.toml`に合わせる。
`MPLBACKEND=Agg`、描画設定は`/tmp`。新しい依存は追加しない。

比較元を`git archive`で`/tmp/gems-issue24-implementation`へ取り出して、
`python -m unittest discover -s tests -p 'test_*.py'`を実行。
109件、45.468秒、既存の5エラーを確認。
HEK初期化3件、対話inputでEOF 1件、旧Task返却型のunpack 1件。
基準ログは`/tmp/gems-issue24-implementation-baseline.log`。

## 機能の検証

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/gems-issue24-mpl \
  /tmp/gems-result-latency-venv/bin/python -m unittest \
  tests.test_experiment_lifecycle tests.test_experiment_scheduling_policy \
  tests.test_lifecycle_persistence tests.test_scheduler_capacity \
  tests.test_cp_sat_scheduler tests.test_dataframe_backends \
  tests.test_persistence_roundtrip tests.test_experiment_simulators
```

最初の統合確認では74件、19.810秒、成功。
自己消去、1親→2子、履歴の独立性、要求・結果の再送、分割途中の例外、
装置待ちと他の実験の継続、倍増・上限・STOP設定、固定予約・不正仕様、
STOP中の結果受領、Pandas/Polars×JSON/pickle、別プロセス再開、
書込み・manifest切替えの中断とハッシュ検査を含む。
以降の変更後に最終結果を追記する。機能検証のログは
`/tmp/gems-issue24-third-tests.log`。

## 発見した失敗と修正

- 初期統合で、Experimentを持たずTask群だけを渡す既存の比較用管理層が
  Experiment検索で`StopIteration`となった。登録のない群の計画を維持し、
  局所保留は登録されたExperimentへ適用するよう修正。既存の全penalty・3方式の試験で確認。
- 新しい履歴再開試験に、永続Task群IDの新規乱数と保存された時間計測が混ざった。
  Task群IDをExperiment UUIDと保存された生成順から決定し、
  壁時計の計測は構造化ログへ分離した。中断なしとの再開比較で確認。
- 予約維持の新テストは、同じ機械の長い固定予約の完了前に親が2回進めると誤って仮定した。
  独立した機械で長い実行中Taskを置くケースに修正し、親終了と固定予約不変を確認。
- 旧2ファイルの直接編集を前提とするreload試験は、
  manifestなしの保存を明示して旧readerを検証するよう変更。
  新世代の中断・破損・互換出力の扱いは専用の障害注入試験で確認する。

## 残る検証

最終の全テスト比較、有限の培養モデルの統合例、規模評価、
Issueの完了条件との対応、最終PRの検証結果は継続して追記する。
規模評価は最大21600秒。実機・同時書込み・不可逆操作の復旧は未検証かつ対象外。

## 追加の受入確認

復元環境で関連81件、25.387秒、成功。有限の仮想培養を2回観測してから
明示的に2子へ取り分け、各子が2回観測して自己消去する例を両バックエンドで確認した。
初期confluenceは手順指定の0.12/0.18、量は10から5/5の仮想単位で、
履歴から算出するlogistic growthと一致する。物理的な量配分の自動推測はしない。
2世代の継承、子の生成Taskの事前検査、結果ファイル再送時の時刻・step不変も検証した。

保存のmanifest切替え直前／直後に別プロセスを`os._exit(77)`で終了させ、
直前は旧世代、直後は新世代だけを復元できた。
mainそのものから生成したJSON/pickleは`tests/fixtures/issue24_main`へ保存し、
同じ培養手順を継続できる移行試験を追加した。

規模評価器は`scripts/measure_experiment_lifecycle.py`、
独立した保存入力・予約・系譜検査は`scripts/verify_experiment_lifecycle_capacity.py`。
設定は`experiment_lifecycle_capacity_config.json`に固定する。
計画入口から検査済み反映までを1秒/60秒判定し、同じ探索上限で再試行を無効化する。
親1〜128の走査、8親で子4/8・群間隔1/2/4・2世代、5負荷と同条件のflat比較を行い、
成功境界と次の失敗点を合計3回測る。前処理の履歴は決定的な完了済み入力であり、
生物学的な実行時間や物理操作時間は測らない。

### 最終要件照合による補完

終了記録の初期条件だけでは手順設定・最終State進捗の追跡が不十分だったため、
`EndRecord.settings`にState群・バックエンド・管理層設定を独立コピーで保存した。
INFEASIBLEの理由欄には、solverが矛盾箇所を特定していないことを明示し、
不正Taskの事前診断にもTask IDを追加した。
`eb0cee3`の規模測定は25試行で停止した。
検査済み23出力は有効、11比較組の物理条件も一致したが、
最終ソースの容量・保存評価には混ぜない。
予備資料は元の`eb0cee3`の検査器で25入力・25有効出力・12組を照合した。
最終コードのreaderは追加前の未公開EndRecordに欠けた設定を補わずエラーにするため、
予備資料の復元には生成元のcommitを使う。旧mainの保存の移行は別の受入試験で保証する。

補完後の関連82件は26.151秒で成功。
READMEの通常登録・UNKNOWNの説明と共通保存仕様も、新しい既定動作に更新した。

最終機能commit `0832eb2`の隔離全試験は140件、53.662秒。
比較元109件・46.020秒と同じ5エラーだけで、新規の失敗はない。
全試験の成功とは記載せず、既存エラーは対象外のまま明示する。
統計・図の再生成は`scripts/analyze_experiment_lifecycle_capacity.py`を使用し、
再探索せずに検査済みのJSONLを読む。

追加受入2件（15.780秒）は、分割確定後のUNKNOWN/INFEASIBLEでも全子・系譜を保持する確認と、
要求前・確定前・確定後・装置待ち・再試行中・STOP・全終了の7時点を
Pandas/Polars×JSON/pickleの28組で別プロセスから再開する確認。
各再開後は重複登録なしで3件の終了記録へ到達した。
入力のTask仕様を改変した出力は、独立検査器が保存内容との一致だけでなく
元の処理時間・順序・種類・永続IDとの比較でも拒否する（関連2件、4.442秒）。

### 一時環境の消失と再開

一時領域と計測プロセスが失われたため、`a719506`で完了していた54試行を
ワークスペースのチェックポイントから復元する。予備測定`eb0cee3`の生資料は
退避前に失われており、上記の照合結果の記録だけが残る。最終評価には使わない。
依存は退避した`environment.txt`から
`volatile_issue24_work/venv`へ復元した（Python・freeze・主要4依存の一致を検査）。

再開処理は既存試行を再探索せず、重複試行も拒否する。
`gems_python`全ファイルと基礎計測器を元commitのバイト列で照合し、
ParentState/ChildState、入力生成、条件抽出、worker、試行列挙はASTでも一致を検査する。
変更は再開用の実行管理と別計測の初期化プロファイルに限る。
再開時のソースハッシュ・版・開始時刻は`resume-*.json`へ保存する。
最初の予備開始から6時間の締切`2026-10-30T13:25:22.780653+00:00`は延長しない。

初期54試行は`/tmp`のtmpfs、以後はワークスペースのファイルシステムを使う。
計画入口から反映までのタイマーに保存I/Oは入らない。
保存時間・全工程時間は異なる保存先を同じ母集団にまとめない。
親・機械・管理層の初期化は元の前処理タイマーに入っていないため、
同じ入力ハッシュを確認する別プロセスの補足計測で示す。
この補足時間を過去試行の全工程時間へ加算しない。
