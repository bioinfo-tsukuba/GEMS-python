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
