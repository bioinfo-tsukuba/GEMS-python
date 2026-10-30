# Issue #24 実装・検証記録

起点main `2140ab1919adb453122fd964cfdb3a7c216c74b9`。
実装ブランチ `feat/24-experiment-lifecycle`、要件はIssue #24、文書PRは#26。
ユーザー指定の作業計画に基づく。確定設定と使用APIは
[実装仕様](../experiment_lifecycle.md)に記載する。

## 環境・基準

Python 3.14.4、OR-Tools 9.15.6755、Pandas 3.1.0、Polars 1.44.2、
PyArrow 26.0.0。既存の`/tmp/gems-result-latency-venv/bin/python`を使用。
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
