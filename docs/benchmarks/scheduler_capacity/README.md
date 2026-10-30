# Issue #23: 評価の結論と保存資料

CP-SATの探索上限へ1秒／60秒を直接設定しました。構築・反映・検査などの時間は成功判定に含めません。
成功は制約を満たす計画の返却です。FEASIBLEは最適性未証明、UNKNOWNは解未発見です。

## 評価した問題の要約

共通の基本構成は、1実験につき現在のタスク群1つ、処理時間5・3・4分の3工程です。
群開始から第1工程まで2分、第1工程終了から第2工程まで2分、第2工程終了から第3工程まで1分の間隔を固定し、
群の開始時刻と各工程の機械割当を決めます。目的は開始時刻のペナルティ最小化で、同じペナルティなら早い開始を選びます。
以下の工程数は探索に渡した未完了工程数で、将来生成されるタスク群は含みません。

| 負荷 | 問題の要約 | 1秒: 全反復成功の最大実測規模（実験／工程） | 60秒: 同左 |
| --- | --- | ---: | ---: |
| 同型機械の競合 | 全3工程を同型2台に割り当てる。希望開始時刻が近い多数の群について、同じ機械を奪い合う競合を評価する。 | 256／768 | 512／1,536 |
| 異種工程 | 第1・第2・第3工程が別の機械種類を使い、台数は順に2・2・1台。種類ごとの競合と、固定間隔を保った工程の連携を評価する。 | 512／1,536 | 512／1,536 |
| 割当候補多数 | 全3工程の割当先を同型8台から選ぶ。機械割当の選択肢が多い問題を評価する。 | 128／384 | 512／1,536 |
| 複雑ペナルティ | 同型2台を使い、希望時刻からのずれ、許容範囲付き、周期的休止帯付きの3種類の開始時刻ペナルティを群ごとに混在させる。時刻に対する費用の違いを評価する。 | 128／384 | 512／1,536 |
| 実行中作業を含む再計画 | 同型2台で、群の1/4は開始済み・先頭工程完了済み。開始済み群の残り工程の時刻と機械IDを固定し、長い待ち時間を挟む将来の予約を維持して未開始群を配置する。 | 256／704 | 512／1,408 |

再計画の開始済み群では第2工程前の間隔を延ばしており、上記の基本間隔とは異なります。
固定された未完了工程は1秒側の規模で128、60秒側で256です。
複雑ペナルティの休止帯は群の開始時刻への費用で、機械を使用できない時間帯という制約ではありません。
入力構成は [`scheduler_capacity_config.json`](../../../scripts/scheduler_capacity_config.json) と
[`make_case` / `CapacityState`](../../../scripts/measure_scheduler_capacity.py) で再現できます。

各値の境界条件は5/5成功。512は測定上限であり、真の処理可能上限ではありません。
静的成功は継続更新を保証しません。1秒・128実験の割当候補多数／複雑ペナルティは更新途中でUNKNOWN、
60秒の運用は上表の**1秒側の規模**で全5負荷8更新成功でした。
最終ソースではこの2失敗を再現し、補助64実験・1秒は両負荷8更新成功（各1系列）でした。

同じ入力で両探索上限が成功した39条件では、60秒のペナルティ中央値は23条件で改善、16条件で同じでした。
最大遅延は20条件で改善、17条件で同じ、2条件で増加しました。
同じ決定的入力・seedの少数反復による記述統計で、任意の入力への成功率、最適性、応答時間、設備処理量の保証は行いません。

## 結果受領後の応答時間（2026-10-30）

受信済みの結果DataFrameを渡してから、作業固定・履歴更新・再計画・全スケジュールJSON生成までを追加測定しました。
探索は応答ごとに1回です。前の運用時間（約2秒／120秒）には受領前の探索も含まれるため、別の指標として扱います。
前回と同じ5負荷・規模で各3反復し、成功応答の中央値は1秒設定で1.016〜1.035秒、60秒設定で60.014〜60.025秒でした。
複雑ペナルティ128実験・1秒は3/3 UNKNOWN（失敗応答中央値0.550秒）、他の条件は3/3成功でした。

今回はタスク群途中の最初の結果1件を再生しています。状態遷移・次群生成が起きる群終端の応答時間、
複雑な実験解析、機器通信、描画、ディスク保存は含みません。継続更新の保証でもありません。
図・全条件表・CSV・生入力/出力は
[保存資料](https://github.com/bioinfo-tsukuba/GEMS-python/blob/747902191e754bb79de902145b2ac6bd9cca456e/docs/benchmarks/result_update_latency_2026-10-30/README.md)
へ分離しています。保存ブランチ `archive/23-result-update-latency`、固定コミット
`747902191e754bb79de902145b2ac6bd9cca456e` を継続保持します。
30入力・30探索・27有効出力・3 UNKNOWNを照合し、関連65テスト成功、統計再生成と165ファイルのアーカイブハッシュ検査も成功しました。

## 詳細資料を履歴に保持

図・統計表・生データを実装PRの最終差分から分離し、保存用ブランチ
`archive/23-scheduler-capacity` の固定コミット
`bf457334a0bddc3a75ff6560726a80e3c42dbf80` に保持しています。
このブランチは資料の継続参照のため、実装PRのマージ後も保持します。
図はIssue #23／PR #25のコメントからも閲覧できます。

- [統計量・4図（PNG／SVG）](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/statistics/README.md)
- [全条件の統計CSV](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/statistics/condition_statistics.csv)
- [生データ・復元手順](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/README.md)
- [統計の検証](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/statistics/verification.json)
- [実装検証・既存の5エラー・制限](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/verification.md)

主評価は `9e269d4`（193試行）、最終ソース確認は `b08dbe8`（17条件）です。
ソースを混ぜずに集計し、入力ハッシュ・全呼出し・失敗・環境を保存しています。
追加分析は再探索を行わず、元の105条件の集計との一致と別出力先での再現を確認しました。関連63テスト成功。

## 再現

保存資料を別ディレクトリへ取り出し、現ブランチのスクリプトで統計と図を再生成できます。
以下は保存ファイルの読み出しと集計のみで、探索を実行しません。

```bash
mkdir -p /tmp/gems-capacity-records
git archive bf457334a0bddc3a75ff6560726a80e3c42dbf80 \
  docs/benchmarks/scheduler_capacity |
  tar -x -C /tmp/gems-capacity-records
MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg poetry run python \
  scripts/analyze_scheduler_capacity.py \
  /tmp/gems-capacity-records/docs/benchmarks/scheduler_capacity \
  --output /tmp/gems-capacity-statistics
```
