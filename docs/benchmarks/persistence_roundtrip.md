# 保存・復元の検証記録（2026-10-28）

当時、技術的に解決できなかった双方向の保存・復元を完成させる修正です。
Issue #19、`fix/previously-unresolved-roundtrip`。元の問題はIssue #2にも記録があります。

## 検証対象と環境

- 比較元main：`3641e77817d8f4821b01026884be1a04543a7c1b`
- UUID・描画修正：`7650d8c`
- 保存・復元の基本実装：`8f8da392f98318e6b86abb200943efbc88fbec6b`
- 最終コード・受入テスト：`1e38d2cbd42879aec5d0c447e511f92eabd3b367`（旧形式の曖昧なState indexを拒否する検査を追加）
- Python 3.14.4 / Pandas 3.1.0 / Polars 1.44.2 / PyArrow 26.0.0
- OR-Tools 9.15.6755 / NumPy 2.5.4 / Matplotlib 3.11.2
- 既存の環境 `/tmp/gems-venv/bin/python`、`MPLBACKEND=Agg`、`MPLCONFIGDIR=/tmp/gems-mpl`
- OR-Tools比較は既存のseed 0、worker 1、5秒制限を使用。時間計測は一致判定から除外。
- mainと修正後の全テストはそれぞれ`git archive`の隔離コピーから実行し、生成物を作業ツリーへ出力しないようにした。
- 最終コードではreloadの指定step・最新stepの選択、片方だけのファイル、旧State名の曖昧さの拒否も検証。その後の変更はドキュメントのみ。

## 結果

| 検証 | 結果 |
| --- | --- |
| mainの全テスト | 74件、10エラー |
| 修正後の全テスト | 87件、5エラー。残る5件はmainで再現した今回の対象外エラーと同一 |
| 関連テスト5モジュール | 43件成功 |
| 通常のdiscoverに含まれない既存dumperテスト | 4件成功 |
| Pandas / Polarsの仮想実験比較 | 9種類、各162ステップで全比較一致。全solver結果OPTIMAL |
| JSON / pickleからの実行再開 | 9種類×2バックエンド×2形式。中断しない実行と以降のイベント・スケジュール・全属性・共有参照が一致 |
| penalty復元・OR-Tools再スケジュール | 5種類×3管理系×2バックエンド×2形式で成功 |
| mainが生成した旧JSON | 単純タスクExperiment、単一・複数機械の区間タスクExperimentsの3ファイルを読込成功 |
| UUID | IPSの2サンプルでインスタンスごとの別ID、UUID文字列、明示UUID・文字列ID維持を確認 |
| 描画 | vis / vis_with_diffが非空のPNGを生成。画像を読み込み、寸法・画素差・タスク非変更を確認 |
| reload | 一致、不一致、明示選択、片方のみ存在、片方破損、両方読込失敗、指定step・最新stepの選択、失敗・成功時の呼出元非変更を確認 |
| 型・進捗 | 必須設定を持つState、Experiment派生型、動的属性、tuple / Enum / Path / UUID / NumPyスカラー、入れ子のdataclass、凍結・slots、非ゼロ時刻・ID・step、タスク状態・機械割当を確認 |
| 履歴 | 空、nullable数値・boolean、時刻、浮動小数、NaN、文字列の型・列順・バックエンドを確認 |
| 副作用・失敗経路 | constructor非実行、復元でディレクトリを作らないこと、入力辞書非変更、別プロセスでの復元、循環・未対応値・欠けた旧設定・未知のクラス/版/参照のエラーを確認 |

各ケースの入力SHA-256、実行時間、solver呼出数は
[比較CSV](persistence_backends_2026-10-28.csv)、環境と旧fixtureのSHA-256は
[metadata](persistence_backends_2026-10-28_metadata.json)に記録。
比較時のdirtyフラグは説明書の未コミット変更によるもので、コードと受入テストは上記実装コミットに一致している。

## 再実行コマンド

プロジェクトの依存を導入したPythonを使用してください。以下の`python`は今回の実行では
`/tmp/gems-venv/bin/python`です。

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/gems-mpl
python -m unittest tests.test_persistence_roundtrip tests.test_dataframe_backends tests.test_cp_sat_scheduler tests.test_experiment_simulators tests.test_scheduler_pattern_study
python -m unittest tests.dumper_test.test_dumper_deserialise tests.experiment_samples_multi_machine_one_interval_task.test_class_dumper
python scripts/compare_dataframe_backends.py --output /tmp/gems-persistence-comparison-new
python -m unittest discover -s tests -p 'test_*.py'
```

比較スクリプトの出力先は未存在のディレクトリを指定する。
全テストと旧dumperテストは一部がファイル・画像・`volatile`を作るため、隔離コピーでの実行を推奨。
たとえば実装コミットでの全テストは次のように再現できる。

```bash
verification_dir=$(mktemp -d /tmp/gems-persistence-check-XXXXXX)
git archive 1e38d2cbd42879aec5d0c447e511f92eabd3b367 | tar -x -C "$verification_dir"
cd "$verification_dir"
PYTHONPATH="$verification_dir" MPLBACKEND=Agg MPLCONFIGDIR=/tmp/gems-mpl /tmp/gems-venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

比較元の検証は同じ手順でarchiveするコミットをmainの上記ハッシュに置き換える。
全テストは後述の残存エラーのため終了コード1になる。

## 検証中の失敗と修正

| 観測・再現 | 原因 | 修正と確認 |
| --- | --- | --- |
| 旧simple task JSONのpenaltyが文字列で戻る | task側がpenaltyのJSON文字列を保存し、読込でその文字列を代入していた | 共通の型付き保存と旧形式adapter。具体的penaltyと計算値を確認 |
| Stateの設定が消え、ReactionState復元でcandidates不足になる。旧State.from_dictは入力辞書も変更 | Stateの旧to_dictがクラス名しか保存せず、from_dictがpop後にconstructorを呼んでいた | 全属性・具体的型を保存しconstructorを呼ばず復元。旧設定不足はパス付きエラー。繰返し復元でも入力不変 |
| IPSExperiment.from_jsonでunexpected experiment_name | 基底dataclassの引数で派生クラスconstructorを呼んでいた | 保存属性を直接復元。派生型保持・constructor呼出回数不変を確認 |
| 旧State indexがなく、同名のStateが複数ある | 保存データからindexを一意に決定できない | 推定をやめ、パス付きエラーにする検査と再現テストを追加 |
| CyclicalRestPenaltyのtupleがlistへ変化 | JSONの配列だけではtupleを区別できなかった | 新形式のtuple情報と旧形式の型注釈による復元を確認 |
| 新しい受入テストで空のPandas履歴が一致しない | Arrowの0列テーブルがRangeIndexの列情報を保持しない | 空のRangeIndex列の補足情報を保存。両形式・全管理系で一致確認 |
| 単一機械の途中再開でExperiment側のcompletedが更新されない | 管理側とExperiment側の共有TaskGroupが別々に復元された | 共有参照を保存。再開テストと共有を外した不一致検出、キー整列したJSONの読込を確認 |
| 既存のlegacy testがto_dictのキーを直接編集してKeyError | 新形式でトップレベル構造が変わった | 実際の旧形式入力を明示し、旧reader互換性を検証。保存済みcurrent_task=Noneを再生成しない |
| 新規testのpenalty名linearが見つからない | テストで既存factoryの引数を誤記 | クラス名LinearPenaltyに修正。全penaltyの受入試験成功 |
| 全テストのUUID例外と描画API例外 | IPSのUUID既定値がimport時のUUIDオブジェクト、Matplotlibの旧cm.get_cmap | インスタンスごとの文字列UUID、plt.get_cmapへ変更。元の5エラーが消え、PNGも読込確認 |

## 今回の対象外として残る5エラー

| テスト | 例外・原因 | 影響範囲 |
| --- | --- | --- |
| test_experiment_loop.TestExperimentStructure.test_experiment_structure | HekCellCulture()にcurrent_state_name未指定でTypeError | HEKサンプルのテスト初期化 |
| 同test_experiment_structure_graph | 同じsetUpのTypeError | 同上 |
| 同test_experiment_structure_task_generation | 同じsetUpのTypeError | 同上 |
| test_one_machine_problem_interval_task.TestClass.test_task_group | inputでEOFError | 対話を前提とする既存テストの自動実行 |
| test_transition_manager.TestTransitionManager.test_task_generator | OneMachineTaskLocalInformationをtupleとしてunpackしてTypeError | 古い返却形式を前提とする既存テスト |

この5件を今回の修正の対象には加えていない。新しい保存・復元テストと既存の関連テストは成功している。

## 実用上の限界

検証したのは仮想実験と同じ入力・観測器・solver設定での再開であり、実機との通信や外部サービスの状態復旧ではない。
旧JSONで失われたState設定・派生型・共有参照を後から再生することはできない。
詳細な対応型、読込前提、to_dict形状の変更と旧形式の制約は[保存・復元仕様](../persistence.md)に記載。
