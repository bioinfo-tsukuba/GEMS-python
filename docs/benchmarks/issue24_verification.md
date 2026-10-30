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
最終確認の結果は後述する。初回機能検証のログは
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

## 最終確認と対象外

機能受入、有限の培養・取り分け、規模評価、全試験の比較、資料の再生成を完了した。
最終全試験は147件・70.136秒で、比較元と同じ既存5エラーだけ。
344試行・172比較組、全344保存を照合し、基本条件260試行に欠落・重複はない。
時間内88、有効な時間超過250、UNKNOWN 6。初期化の補足計測も344件、失敗・スキップなし。
表・図等の全6ファイルは別出力先への再生成でバイト一致し、PNGの表示も確認した。

規模評価の終了は2026-10-30 10:13:46 UTC、補足初期化の終了は10:17:52 UTC。
最初の開始07:25:22 UTCから6時間の期限内に終えた。
実機・同時書込み・不可逆操作の復旧は未検証かつ対象外。
静的な合成入力の規模評価であり、継続更新下の実施設の処理能力を保証しない。
メモリはプロセスピークRSS、履歴行数、シリアライズした入力サイズを記録し、
履歴・系譜だけの常駐メモリを分離した測定ではない。

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
変更は再開用の実行管理、境界反復の選択、別計測の初期化プロファイルに限る。
再開時のソースハッシュ・版・開始時刻は`resume-*.json`へ保存する。
最初の予備開始から6時間の締切`2026-10-30T13:25:22.780653+00:00`は延長しない。

初期54試行は`/tmp`のtmpfs、以後はワークスペースのファイルシステムを使う。
計画入口から反映までのタイマーに保存I/Oは入らない。
保存時間・全工程時間は異なる保存先を同じ母集団にまとめない。
親・機械・管理層の初期化は元の前処理タイマーに入っていないため、
同じ入力ハッシュを確認する別プロセスの補足計測で示す。
この補足時間を過去試行の全工程時間へ加算しない。

再開後の関連85件は42.510秒で成功。
`6055d8d`を`git archive`で隔離した全143件は70.196秒、比較元と同じ5エラーのみ。
全試験でのエラーID一覧を比較して一致を確認した。
期限切れの再開でsolverを呼ばず完了済み試行を保持すること、
保存先が違う反復の保存・全工程時間を混ぜないことも追加2件で確認した。
補足初期化の試運転は元の第1試行と入力ハッシュが一致した。

現環境での再現コマンドは次のとおり。受入試験は規模計測から独立して実行する。

```bash
MPLBACKEND=Agg MPLCONFIGDIR=volatile_issue24_work/mpl \
  volatile_issue24_work/venv/bin/python3 -m unittest \
  tests.test_experiment_lifecycle tests.test_experiment_scheduling_policy \
  tests.test_lifecycle_persistence tests.test_scheduler_capacity \
  tests.test_lifecycle_capacity tests.test_cp_sat_scheduler \
  tests.test_dataframe_backends tests.test_persistence_roundtrip \
  tests.test_experiment_simulators

MPLBACKEND=Agg MPLCONFIGDIR=volatile_issue24_work/mpl \
  volatile_issue24_work/venv/bin/python3 -m scripts.measure_experiment_lifecycle \
  --resume --config volatile_issue24_work/capacity-a719506/config.json \
  --output volatile_issue24_work/capacity-a719506

# 元の計測後、solverを呼ばずに補足初期化・出力検査・統計再生成を行う。
MPLBACKEND=Agg MPLCONFIGDIR=volatile_issue24_work/mpl \
  volatile_issue24_work/venv/bin/python3 -m scripts.measure_experiment_lifecycle \
  --profile-all-initializations volatile_issue24_work/capacity-a719506
volatile_issue24_work/venv/bin/python3 -m scripts.verify_experiment_lifecycle_capacity \
  volatile_issue24_work/capacity-a719506
MPLCONFIGDIR=volatile_issue24_work/mpl volatile_issue24_work/venv/bin/python3 \
  -m scripts.analyze_experiment_lifecycle_capacity \
  volatile_issue24_work/capacity-a719506 --output volatile_issue24_work/statistics
```

計測中の追加受入・全試験実行は短い併行負荷になったため、対象期間を別記する。
反復は同じ決定的入力・seedの記述統計であり、独立な成功確率の推定ではない。
RSSはfixture生成とflat化を含むプロセス全体のピークで、履歴だけの常駐メモリではない。
群間隔の比較は分割前に合成した完了観測の量を変えるもので、実機の分割頻度の実測ではない。

60秒の境界候補で59.998秒／60.004秒の差が観測されたため、
最大の走査成功点が3反復で全て時間内に通らなければ、一段小さい走査成功点へ下げて確認する。
走査・反復の既存資料は保持し、新しい候補も同じworker・同じ入力条件で測る。
最初の走査と候補反復が終わった後に再開処理を呼び、未実施の補助反復だけを追加する。
この選択処理を含む再開・集計の制御試験3件は0.007秒で成功した。

出力検査ではTask仕様だけでなく、科学的な初期条件・由来、State、生成順、
親の継承履歴と子自身の観測を別々に元入力と比較する。
初期条件だけの改変、結合履歴を保ったまま親由来の観測を子自身へ移す改変、
処理時間の改変を、出力と保存を両方書き換えた場合でも拒否する。
登録装置・論理時計・保存された方針も不変を検査し、架空の装置追加を拒否する
障害注入を含む関連6件は4.489秒で成功した。
制御・検査の関連6件は4.489秒で成功。
途中119試行の独立検査では113有効出力、6 UNKNOWN、119保存、59比較組を照合した。
UNKNOWNは再試行無効の比較設定でSTOPし、判定・入力と全子・系譜を保持していた。
Markdownレポートも集計JSONから再生成し、CSVの値が従来集計と変わらないことを確認した。
119試行の別出力先への2回再生成では、README・2 CSV・集計JSON・PNG・SVGの
全6ファイルがバイト一致した。SVGの自動日時とランダムIDは生成時に固定し、
測定値・図の内容は変更しない。
群間隔・世代数の変更では履歴量と論理時刻を変えるため、周期的費用の位相も変わり得る。
条件間の差を純粋な履歴負荷の差として解釈せず、各条件内のflat比較を使う。

### 最終ソースと保存資料

最終全試験の対象は `d5f0cdbab8c11429f818c615df7488aa94ec02e4`。
計測対象 `a719506` の実装・入力生成・workerは、この節の検証時点までは不変で、
当時の変更は再開、独立検査、集計、制御試験、文書に限る。
後述の終了後可視化は実行管理に保存台帳を追加するため、既存の容量数値は旧測定対象に固定した結果として扱う。

5つの既存エラーは `test_experiment_loop` の3件（HEK初期化）、
`test_one_machine_problem_interval_task.test_task_group`（対話inputのEOF）、
`test_transition_manager.test_task_generator`（旧返却型unpack）。
基準ログと最終ログのエラーIDが一致し、例外内容も確認した。全試験成功とは扱わない。

[測定資料・集計レポート](https://github.com/bioinfo-tsukuba/GEMS-python/blob/8ca9080bf62a4e6c2b70fe73a0f830c5f627b183/docs/benchmarks/experiment_lifecycle_capacity_2026-10-30/README.md)へ、
全入力・出力・保存・設定・環境・再開記録、試験ログ、照合結果、CSV/JSON、PNG/SVGを保存する。
`final_tests.json`、`final_coverage.json`、`final_regeneration.json`、
`figure_visual_check.json` は圧縮内に含まれる。
圧縮前の全ファイルと公開資料をSHA-256で照合する。
実装PR #27はレビュー用に準備し、マージは行わない。

保存コミット `8ca9080bf62a4e6c2b70fe73a0f830c5f627b183`。4209ファイルを圧縮内から読み出して全SHA-256を照合し、
公開資料のSHA-256も一致した。保存ブランチは資料の固定参照用に残す。

## 終了後の可視化の実装と検証

実装対象 `4e5f6894692eba8b3ad95bc0f132cbe751b0f4ef`。①時間軸上の親子関係、③家系別の親子・時間図、④親子図と装置図の選択連動を通常の API / CLI として実装した。②と旧比較入力・描画器・試作を削除し、[使用方法と実行出力](../design/experiment_retrospective/README.md)へ置換した。図中は名前・装置・時間軸・最小限の凡例と操作欄に絞る。

終了で削除されるTaskGroupとは別に、登録名・登録論理時刻と完了した作業の最終予約・装置を`retrospective_trace`へ残す。子の登録記録は分割の確定・ロールバックと一緒に扱う。結果と分割の再送で記録を増やさない。旧形式は明示的に不完全とし、欠落した時刻・作業を推測しない。実機の実測開始・終了は記録した予約と区別する。

```bash
MPLBACKEND=Agg MPLCONFIGDIR=volatile_issue24_work/mpl \
  volatile_issue24_work/venv/bin/python3 -m unittest \
  tests.test_retrospective tests.test_lifecycle_persistence tests.test_experiment_lifecycle \
  tests.test_persistence_roundtrip tests.test_dataframe_backends tests.test_simulate_multi \
  tests.test_lifecycle_capacity tests.test_lifecycle_capacity_recovery

MPLBACKEND=Agg MPLCONFIGDIR=volatile_issue24_work/mpl \
  volatile_issue24_work/venv/bin/python3 -m unittest discover -s tests
```

関連61件は29.391秒で成功。全154件は71.645秒、149成功、上記と同じ既存5エラー。変更前109件のログとエラーID・例外文言を照合した。全試験成功とは扱わない。Pandas/Polarsの全件終了・保存再読込、台帳の二重登録防止、分割確定障害のロールバック、旧JSON/pickleの欠落保持、任意の名前・UUID、ページをまたぐ祖先、登録時刻を検証した。

仮想プロトコルをCP-SAT（上限1秒、worker 1、seed 0）で実行し、8 Experiment・24作業・3分割が59分で全件終了した。JSON/pickleの最終保存を再読込して描画し、掲載JSONから別プロセスで再生成した全9ファイルがバイト一致した。全体図・家系2図・実ブラウザ画面を目視し、入力保存の親子と装置の予約非重複も検査した。

検証用 Playwright 1.58.0、Chromium 145.0.7632.6 で123項目成功。HTMLだけを別ディレクトリへコピーして開き、全8件×3範囲、全8箱と全24作業のクリック・Enter/Space、全体への解除、390px幅を確認した。SVG ID重複なし、ページエラーなし、HTML自体の読込み以外の通信なし。日本語UIのスクリーンショットには作業環境のWindowsフォントをfontconfigで参照した。フォント本体を配布しない。

有用な失敗と修正:

- 旧HTML単独では外部JavaScriptが読めず選択件数0、同梱JSありではB2の14要素を選択できることを実ブラウザで再現。新しいHTMLにJavaScript・データ・SVGを内包した。
- SVG文字と目盛線によるクリック妨害を実操作で検出。文字のポインター設定と作業棒の描画順序を修正した。選択対象外の親子の線も薄くした。
- Ubuntu 26.04 が検証用Playwrightの対応範囲外だったため、24.04向けビルドを明示し、不足したnspr/nss/asoundのライブラリを`/tmp`に展開して起動した。システムへインストールせず、製品依存を増やしていない。ブラウザのプロセス通信はサンドボックスで拒否されたため、承認されたローカル検証として実行した。

[環境・入力・出力ハッシュと確認結果](../design/experiment_retrospective/verification.json)を保存した。計画変更の全履歴、待機理由、実機の実績、利用者の比較評価は今回の対象外。旧保存に失われた作業を後付けで復元しない。保存台帳の追加後の容量測定は未実施。既存の容量評価は`a719506`の固定結果として保持する。

SVGの追加コミット時にMatplotlibのパス出力の末尾空白を差分チェックで検出した。チェック失敗で停止しないシェルだったため進捗コミットに含まれたが、描画器で行末を正規化して修正した。最終確認は`set -e`で失敗時に停止する。SVGのXML妥当性、PNG/HTML/データが変わらないこと、再生成9ファイル一致、可視化関連7件の成功、差分チェック成功を確認し、記録とハッシュを更新した。
