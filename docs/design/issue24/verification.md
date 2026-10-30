# Issue #24の要件更新・比較図の検証

2026-10-30 JST。起点はmain
`2140ab1919adb453122fd964cfdb3a7c216c74b9`、
作業ブランチは`docs/24-experiment-lifecycle`。
Issueの要件更新、3つの概念の比較図、再生成手順を扱う。
Experimentの自己消去・試料分割の機能実装は行っていない。

## 範囲と設計上の区別

- 要件：試料分割・履歴継承と、終了したExperimentの自己消去を実装対象にする。
- 要件：自己消去は管理対象のExperimentの削除。UUID・履歴・終了理由は別記録に残す。
- 要件：分割後の子は独立したExperiment。親を削除し、全子を一括登録する。
- 要件：保留は通常登録にも既定で適用する。機械未登録は該当Experimentだけを装置待ちにし、他の実験は継続する。
- 要件：`UNKNOWN`は警告と探索時間上限の倍増再試行を既定とし、再試行なしのSTOPも選択可能にする。上限内で計画が得られない場合、制約矛盾、不正仕様は全体STOPとする。
- 要件：Issueは単独で読める本文へ更新する。初回の比較図は確認用として再生成し、掲載承認後にIssue本文へ追加する。
- 既存仕様：通常の状態遷移は次のState名を返し、同じUUIDで進む。削除は別途明示的なAPI呼出し。
- 設計方針：既存のState名や空Task群だけで一律に終了を判定しない。明示的な終了契約を定義する。
- 未確定：終了API、継承履歴をStateへ渡す方法、子の初期観測・Stateの生成規則、分割量・容器IDの範囲。
- 未確定：探索時間・再試行回数上限の具体値、装置待ちから再計画する契機、STOPからの再開方法、実行中Taskの完了確認・記録。数値や自動再開の動作は選定していない。
- 評価条件：Issue #24の従来要件である計画全体1秒／60秒の評価を保持し、Issue #23の探索上限1秒／60秒とは区別した。計測境界、探索上限、反復数、費用上限は実行前に確定する。

## 確認した入力・根拠

Issue #24の更新前本文・コメントと、Issue #23の最新本文を`gh issue view`で取得した。
GitHubのmainのSHAも取得し、ローカルの起点と一致することを確認した。

`gems_python/multi_machine_problem_interval_task/transition_manager.py`の
`execute_one_step`、`determine_next_state_name`、`add_experiment`、
`delete_experiment_with_experiment_uuid`、`update_shared_variable_history`を確認した。
終了後のExperiment削除が通常の更新処理に含まれていないこと、
空の次Task群は追加されないがExperimentは一覧から削除されないこと、
個別追加ごとに再計画することをコードから確認した。
これらはコードの読み取りによる確認であり、新しい挙動の実行試験ではない。

論文の試料分割の制約は、既存の
`2667348:docs/limitations_and_unresolved_issues_2026-10-29.md`のL2と
`2667348:docs/scheduling_capacity_and_experimental_split_2026-10-29.md`第3節を参照した。
前者は元PDFの誌面p.1423・物理13ページとPDFのSHA-256を記録している。
公開本文URLの2025年・2026年経路を`curl --fail --location`で取得したが、
両方ともHTTP 403だったため、本作業では論文本文を再取得できていない。
自己消去を論文に明記された制約としては記載していない。

## 検証コマンドと結果

検証では既存の`/tmp/gems-result-latency-venv/bin/python`を使用した。
Python 3.14.4、Pillow 13.0.0。

```bash
/tmp/gems-result-latency-venv/bin/python \
  scripts/render_experiment_lifecycle_comparison.py \
  --font /tmp/gems-issue24-diagrams/NotoSansCJKjp-Regular.otf \
  --output docs/design/issue24/figure

/tmp/gems-result-latency-venv/bin/python -m py_compile \
  scripts/render_experiment_lifecycle_comparison.py

/tmp/gems-result-latency-venv/bin/python \
  scripts/render_experiment_lifecycle_comparison.py \
  --font /tmp/gems-issue24-diagrams/NotoSansCJKjp-Regular.otf \
  --output /tmp/gems-issue24-diagrams/three-way-reproduction

git diff --check
```

- 生成PNGを実際に表示し、日本語、3列の対応、矢印、親履歴と子の観測の表示を確認した。
- Pillowの`Image.verify()`でPNGを検査した。寸法は2400×1340。
- 独立した出力先で再生成し、PNGと`metadata.json`のバイト単位の一致を確認した。
- PNGのSHA-256：
  `97b9e72c8fab2de28fcc4897189a5550df495e7458b7a06dd9976083cf314c7d`。
- 使用フォントのSHA-256：
  `68a3fc98800b2a27b371f2fb79991daf3633bd89309d4ffaa6946fd587f375b5`。
- 固定commitのフォントを再取得し、使用したフォントとバイト単位で一致した。
- 構文検査・差分の空白検査は成功。生成された構文検査キャッシュは除去した。
- 本番機能は未変更であり、その実装試験・既存テスト全体の再実行はしていない。

Gitの作者情報が未設定で、初回コミットは`Author identity unknown`となった。
直近の文書コミットで使われている`Codex <codex@openai.com>`を確認し、
`git -c user.name=Codex -c user.email=codex@openai.com commit ...`で
そのコマンドだけに同じ作者情報を指定してコミットした。永続的なGit設定は変更していない。

初回は画像を掲載しない範囲だったため、生成物を
`/tmp/gems-issue24-diagrams/three-way/`へ移動した。
単純な`Path.rename`は異なるファイルシステム間のため`EXDEV`となり、
`shutil.move`で移動を完了した。再生成物も別の出力先に保持した。
初回はフォント・画像をリポジトリへ追加せず、メタデータのみを
[figure_metadata.json](figure_metadata.json)に記録した。
比較図の掲載時にPNGを追加した。フォントは同梱せず、固定フォントとスクリプトから再生成できる。

## 初回の要件本文の反映

書換え直前にIssue #24を再取得し、準備中に本文・タイトル・コメントが変更されていないことを確認した。

```bash
gh issue edit 24 --repo bioinfo-tsukuba/GEMS-python \
  --title '未実装機能の補完：試料分割・履歴継承と終了したExperimentの自己消去' \
  --body-file docs/design/issue24/issue.md

gh issue view 24 --repo bioinfo-tsukuba/GEMS-python \
  --json number,title,body,comments,url,state
```

初回更新後のGitHub本文はその時点の[issue.md](issue.md)と完全一致し、タイトルも一致した。
IssueはOPENのまま、コメントは変更していない。
本文中に会話への言及、Mermaid、画像のMarkdown・HTMLが含まれないことを検査した。
図の添付・コメント投稿は行っていない。

- 更新前本文SHA-256：
  `5312bf70014f588cdc3b237adc93a354b353b54d3c0fee408c2016d0e95bc36d`
- 更新後本文SHA-256：
  `63c88c859f365c985e97925f8946e4d5654b9af2253f2000f977d5b6ccc7b994`

## リポジトリへの保存

要件原稿、生成スクリプト、検証記録、図のメタデータを
`8513cce`としてコミットし、`docs/24-experiment-lifecycle`へpushした。
文書PR [#26](https://github.com/bioinfo-tsukuba/GEMS-python/pull/26)を作成した。
初回のPRは要件と再現資料だけを扱い、画像・フォント・機能実装を含めなかった。
掲載承認後に比較図PNGを追加した。フォント・機能実装は含めない。
マージはしていない。

## 比較図の掲載（2026-10-30）

掲載対象は、従来・自己消去・試料分割を左右に並べた承認済みPNG。
画像のSHA-256は初回生成・表示確認時と一致した。
`32a5374e277a2bc9d02b91164cc29b5f8241d91b`で
[figure/comparison.png](figure/comparison.png)を保存し、文書ブランチへpushした。
Issue本文では、この固定コミットの公開PNGを参照する。

掲載前に最新のIssue本文・タイトル・コメントを取得し、準備中の同時更新がないことを確認した。
本文の差分が「概念比較」の画像と説明文だけで、他の要件に変更がないことを確認した。

```bash
curl --fail --location \
  --output /tmp/gems-issue24-diagrams/published-comparison.png \
  https://raw.githubusercontent.com/bioinfo-tsukuba/GEMS-python/32a5374e277a2bc9d02b91164cc29b5f8241d91b/docs/design/issue24/figure/comparison.png

gh api markdown --input /tmp/gems-issue24-diagrams/markdown-input.json

gh issue edit 24 --repo bioinfo-tsukuba/GEMS-python \
  --body-file docs/design/issue24/issue.md

gh issue view 24 --repo bioinfo-tsukuba/GEMS-python \
  --json number,title,body,comments,state,url
```

`markdown-input.json`は更新する本文、`mode: gfm`、
`context: bioinfo-tsukuba/GEMS-python`を指定したGitHub Markdown APIの入力。

- 固定コミットの公開PNGをダウンロードし、承認済みローカル画像とバイト単位で一致した。
- GitHub Markdown APIのHTMLに、対応するaltを持つ比較画像が1つ出力された。
- 掲載後のIssue本文を再取得し、画像・説明文を含む[issue.md](issue.md)と完全一致した。
- タイトル・OPEN状態・コメントは掲載前と一致した。
- 掲載後本文SHA-256：
  `a055babdb31dec20a8294db9eba7ae7122537f2666053a2f608ad1b9bd7ffaf4`
- 図のPNG・メタデータを照合し、文書リンクと差分の空白の検査が成功した。
- 機能実装・生成スクリプトは変更していないため、実験処理のテストや図の再生成は繰り返していない。

## 登録保留・探索再試行・STOP方針の明文化（2026-10-30）

作業起点は文書ブランチの
`0e70f0b6e292a0b7980277b9b289d3dcd2e5c6f6`。
Issue #24の「C. 失敗、再試行、保存・再開」を、
通常登録と子の登録に共通する保留、入力検査・計画結果、STOPと診断、
再送と保存・再開に分けた。目的、分割要件、実装・検証順序、完了条件、
未確定事項も整合させた。

今回反映した方針はユーザーの指定・承認に基づく。
初回Task群全体を検査し、機械未登録では該当Experimentだけを保留する。
機械が使用中でも将来の有効な計画が得られる場合は装置待ちにしない。
`OPTIMAL`／`FEASIBLE`は採用し、`UNKNOWN`は警告と`T → 2T → 4T …`の探索上限倍増を既定とする。
再試行なしでSTOPする設定も設け、上限内で有効な計画が得られなければ全体STOPする。
制約矛盾と不正仕様は理由を示したErrorで全体STOPする。
全体STOPは管理ループと新しいTaskの開始を止め、実行中装置操作の強制中断を意味しない。

従来の「適合機械なしでは子を登録しない」「再計画失敗では全子を保留」は、
原因別の方針に置き換えた。機械未登録でも全子の一括登録と親の削除は確定し、
該当する子だけを装置待ちにする。確定後の再試行・STOPでも出自と全子登録を保持する。
再試行待ちのTask群は再計画対象であり、機械未登録の未開始Task群と区別する。
保留理由・STOP状態・再試行設定と進捗の保存を完了条件に追加した。

`transition_manager.py`の`add_experiment`と
`cp_sat_scheduler.py`の`_solve`を再確認した。
現在の登録は一覧追加後に計画し、solverは`OPTIMAL`／`FEASIBLE`以外で例外を出す。
今回の保留・倍増再試行・原因別STOPは追加する仕様であり、実装済みとは扱わない。
上限の具体値は未指定のまま、実装前に確定する設定項目として残した。
診断で矛盾する制約を特定できない場合は、その旨と再現入力を記録する要件を明示した。
規模評価では各探索上限・実測時間・累積時間を記録し、探索上限を増やした再試行を初回の時間条件の成功と数えない。

```bash
gh issue view 24 --repo bioinfo-tsukuba/GEMS-python \
  --json number,title,body,comments,url,state

gh pr view 26 --repo bioinfo-tsukuba/GEMS-python \
  --json number,title,body,state,headRefName,headRefOid,baseRefName,url,files

gh api markdown --input /tmp/gems-issue24-diagrams/policy-markdown-input.json

gh issue edit 24 --repo bioinfo-tsukuba/GEMS-python \
  --body-file docs/design/issue24/issue.md

git diff --check
```

Markdown APIの入力は[issue.md](issue.md)の全文、
`mode: gfm`、`context: bioinfo-tsukuba/GEMS-python`。
更新直前にIssueを再取得して、準備中の同時更新がないことを確認した。
更新後も再取得して実際の本文を検査した。

- GitHub Markdownで比較表と判定表の2表、承認済み比較画像1つの出力を確認した。
- GitHub本文は原稿と完全一致し、タイトル・コメント・OPEN状態を維持した。
- 更新前本文SHA-256：
  `a055babdb31dec20a8294db9eba7ae7122537f2666053a2f608ad1b9bd7ffaf4`
- 更新後本文SHA-256：
  `40e7d099328da8899249a3e1274ddf838e27148336d7fd597b58dc4f03999a74`
- 図の固定commit参照、PNGとメタデータのハッシュ、文書の相対リンクを検査した。
- 会話への言及がなく、実装の完了チェックが未完了のままであることを確認した。
- 変更は要件原稿・README・本検証記録のみ。機能実装・比較図・生成スクリプトを変更していないため、機能テストや図の再生成は行っていない。

## 残る作業

自己消去・分割・保留・原因別STOP・保存世代の整合性・倍増再試行・規模評価はいずれも未実装・未検証。
Issueのチェックリストは未完了のままとする。
文書・生成スクリプトのPRではIssue #24を閉じない。
作業ブランチは文書PRのマージ後に削除する。
