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
- 要件：Issueは単独で読める本文へ更新する。比較図は確認用として再生成し、画像は掲載しない。
- 既存仕様：通常の状態遷移は次のState名を返し、同じUUIDで進む。削除は別途明示的なAPI呼出し。
- 設計方針：既存のState名や空Task群だけで一律に終了を判定しない。明示的な終了契約を定義する。
- 未確定：終了API、継承履歴をStateへ渡す方法、子の初期観測・Stateの生成規則、分割量・容器IDの範囲。
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
そのコマンドだけに同じ作者情報を指定する。永続的なGit設定は変更しない。

画像は掲載しない範囲に合わせ、初回生成物を
`/tmp/gems-issue24-diagrams/three-way/`へ移動した。
単純な`Path.rename`は異なるファイルシステム間のため`EXDEV`となり、
`shutil.move`で移動を完了した。再生成物も別の出力先に保持した。
フォント・画像はリポジトリへ追加せず、メタデータのみを
[figure_metadata.json](figure_metadata.json)に記録する。
一時出力先の永続性は保証しないため、保存したスクリプトと固定フォントから再生成する。

## Issueへの反映

書換え直前にIssue #24を再取得し、準備中に本文・タイトル・コメントが変更されていないことを確認した。

```bash
gh issue edit 24 --repo bioinfo-tsukuba/GEMS-python \
  --title '未実装機能の補完：試料分割・履歴継承と終了したExperimentの自己消去' \
  --body-file docs/design/issue24/issue.md

gh issue view 24 --repo bioinfo-tsukuba/GEMS-python \
  --json number,title,body,comments,url,state
```

更新後のGitHub本文は[issue.md](issue.md)と完全一致し、タイトルも一致した。
IssueはOPENのまま、コメントは変更していない。
本文中に会話への言及、Mermaid、画像のMarkdown・HTMLが含まれないことを検査した。
図の添付・コメント投稿は行っていない。

- 更新前本文SHA-256：
  `5312bf70014f588cdc3b237adc93a354b353b54d3c0fee408c2016d0e95bc36d`
- 更新後本文SHA-256：
  `63c88c859f365c985e97925f8946e4d5654b9af2253f2000f977d5b6ccc7b994`

## 残る作業

自己消去・分割・保存世代の整合性・再試行・規模評価はいずれも未実装・未検証。
Issueのチェックリストは未完了のままとする。
文書・生成スクリプトのPRではIssue #24を閉じない。
作業ブランチは文書PRのマージ後に削除する。
