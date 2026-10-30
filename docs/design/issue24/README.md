# Experimentの自己消去・試料分割の設計資料

Issue [#24](https://github.com/bioinfo-tsukuba/GEMS-python/issues/24)の要件原稿は
[issue.md](issue.md)。終了したExperimentを管理対象から自動削除する機能と、
親の履歴を引き継ぐ独立した子Experimentを一括登録する試料分割を対象とする。
機能は未実装であり、この資料と比較図は実装完了の報告ではない。
資料と再生成スクリプトは文書PR
[#26](https://github.com/bioinfo-tsukuba/GEMS-python/pull/26)で管理する。

比較図は次の3つを左右に並べる。

1. 従来：同じExperimentが状態遷移し、終了後の削除は手動。
2. 自己消去：終了したExperimentを自動削除し、別の終了記録に履歴を残す。
3. 分割：親を削除し、親の履歴を出自として持つ独立した子を登録する。

`H`は従来のExperimentまたは分割前の親の履歴、
`H1`・`H2`は分割後に各子へ追加する観測履歴を表す。
親履歴の出自参照と、子自身の観測を区別する。
DataFrameの物理的なコピー方法やStateへの入力APIを確定した図ではない。

比較図は[comparison.png](figure/comparison.png)として保存し、Issue #24の本文に掲載する。
Issueからは固定commit `32a5374e277a2bc9d02b91164cc29b5f8241d91b`の画像を参照する。
このディレクトリには原稿、画像、生成結果のメタデータ、検証記録を保持する。
フォントは同梱しない。

## 再生成

Pillowと日本語フォントを使用する。Pillowは既存のMatplotlibの依存であり、
新たな実行時依存は追加しない。フォントは同梱せず、次の固定版を使用する。

- Noto Sans CJK JP Regular、`notofonts/noto-cjk`
- 固定commit: `165c01b46ea533872e002e0785ff17e44f6d97d8`
- フォントSHA-256: `68a3fc98800b2a27b371f2fb79991daf3633bd89309d4ffaa6946fd587f375b5`
- フォントの利用条件: 同リポジトリの`Sans/LICENSE`（SIL Open Font License 1.1）

```bash
mkdir -p /tmp/gems-issue24-font
curl --fail --location \
  --output /tmp/gems-issue24-font/NotoSansCJKjp-Regular.otf \
  https://raw.githubusercontent.com/notofonts/noto-cjk/165c01b46ea533872e002e0785ff17e44f6d97d8/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf
sha256sum /tmp/gems-issue24-font/NotoSansCJKjp-Regular.otf

poetry run python scripts/render_experiment_lifecycle_comparison.py \
  --font /tmp/gems-issue24-font/NotoSansCJKjp-Regular.otf \
  --output /tmp/gems-issue24-comparison
```

出力は`comparison.png`（2400×1340）と`metadata.json`。
既存の出力ディレクトリを上書きしないため、再実行時は新しい出力先を指定する。
同じ画素・ファイルを再現するための検証環境はPython 3.14.4、Pillow 13.0.0。
異なるPillow・フォント等でファイルのハッシュが一致することは保証しない。

実行コマンド、検証結果、現行仕様と論文の確認根拠、未検証範囲は
[verification.md](verification.md)に記録する。
