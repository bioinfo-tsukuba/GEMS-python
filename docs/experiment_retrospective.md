# 終了したExperimentを振り返る

[README](../README.md#review-completed-experiments) → この使い方 → [動く例](../examples/experiment_retrospective.py)。
[終了・分割・保留の仕様](experiment_lifecycle.md)と[保存・再開](persistence.md)も参照できる。

## 自分の実験から描く

全Experimentが終了した管理オブジェクトで呼ぶ。

```python
lab.render_retrospective("review", time_unit="min")
```

`review/review.html`をブラウザで開く。このHTMLだけで選択操作が動く。GitHubのファイル表示ではHTMLはソース表示になるので、ダウンロードしたファイルを開く。

出力先は新しいディレクトリを指定する。既存の出力は上書きしない。活動中・保留中のExperimentが残る場合は終了後に呼ぶ。

## まず例を試す

リポジトリのルートで実行する。

```bash
MPLBACKEND=Agg poetry run python -m scripts.run_experiment_retrospective_example \
  --output /tmp/gems-review-example
```

`/tmp/gems-review-example/figures/review.html`を開く。
[例のコード](../examples/experiment_retrospective.py)は2つの親から子・孫へ分割し、2台の装置を共有する仮想実験。全件終了させ、JSON/pickleで保存し、再読込して図を生成する。必要な入力と図は実行時に生成する。

Polarsで試す場合は`--backend polars`を付ける。同じ場所へ再実行する場合は別の出力先を指定する。

## 保存済みの実験から描く

実験を再実行せず、最終スナップショットを読む。

```bash
MPLBACKEND=Agg poetry run python -m scripts.render_experiment_retrospective \
  --snapshot /path/to/snapshot --output /path/to/new-review
```

`--snapshot`は`save_snapshot`の保存ディレクトリ、または`step_00000000`などの保存ステップ。JSONだけを読む場合は`--source json`、pickleだけなら`--source pickle`を指定する。既定の`auto`は両形式の整合性を検査する。

保存したStateクラスを読み込めるコードも必要。例のStateは`examples.experiment_retrospective`で定義しているため、別プロセスのこのコマンドでも読み込める。

## 図の読み方と選択操作

| 出力 | 読み方 |
| --- | --- |
| `overview-01.png` / `.svg` | 全体の時間図。薄い帯は登録から終了まで、濃い棒は作業の最終予約。◆は分割、■は終了。線で親子を結ぶ。 |
| `family-001-01.png` / `.svg` | 家系ごとの親子図と時間図。同じ行が同じExperiment。全体図と時間範囲を揃える。 |
| `review.html` | 親子図と装置図の選択連動。実験名、箱、作業の棒から選ぶ。「祖先・子孫」「家系」「選択のみ」を切り替え、「全体」で解除する。 |
| `review-data.json` / `manifest.json` | 描画に使った記録と、入力・出力・描画器のハッシュ。 |

HTMLの図形はTabで移動してEnter / Spaceでも選べる。選択した対象を両図で強調し、他の対象は薄く残す。作業の詳細はマウスを重ねて確認でき、読み上げ用のラベルも持つ。

件数が増えると全体図・家系別図は24件ずつに分かれる。ページをまたぐ祖先は文脈として再表示する。HTMLでは全件を表示する。同名のExperimentには表示上の連番を付ける。

## 日本語と時間単位

日本語の実験名には、日本語を含むフォントを指定する。

```python
lab.render_retrospective(
    "review",
    font="/path/to/JapaneseFont.ttf",
    time_unit="min",
)
```

CLIでは`--font /path/to/JapaneseFont.ttf --time-unit min`を付ける。フォント指定を省略するとMatplotlibの既定フォントを使う。SVGは文字をパス化する。

時間単位はプロトコルの論理時間に合わせて指定する。既定は`min`で、数値の単位変換は行わない。

## 記録の範囲

棒は完了時点の**最終予約**であり、実機の実測開始・終了ではない。結果受領時の管理論理時刻を実測終了として扱わず、作業間の空白の理由も推測しない。

登録名・親UUID・登録論理時刻と、完了した作業の最終予約・装置を`retrospective_trace`に保存する。Experiment / TaskGroupが活動一覧から消えても再読込して描ける。親の継承履歴を子の作業として重複表示しない。

旧保存に欠落した時刻・作業は生成しない。部分的な図を出力し、警告と`complete: false`で不足を知らせる。計画変更の全履歴、待機理由、実機の実績は対象外。記録は完了作業数に応じて増える。

## 開発・検証資料

機能の回帰テストは`tests/test_retrospective.py`。操作確認には[ブラウザ検証スクリプト](../scripts/verify_experiment_retrospective_browser.py)を使える。PlaywrightとChromiumは任意の検証用環境に入れ、製品の依存には含めない。

過去の図・保存データ・詳細な検証記録は[固定コミットの資料](https://github.com/bioinfo-tsukuba/GEMS-python/blob/cd78405f3c66061c3b71b72623a83540aaca83fb/docs/design/experiment_retrospective/README.md)で参照できる。[分割後の容量評価](https://github.com/bioinfo-tsukuba/GEMS-python/blob/8ca9080bf62a4e6c2b70fe73a0f830c5f627b183/docs/benchmarks/experiment_lifecycle_capacity_2026-10-30/README.md)は旧測定対象の結果であり、可視化の保存台帳を追加した後の容量測定ではない。
