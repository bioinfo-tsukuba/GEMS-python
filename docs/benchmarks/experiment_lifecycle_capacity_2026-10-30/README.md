# Issue #24 規模評価の保存資料

計測対象は `a719506311b75dcbc55598dd1055f31b9c2a9a00`。
344試行・172比較組・344保存を照合し、欠落・重複・スキップなしで完了した。
時間内88、有効な時間超過250、UNKNOWN 6。補足初期化も344件、失敗なし。
最終の隔離全試験は147件・70.136秒、比較元と同じ既存5エラーのみ。
表・図等の6ファイルを別出力先で再生成してバイト一致を確認した。
実装PR #27とは別の保存ブランチ `archive/24-experiment-lifecycle-capacity` に保持する。
資料の固定参照のため、実装PRのマージ後も保存ブランチを残す。

結論・表・図は [集計レポート](statistics/README.md)。
圧縮ファイルの全内容のハッシュ照合は [archive_verification.json](archive_verification.json)。
入力・出力の照合は [verification.json](verification.json)、
反復の判定は [boundaries.json](boundaries.json) に記録する。
実装と受入確認は [検証記録](../issue24_verification.md)、
契約は [ライフサイクル仕様](../../experiment_lifecycle.md) を参照する。

## 内容と入力規則

- `raw.tar.gz`: 全試行のjob、入力、出力、solver報告、診断、同一世代のJSON/pickle保存、
  初期化の補足計測、設定・環境・再開記録を圧縮した資料。
- `raw-manifest.json`: 圧縮前の各ファイルのSHA-256。`SHA256SUMS.txt`は公開資料の照合用。
- `statistics/`: 試行CSV、条件別CSV、集計JSON、README、PNG/SVG。
- `test-logs/`: 基準・実装後の試験と独立検査・再生成の記録。既存5エラーも保持する。

親は空群のStateで、分割前の履歴に`3 × group_interval`行の決定的な完了観測を置く。
値は親のindex、時刻は0からの連番とする。子は初期観測1行と独立した新しいStateを持つ。
2世代目の前処理は論理時刻100からの合成観測を追加する。
これらは評価用の選択であり、生物学的・物理的な過去の作業を再演した時間ではない。

子の現在の群は3工程、処理時間5/3/4分、間隔2/2/1分。
機械構成は同型競合2台、異種工程2/2/1台、割当候補多数8台、複雑ペナルティ2台、
固定予約を含む再計画2台。ペナルティと固定予約の生成は#23の計測器を再利用する。
入力生成、正確な固定予約、初期条件と由来は保存元commitと各`input.json`で確認できる。

群間隔・世代数を変えると、履歴量だけでなく論理時刻・周期的費用の位相も変わり得る。
条件間の差を純粋な履歴管理の費用として解釈しない。
各条件内の分割あり／flatの物理条件はハッシュで一致を検査する。
flatも同じfixtureから作るため、前処理・ピークRSSを「分割を一度も行わない実装」の費用とは扱わない。

## 保存・再開と計測範囲

初期54試行はtmpfs。一時環境の消失後はワークスペースのext4へ再開した。
実装全ファイル、基礎計測器、入力生成とworkerのAST、Python・依存freezeを照合している。
保存・全工程時間は保存先の異なる反復を混ぜて集計しない。
一部の計測中に追加受入試験・独立検査を実行したことは`auxiliary_verification.json`に記録する。

最大時間は予備測定の最初の開始から21600秒で、再開後も終了期限を延長しない。
予備commit `eb0cee3`の生資料は退避前に失われており、最終評価には使わない。
スケジューラの変更は行わず、上限付近の最大走査成功点が3反復で安定しなければ
小さい走査成功点も確認する。過去の走査・反復は保持する。

計画全体は管理層の入口から検査済み反映まで。探索上限1秒/60秒、再試行無効。
有効な時間超過、UNKNOWN、制約矛盾、検査失敗を区別する。
初期化の補足計測は元と同じ入力ハッシュで、別プロセス・solverなしで行い、
元試行の全工程時間には加算しない。各試行の全工程時間自体にはその試行の初期化が含まれる。

静的な合成入力の記述統計であり、任意の実験手順・設備・継続更新・物理操作を保証しない。
自己消去、有限の培養・取り分け、2世代、保留・STOP、保存・別プロセス再開は別の受入試験で確認する。

## 照合・再生成

保存資料を別ディレクトリへ取り出してから実行する。以下はsolverを呼ばない。
Python 3.14.4と記録された`environment.txt`の依存を使う。
検査・集計スクリプトと保存クラスのコードはこの保存ブランチに含まれる。

```bash
sha256sum --check SHA256SUMS.txt
mkdir unpacked
tar -xzf raw.tar.gz -C unpacked
python3.14 -m venv /tmp/gems-issue24-replay-env
/tmp/gems-issue24-replay-env/bin/python -m pip install -r unpacked/environment.txt

# リポジトリのルートから実行し、展開先を指定する。
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/gems-issue24-replay-mpl \
  /tmp/gems-issue24-replay-env/bin/python -m scripts.verify_experiment_lifecycle_capacity \
  /path/to/unpacked
MPLCONFIGDIR=/tmp/gems-issue24-replay-mpl \
  /tmp/gems-issue24-replay-env/bin/python -m scripts.analyze_experiment_lifecycle_capacity \
  /path/to/unpacked --output /tmp/gems-issue24-replay-statistics
```

圧縮前の全ファイルのハッシュは`raw-manifest.json`と照合する。
図のSVG日時・IDを固定し、同じ環境でCSV・JSON・README・PNG・SVGを再生成する。
再探索は時間上限・実行費用を伴う別操作であり、過去の出力の再生成とは分ける。
