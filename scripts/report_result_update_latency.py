"""Tabulate and plot saved received-result latency records without running a solver."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_scheduler_capacity import describe


def report(root):
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    inputs = json.loads((root / "inputs.json").read_text())
    env = json.loads((root / "environment.json").read_text())
    metrics = [
        "response_seconds", "progress_seconds", "update_api_seconds", "json_seconds",
        "search_seconds", "non_search_seconds",
    ]
    flat = [
        {**{key: row[key] for key in
            ("pattern", "size", "search_limit_seconds", "repetition", "outcome",
             "input_hash", "response_sha256", "response_bytes", "solver_calls") + tuple(metrics)},
         "completes_group": row["event"]["completes_group"]}
        for row in rows
    ]
    with (root / "trial_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    condition_stats = []
    for pattern in inputs:
        for limit in [1.0, 60.0]:
            selected = [r for r in rows if r["pattern"] == pattern
                        and r["search_limit_seconds"] == limit]
            success = [r for r in selected if r["outcome"] == "VALID"]
            failed = [r for r in selected if r["outcome"] != "VALID"]
            item = {
                "pattern": pattern, "size": inputs[pattern]["size"],
                "search_limit_seconds": limit, "runs": len(selected), "valid": len(success),
                "failed_response_seconds": describe(r["response_seconds"] for r in failed),
            }
            for metric in metrics:
                item[metric] = describe(r[metric] for r in success)
            condition_stats.append(item)
    (root / "statistics.json").write_text(json.dumps(condition_stats, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    patterns = list(inputs)
    names = ["contention", "heterogeneous", "candidates", "penalties", "replanning"]
    for ax, limit in zip(axes, [1.0, 60.0]):
        for position, pattern in enumerate(patterns):
            selected = [r for r in rows if r["pattern"] == pattern
                        and r["search_limit_seconds"] == limit]
            valid = [r["response_seconds"] for r in selected if r["outcome"] == "VALID"]
            failed = [r["response_seconds"] for r in selected if r["outcome"] != "VALID"]
            for values, color, marker in [(valid, "#0072B2", "o"), (failed, "#D55E00", "x")]:
                offsets = np.linspace(-.12, .12, len(values)) if len(values) > 1 else [0]
                ax.scatter(position + np.array(offsets[:len(values)]), values,
                           color=color, marker=marker, s=35)
            if valid:
                stats = describe(valid)
                ax.errorbar(position, stats["median"],
                            yerr=[[stats["median"] - stats["min"]],
                                  [stats["max"] - stats["median"]]],
                            fmt="D", color="#222222", capsize=5, markersize=4)
            ax.text(position, max(valid + failed, default=0) + limit * .025,
                    f"{len(valid)}/{len(selected)} valid", ha="center", fontsize=9)
        ax.set_xticks(range(len(patterns)),
                      [f"{name}\nn={inputs[p]['size']}" for name, p in zip(names, patterns)],
                      rotation=22, ha="right", fontsize=9)
        ax.set_ylim(0, limit * 1.12)
        ax.set_ylabel("Real seconds to response JSON")
        ax.set_title(f"{limit:g}-second CP-SAT limit; one search per received result")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"Received result -> schedule JSON: {env['repetitions']} repeats of the first completion event")
    handles = [
        plt.Line2D([], [], color="#0072B2", marker="o", ls="", label="Valid schedule"),
        plt.Line2D([], [], color="#D55E00", marker="x", ls="", label="UNKNOWN / error response"),
        plt.Line2D([], [], color="#222222", marker="D", ls="", label="Success median; min–max"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9)
    fig.savefig(root / "response_latency.png", dpi=170)
    fig.savefig(root / "response_latency.svg")
    plt.close(fig)
    lines = [
        "# 実験結果受領→スケジュールJSON生成",
        "",
        f"実測ソース: `{env['commit']}`。保存計画の参照: `{env['source_archive_commit']}`。",
        f"開始: `{env['started_at_utc']}`。CPU: {env['cpu']}、CP-SAT worker 1 / seed 0。",
        "依存版とインストール全一覧は環境ファイルへ記録しました。",
        "",
        "受信済みDataFrameをアプリへ渡してから、開始済み仕事の固定、履歴更新・完了処理、"
        "必要な状態遷移・タスク生成、CP-SAT再計画、全スケジュールJSON生成までを計時しました。",
        "起動・入力準備・機器通信・描画・ファイル保存は含みません。デバッグ出力はメモリ上に捕捉します。"
        "既存更新APIを一度呼び、結果受領前の再計画を挟まず、探索は各応答1回です。",
        "",
        "今回の入力は、前回の有効計画における最初の結果イベントです。"
        "すべて群途中のため、状態遷移・次群生成の分岐は今回の実測では実行されていません。"
        "その分岐は別の小規模回帰テストで確認しました。複雑な実験解析処理も模擬していません。",
        "",
        "![結果受領後の応答時間](response_latency.png)",
        "",
        "| 負荷 | 規模 | 探索秒 | 成功/試行 | 成功応答 中央値 / p95 / 最大（秒） | うち探索 中央値（秒） | その他 中央値（秒） | 失敗応答 中央値（秒） |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in condition_stats:
        success = row["response_seconds"]
        failure = row["failed_response_seconds"]
        def value(stats, key="median"):
            return f"{stats[key]:.6f}" if stats["n"] else "—"
        lines.append(
            f"| {row['pattern']} | {row['size']} | {row['search_limit_seconds']:g} | "
            f"{row['valid']}/{row['runs']} | {value(success)} / {value(success, 'p95')} / "
            f"{value(success, 'max')} | {value(row['search_seconds'])} | "
            f"{value(row['non_search_seconds'])} | {value(failure)} |")
    lines += [
        "",
        f"同じ受信済み結果・同じ計画を1/60秒に渡した各{env['repetitions']}反復の記述統計です。"
        "p95は少数点の線形補間で、将来の保証ではありません。"
        "中央値の内訳を別々に集計するため、内訳中央値の和は総時間中央値と厳密には一致しない場合があります。",
        "UNKNOWNの時間はスケジュール生成成功の時間へ混ぜていません。",
        "",
        "旧運用の約2/120秒は受領前の探索も含みます。本測定は受領後の1回の探索を計時しています。"
        "この初回成功は継続更新の成功を保証しません。",
        "",
        "trial_metrics.csvは全試行、statistics.jsonは条件別の中央値・四分位範囲・p95・"
        "最小最大・平均・標本標準偏差です。raw_results.tar.gzに入力・出力・探索・失敗・"
        "ログを保持しています。入力ハッシュ・出力ハッシュ・環境・検証も併記しています。",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/gems-latency-mpl MPLBACKEND=Agg poetry run python \\",
        "  scripts/measure_result_update_latency.py \\",
        "  --output /tmp/gems-result-latency-new --repetitions 3",
        "",
        "# 保存された結果を再集計・図示するだけの場合",
        "poetry run python -m scripts.report_result_update_latency /tmp/gems-result-latency-new",
        "poetry run python -m scripts.verify_result_update_latency /tmp/gems-result-latency-new",
        "```",
        "",
        "新規測定には約15分の探索を要します。出力先は新しいディレクトリを指定します。",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    return condition_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)
