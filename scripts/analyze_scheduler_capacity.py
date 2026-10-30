"""Describe saved capacity measurements and draw figures without running a solver."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PATTERNS = [
    "contention", "heterogeneous", "many_candidates", "mixed_penalties", "replanning",
]
LIMITS = [1.0, 60.0]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6C6C00"]
QUALITY = [
    "movable_penalty", "delay_median_minutes", "delay_p95_minutes",
    "delay_max_minutes", "machine_utilization",
]
TIMING = [
    "search_seconds", "build_seconds", "apply_seconds", "validation_seconds",
    "other_seconds", "total_seconds", "memory_peak_mib",
]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def describe(values):
    """Linear-interpolated quantiles; sample SD is undefined for one observation."""
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0}
    quantiles = np.quantile(values, [.25, .5, .75, .95], method="linear")
    return {
        "n": len(values), "min": values[0], "q1": float(quantiles[0]),
        "median": float(quantiles[1]), "q3": float(quantiles[2]),
        "p95": float(quantiles[3]), "max": values[-1],
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def condition_key(row):
    return (row["phase"], row["pattern"], float(row["search_limit_seconds"]), int(row["size"]))


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[condition_key(row)].append(row)
    return groups


def summarize_trials(rows):
    result = []
    for (phase, pattern, limit, size), selected in sorted(group_rows(rows).items()):
        valid = [row for row in selected if row["outcome"] == "VALID"]
        item = {
            "phase": phase, "pattern": pattern, "search_limit_seconds": limit,
            "size": size, "runs": len(selected), "valid": len(valid),
            "valid_fraction": len(valid) / len(selected),
            "outcomes": dict(Counter(row["outcome"] for row in selected)),
        }
        for metric in TIMING + QUALITY:
            # UNKNOWN attempts still consume time/memory; quality has no valid output.
            source = valid if metric in QUALITY else selected
            item[metric] = describe(row[metric] for row in source if row.get(metric) != "")
        result.append(item)
    return result


def pair_quality(rows):
    """Compare condition medians only when both limits used exactly the same input."""
    groups = group_rows(row for row in rows if row["phase"] == "scan")
    result = []
    for pattern in PATTERNS:
        sizes = sorted({key[3] for key in groups if key[1] == pattern})
        for size in sizes:
            first = groups.get(("scan", pattern, 1.0, size), [])
            second = groups.get(("scan", pattern, 60.0, size), [])
            if not first or not second:
                continue
            hashes = {row["input_hash"] for row in first + second}
            if len(hashes) != 1:
                raise ValueError(f"Paired input mismatch: {pattern}, {size}")
            valid = [[row for row in part if row["outcome"] == "VALID"]
                     for part in (first, second)]
            item = {
                "pattern": pattern, "size": size, "input_hash": next(iter(hashes)),
                "valid_1": len(valid[0]), "runs_1": len(first),
                "valid_60": len(valid[1]), "runs_60": len(second),
            }
            for metric in ("movable_penalty", "delay_max_minutes"):
                medians = [describe(row[metric] for row in part).get("median") for part in valid]
                item[f"{metric}_1"] = medians[0]
                item[f"{metric}_60"] = medians[1]
                item[f"{metric}_delta_60_minus_1"] = (
                    medians[1] - medians[0] if all(value is not None for value in medians) else None
                )
            result.append(item)
    return result


def summarize_operations(trials, updates):
    groups = group_rows(updates)
    result = []
    for row in trials:
        if row["phase"] != "operation":
            continue
        events = groups[condition_key(row)]
        successful = [event for event in events if event["status"] == "ok"]
        result.append({
            "pattern": row["pattern"], "size": int(row["size"]),
            "search_limit_seconds": float(row["search_limit_seconds"]),
            "outcome": row["outcome"], "successful_updates": len(successful),
            "attempted_updates": len(events), "solver_calls": int(row["solver_calls"]),
            "successful_update_seconds": describe(event["update_seconds"] for event in successful),
            "attempted_update_seconds": describe(event["update_seconds"] for event in events),
            "total_seconds": float(row["total_seconds"]),
            "memory_peak_mib": float(row["memory_peak_mib"]),
        })
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def draw_static(conditions, output):
    static = [row for row in conditions if row["phase"] == "scan"]
    lookup = {(row["pattern"], row["search_limit_seconds"], row["size"]): row for row in static}
    sizes = sorted({row["size"] for row in static})
    image = np.full((10, len(sizes)), np.nan)
    labels = [(pattern, limit) for pattern in PATTERNS for limit in LIMITS]
    fig, ax = plt.subplots(figsize=(12, 6.3), layout="constrained")
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("#e5e5e5")
    for y, (pattern, limit) in enumerate(labels):
        for x, size in enumerate(sizes):
            row = lookup.get((pattern, limit, size))
            if row:
                image[y, x] = row["valid_fraction"]
            text = f"{row['valid']}/{row['runs']}" if row else "unmeasured"
            ax.text(x, y, text, ha="center", va="center", fontsize=9,
                    color="white" if row and (image[y, x] < .2 or image[y, x] > .8) else "#333333")
    mesh = ax.imshow(image, vmin=0, vmax=1, cmap=cmap, aspect="auto", zorder=0)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_yticks(range(10), [f"{pattern} / {limit:g} s" for pattern, limit in labels])
    ax.set_xlabel("Active experiments")
    ax.set_title("Static observed success: valid plans / attempts (primary source 9e269d4)")
    fig.colorbar(mesh, ax=ax, label="Observed valid fraction")
    save_figure(fig, output, "static_success")

    for name, metrics, titles in [
        ("static_runtime", ["total_seconds", "build_seconds"],
         ["Scheduling elapsed time", "Model construction time"]),
        ("static_quality_memory", ["delay_max_minutes", "memory_peak_mib"],
         ["Maximum task delay (valid outputs only)", "Fresh-worker peak RSS (all attempts)"]),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for column, limit in enumerate(LIMITS):
            for row_index, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[row_index, column]
                for pattern, color in zip(PATTERNS, COLORS):
                    points = [lookup.get((pattern, limit, size)) for size in sizes]
                    median = [p[metric].get("median", np.nan) if p else np.nan for p in points]
                    q1 = [p[metric].get("q1", np.nan) if p else np.nan for p in points]
                    q3 = [p[metric].get("q3", np.nan) if p else np.nan for p in points]
                    ax.plot(sizes, median, "o-", label=pattern, color=color, markersize=4)
                    ax.fill_between(sizes, q1, q3, color=color, alpha=.15)
                ax.set_xscale("log", base=2)
                ax.set_xticks(sizes, sizes, fontsize=8)
                ax.set_xlabel("Active experiments")
                if metric == "delay_max_minutes":
                    ax.set_yscale("symlog", linthresh=1)
                    ax.set_ylabel("Virtual minutes (symlog; zero retained)")
                    maxima = [r[metric]["max"] for r in static if r[metric]["n"]]
                    ax.set_ylim(0, max(maxima) * 1.25)
                elif metric == "memory_peak_mib":
                    ax.set_ylabel("MiB (includes imports)")
                    minima = [r[metric]["min"] for r in static]
                    maxima = [r[metric]["max"] for r in static]
                    ax.set_ylim(min(minima) * .9, max(maxima) * 1.05)
                else:
                    ax.set_yscale("log")
                    ax.set_ylabel("Real seconds (log)")
                    if metric == "total_seconds":
                        ax.axhline(limit, color="#444444", ls=":", lw=1)
                ax.set_title(f"{title} / {limit:g}-second search limit")
                ax.grid(alpha=.2)
        handles, labels_legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc="outside lower center", ncol=3, fontsize=9)
        fig.suptitle("Primary static trials: median and Q1–Q3 band; n = 1 or 5 per condition")
        save_figure(fig, output, name)


def draw_operations(operations, confirmation, target_steps, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for limit, color, offset in [(1.0, "#0072B2", -.18), (60.0, "#D55E00", .18)]:
        selected = [next(row for row in operations
                         if row["pattern"] == pattern and row["search_limit_seconds"] == limit)
                    for pattern in PATTERNS]
        positions = np.arange(5) + offset
        axes[0].bar(positions, [r["successful_updates"] for r in selected], .34,
                    label=f"{limit:g}-second limit", color=color)
        for x, row in zip(positions, selected):
            axes[0].text(x, row["successful_updates"] + .15,
                         f"{row['successful_updates']}/{row['attempted_updates']}",
                         ha="center", fontsize=9)
        for x, row in zip(positions, selected):
            stats = row["successful_update_seconds"]
            if stats["n"]:
                axes[1].errorbar(x, stats["median"],
                                yerr=[[stats["median"] - stats["q1"]],
                                      [stats["q3"] - stats["median"]]],
                                fmt="o", capsize=4, color=color)
            else:
                axes[1].text(x, 1.3, "no success", rotation=90, ha="center", fontsize=8)
    sizes = [next(r["size"] for r in operations if r["pattern"] == p) for p in PATTERNS]
    short_labels = ["contention", "heterogeneous", "candidates", "penalties", "replanning"]
    for ax in axes[:2]:
        ax.set_xticks(range(5), [f"{p}\nn={size}" for p, size in zip(short_labels, sizes)],
                      rotation=25, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, target_steps + 1.2)
    axes[0].axhline(target_steps, color="#444444", ls=":", lw=1)
    axes[0].set_ylabel("Completed updates")
    axes[0].set_title("Primary operations: completed / attempted")
    axes[0].legend(fontsize=9, loc="lower left")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1, 180)
    axes[1].set_ylabel("Real seconds per successful update (log)")
    axes[1].set_title("Primary within-sequence median / Q1–Q3")
    original = [row for row in confirmation["results"]
                if row["phase"] == "operation" and row["pattern"] in PATTERNS[2:4]]
    for x, row in enumerate(original):
        supplemental = row["confirmation_kind"] == "operation_lower_size"
        axes[2].bar(x, row["successful_updates"], color="#009E73" if supplemental else "#CC79A7")
        axes[2].text(x, row["successful_updates"] + .15,
                     f"{row['successful_updates']}/{row['updates']}", ha="center")
    axes[2].set_xticks(range(len(original)),
                      [f"{r['pattern'].replace('many_', '').replace('mixed_', '')}\nn={r['size']}"
                       for r in original], rotation=25, ha="right", fontsize=9)
    axes[2].set_ylim(0, target_steps + 1.2)
    axes[2].set_title("Final source b08dbe8 / 1-second limit")
    axes[2].set_ylabel("Completed updates (one sequence per condition)")
    axes[2].axhline(target_steps, color="#444444", ls=":", lw=1)
    fig.suptitle("Static success does not ensure continued updates; each search has its own limit")
    save_figure(fig, output, "operational_updates")


def formatted(stats, key="median"):
    value = stats.get(key)
    return f"{value:.3f}" if value is not None else "—"


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    trials = read_csv(source / "trial_metrics.csv")
    calls = read_csv(source / "solver_calls.csv")
    updates = read_csv(source / "update_metrics.csv")
    config = json.loads((source / "config.json").read_text())
    confirmation = json.loads((source / "confirmation.json").read_text())
    conditions = summarize_trials(trials)
    paired = pair_quality(trials)
    operations = summarize_operations(trials, updates)
    inputs = ["trial_metrics.csv", "solver_calls.csv", "update_metrics.csv", "config.json",
              "confirmation.json", "environment.json"]
    result = {
        "primary_source": json.loads((source / "environment.json").read_text())["commit"],
        "confirmation_source": confirmation["source_commit"],
        "input_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                         for name in inputs},
        "counts": {"trials": len(trials), "calls": len(calls), "updates": len(updates),
                   "confirmation_conditions": len(confirmation["results"]),
                   "outcomes": dict(Counter(row["outcome"] for row in trials)),
                   "solver_statuses": dict(Counter(row["status"] for row in calls))},
        "quantile_method": "linear", "conditions": conditions,
        "paired_static_quality": paired, "primary_operations": operations,
        "confirmation": confirmation,
    }
    (output / "statistics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    flat = []
    for row in conditions:
        item = {key: row[key] for key in
                ("phase", "pattern", "search_limit_seconds", "size", "runs", "valid", "valid_fraction")}
        for metric in TIMING + QUALITY:
            item.update({f"{metric}_{key}": value for key, value in row[metric].items()})
        flat.append(item)
    write_csv(output / "condition_statistics.csv", flat)
    write_csv(output / "paired_quality.csv", paired)
    draw_static(conditions, output)
    draw_operations(operations, confirmation, config["operation_steps"], output)

    by_phase = {phase: [row for row in trials if row["phase"] == phase]
                for phase in ("scan", "io", "operation")}
    phase_counts = {
        phase: (len(rows), sum(row["outcome"] == "VALID" for row in rows))
        for phase, rows in by_phase.items()
    }
    scan_n, scan_valid = phase_counts["scan"]
    io_n, io_valid = phase_counts["io"]
    operation_n, operation_valid = phase_counts["operation"]
    report = [
        "# Issue #23: 記述統計と図",
        "",
        f"主評価 `{result['primary_source']}` の{len(trials)}試行を再集計した資料です。"
        f"最終確認 `{result['confirmation_source']}` の{len(confirmation['results'])}条件は"
        "独立に表示し、反復へ合算していません。",
        "再探索なし。入力ファイルのSHA-256、全条件の統計は[statistics.json](statistics.json)、"
        "表は[condition_statistics.csv](condition_statistics.csv)へ保存しています。",
        "",
        "## 集計方法と分母",
        "",
        f"- 主評価: 静的{scan_n}試行（{scan_valid}成功・{scan_n - scan_valid}失敗）、"
        f"IO {io_n}試行（{io_valid}成功）、運用{operation_n}系列"
        f"（{operation_valid}成功・{operation_n - operation_valid}失敗）。"
        "今回の失敗はすべてUNKNOWN。静的条件の反復数は1または5。",
        "- 時間・メモリは失敗を含む全試行、遅延・ペナルティ・利用率は有効出力のみを集計。",
        "- 中央値、Q1、Q3、線形補間p95、最大、平均、標本標準偏差を記録。n=1の標準偏差は未定義。"
        "図の帯はQ1–Q3で、信頼区間ではありません。少数反復のp95は将来の保証ではありません。",
        "- 同一の決定的入力・seedでの反復です。施設・実験集団からの独立標本ではなく、"
        "成功率の一般化や統計的有意差の主張を行いません。",
        "- 探索上限は各CP-SAT呼出しに1/60秒を直接設定。"
        "構築・反映・検査は合否対象外。UNKNOWNは実行不能の証明ではありません。",
        "",
        "## 静的成功率",
        "",
        "セルは有効計画数／試行数。灰色は未測定です。規模と失敗の単調性は仮定しません。",
        "",
        "![静的成功率](static_success.png)",
        "",
        "### 全反復成功の最大実測規模における統計",
        "",
        "| 負荷 | 探索秒 | 規模 | 成功/反復 | 計画時間 中央値 / p95 / 最大（秒） | 構築時間 中央値（秒） | ピークRSS 中央値 / 最大（MiB） | 最大遅延 中央値（分） |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for pattern in PATTERNS:
        for limit in LIMITS:
            candidates = [r for r in conditions if r["phase"] == "scan"
                          and r["pattern"] == pattern and r["search_limit_seconds"] == limit
                          and r["valid"] == r["runs"]]
            row = max(candidates, key=lambda r: r["size"])
            time = row["total_seconds"]
            memory = row["memory_peak_mib"]
            report.append(
                f"| {pattern} | {limit:g} | {row['size']} | {row['valid']}/{row['runs']} | "
                f"{formatted(time)} / {formatted(time, 'p95')} / {formatted(time, 'max')} | "
                f"{formatted(row['build_seconds'])} | {formatted(memory)} / "
                f"{formatted(memory, 'max')} | {formatted(row['delay_max_minutes'])} |")
    report += [
        "",
        "512は測定上限です。成功した最大実測規模同士では負荷量が異なるため、"
        "この表の1秒と60秒の遅延・メモリを公平な品質比較として解釈しません。",
        "",
        "## 計算時間・品質・メモリ",
        "",
        "時間図は失敗も含みます。点は中央値、帯はQ1–Q3、横軸は実験数です。"
        "計画時間図の点線は探索設定値であり、計画全体の制限ではありません。",
        "",
        "![計算時間](static_runtime.png)",
        "",
        "最大遅延は各計画内の最大値を条件ごとに集計し、有効出力のない条件は欠測です。"
        "仮想分と計算の実秒は異なる単位です。遅延0を残すためsymlog軸を使います。"
        "RSSは起動・importを含むワーカープロセス全体のピークです。",
        "",
        "![遅延とメモリ](static_quality_memory.png)",
        "",
        "### 同一入力における品質比較",
        "",
    ]
    for metric, label in [("movable_penalty", "移動可能群のペナルティ"),
                          ("delay_max_minutes", "最大遅延")]:
        differences = [r[f"{metric}_delta_60_minus_1"] for r in paired
                       if r[f"{metric}_delta_60_minus_1"] is not None]
        report.append(
            f"- {label}: 両探索上限で有効解のある{len(differences)}条件中、"
            f"60秒の中央値が小さい{sum(v < 0 for v in differences)}、"
            f"同じ{sum(v == 0 for v in differences)}、大きい{sum(v > 0 for v in differences)}条件。")
    report += [
        "",
        "各条件の入力ハッシュ一致を確認し、異なる反復数を持つ条件もそれぞれの中央値を比較しました。"
        "試行の一対一対応や統計的検定は行いません。"
        "[paired_quality.csv](paired_quality.csv)には失敗側も欠測として残しています。"
        "最適化目的はペナルティを優先する重み付き目的値で、最大遅延そのものではありません。"
        "異なる入力・重みの目的値やペナルティはプールしません。",
        "",
        "## 継続更新",
        "",
        "各負荷・探索上限で運用系列は1回のみ。系列内の更新は独立な反復ではありません。"
        "1更新に通常2回の探索があり、実験追加も別の呼出しを行います。"
        "「成功／試行」は未実施更新を成功にも失敗にも数えていません。",
        "",
        "![継続更新](operational_updates.png)",
        "",
        "| 負荷 | 規模 | 探索秒 | 成功/試行更新 | 成功更新時間 中央値 / Q1–Q3 / p95 / 最大（秒） | 系列全体（秒） |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in operations:
        time = row["successful_update_seconds"]
        report.append(
            f"| {row['pattern']} | {row['size']} | {row['search_limit_seconds']:g} | "
            f"{row['successful_updates']}/{row['attempted_updates']} | {formatted(time)} / "
            f"{formatted(time, 'q1')}–{formatted(time, 'q3')} / {formatted(time, 'p95')} / "
            f"{formatted(time, 'max')} | {row['total_seconds']:.3f} |")
    report += [
        "",
        "60秒の運用規模は上表の256/512/128/128/256で、全負荷512ではありません。"
        "最終ソースの1秒・128実験では割当候補多数と複雑ペナルティのUNKNOWNを再現し、"
        "別の補助64実験で両負荷8更新成功（各1系列）でした。信頼できる運用上限は未確定です。",
        "",
        "## 再現",
        "",
        "保存用コミットのリポジトリルートで実行します。スケジューラーは呼び出しません。",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg poetry run python \\",
        "  scripts/analyze_scheduler_capacity.py \\",
        "  docs/benchmarks/scheduler_capacity \\",
        "  --output /tmp/gems-capacity-statistics",
        "```",
        "",
        "PNGとSVGを併記しています。主評価の入力・出力・失敗・全ログは"
        "[元資料](../README.md)の圧縮アーカイブとSHA-256で履歴に保持しています。",
        "",
    ]
    (output / "README.md").write_text("\n".join(report))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.directory, args.output)
