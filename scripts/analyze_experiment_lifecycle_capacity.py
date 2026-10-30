"""Summarize verified Issue #24 records without scheduling or changing raw data."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


KEYS = ("series", "pattern", "budget", "mode", "parents", "children",
        "group_interval", "generations")
METRICS = ("planning_seconds", "preparation_seconds", "fixture_seconds",
           "supplemental_initialization_seconds",
           "save_seconds", "total_pipeline_seconds", "measured_phase_seconds",
           "peak_rss_kib", "input_json_bytes", "own_history_rows",
           "inherited_history_rows", "archived_history_rows", "timing_penalty",
           "delay_median_minutes", "delay_p95_minutes", "delay_max_minutes")


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple("scan_and_boundary" if k == "series" and row[k] in ("scan", "boundary")
                    else row[k] for k in KEYS)
        groups[key].append(row)
    output = []
    for key, group in sorted(groups.items()):
        entry = dict(zip(KEYS, key))
        entry.update(repetitions=len(group),
                     on_time=sum(r["outcome"] == "VALID_ON_TIME" for r in group),
                     late=sum(r["outcome"] == "VALID_LATE" for r in group),
                     outcomes=json.dumps(dict(Counter(r["outcome"] for r in group)), sort_keys=True))
        phases = sorted({r.get("storage_phase", "unspecified") for r in group})
        entry["storage_phases"] = ";".join(phases)
        for metric in METRICS:
            if metric in ("save_seconds", "total_pipeline_seconds", "measured_phase_seconds") \
                    and len(phases) > 1:
                # Keep I/O from tmpfs and the workspace filesystem separate.
                entry[f"{metric}_by_storage_phase"] = json.dumps({
                    phase: {name: function(values) for name, function in (
                        ("min", min), ("median", statistics.median), ("max", max))}
                    for phase in phases
                    if (values := [r[metric] for r in group if r.get(metric) is not None and
                                   r.get("storage_phase", "unspecified") == phase])
                }, sort_keys=True)
                continue
            values = [r[metric] for r in group if r.get(metric) is not None]
            if values:
                for name, value in (("min", min(values)), ("median", statistics.median(values)),
                                    ("max", max(values))):
                    entry[f"{metric}_{name}"] = value
        known_dimensions = [r["dimensions"] for r in group if "dimensions" in r]
        dimensions = known_dimensions[0] if known_dimensions else {}
        assert all(value == dimensions for value in known_dimensions)
        entry.update(dimensions)
        output.append(entry)
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(source, output, stats, verified, metadata, profiles):
    boundaries = json.loads((source / "boundaries.json").read_text())
    lines = [
        "# Issue #24: 分割後の計画全体時間",
        "",
        f"計測対象commit `{metadata['commit']}`。{verified['trial_count']}試行を記録し、"
        f"{verified.get('verified_pairs', 0)}組の物理条件を照合した。",
        f"全条件の計測完了: `{verified['fully_measured']}`。期限等によるスキップ: {verified['skipped']}。",
        f"判定内訳: `{json.dumps(verified['outcomes'], sort_keys=True)}`。",
        "",
        "計画時間は管理層の入口から検査済み反映まで。CP-SAT探索上限を1秒/60秒に設定し、"
        "全体時間もそれぞれ1秒/60秒以内である場合だけ時間内成功とする。再試行は無効。",
        "有効な時間超過は解なし・実行不能と区別する。最適化が探索上限まで続いた場合、"
        "有効な解を返しても構築・検査等を含む全体時間は上限を超え得る。",
        "",
        "## 基本条件の反復確認",
        "",
        "1親→2子・1世代。以下の規模は3反復が全て時間内だった最大の確認点であり、"
        "処理能力の限界や施設での保証値ではない。上限付近の揺れも判定に含む。",
        "",
        "| 負荷 | 全体時間条件 | 確認済み親数 | 活動中の子数 | 次の反復確認点（親数、時間内/回数） |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for boundary in boundaries:
        parents = boundary["maximum_confirmed_on_time_parents"]
        later = sorted((int(n), values) for n, values in boundary["tested"].items()
                       if len(values) >= 3 and (parents is None or int(n) > parents))
        following = (f"{later[0][0]}, {later[0][1].count('VALID_ON_TIME')}/{len(later[0][1])}"
                     if later else "測定上限／未確認")
        lines.append(f"| {boundary['pattern']} | {boundary['budget']:g}秒 | "
                     f"{parents if parents is not None else '未確認'} | "
                     f"{parents*2 if parents is not None else '—'} | {following} |")
    lines += [
        "",
        "## 8親・2子・1世代の追加費用",
        "",
        "秒数は中央値。反復数・最小/最大、遅延・ペナルティ・flat比較はCSVに記録する。",
        "初期化は別プロセスの補足計測で、元試行の全工程時間には加算しない。"
        f"入力ハッシュが一致した補足計測は{profiles}件。",
        "",
        "| 負荷 | 条件 | 計画全体（時間内/回数） | 分割要求・確定 | 補足初期化 | 保存（保存先別） | ピークRSS MiB |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in stats:
        if not (row["series"] == "scan_and_boundary" and row["mode"] == "lifecycle"
                and row["parents"] == 8):
            continue
        def median(name, divisor=1):
            value = row.get(f"{name}_median")
            return "—" if value is None else f"{value/divisor:.4g}"
        save = median("save_seconds")
        if "save_seconds_by_storage_phase" in row:
            save = "; ".join(f"{phase}: {value['median']:.4g}" for phase, value in
                            json.loads(row["save_seconds_by_storage_phase"]).items())
        else:
            save = f"{row['storage_phases']}: {save}"
        planning = row.get("planning_seconds_median")
        planning_text = "—" if planning is None else f"{planning:.6g}"
        lines.append(f"| {row['pattern']} | {row['budget']:g}秒 | "
                     f"{planning_text} ({row['on_time']}/{row['repetitions']}) | "
                     f"{median('preparation_seconds')} | {median('supplemental_initialization_seconds')} | "
                     f"{save} | {median('peak_rss_kib', 1024)} |")
    lines += [
        "",
        "## 資料と範囲",
        "",
        "- [各試行CSV](trials.csv)、[条件別CSV](conditions.csv)、[集計JSON](summary.json)。",
        "- [計画時間の図](planning.png)。線は中央値、反復点のバーは最小/最大。"
        "塗り丸は全反復時間内、白丸は有効な時間超過／時間判定混在、×は失敗を含む。",
        "- 基本走査は親1/2/4/8/16/32/64/128。8親で子4/8、群間隔1/4、2世代を別条件として比較。",
        "- 群間隔は分割前の合成した完了観測量の比較で、実機での分割頻度ではない。",
        "- 初期tmpfsと再開後のワークスペースの保存・全工程時間は混ぜて集計しない。",
        "- RSSはfixture生成・flat化を含むプロセスピークで、履歴だけの常駐量ではない。",
        "- 決定的な入力・seed 0・solver 1 worker・Pandasでの記述統計。"
        "成功確率や任意の実験手順・設備・継続更新を保証しない。",
        "- 全工程タイマーは当該試行の初期化、入力/出力化、独立検査、保存も含む。"
        "前処理＋計画＋保存の合計とは異なる。",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def analyze(source, output):
    verified = json.loads((source / "verification.json").read_text())
    rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines()]
    assert len(rows) == verified["trial_count"]
    profiles = 0
    for row in rows:
        profile = source / row["artifact_directory"] / "initialization_profile.json"
        if profile.exists():
            measurement = json.loads(profile.read_text())
            assert measurement["input_hash"] == row["input_hash"]
            row["supplemental_initialization_seconds"] = measurement["initialization_seconds"]
            profiles += 1
    stats = summarize(rows)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "conditions.csv", stats)
    trials = []
    for row in rows:
        trial = {k: row[k] for k in (*KEYS, "repetition", "outcome", "artifact_directory")}
        trial.update({k: row.get(k) for k in METRICS})
        trial["storage_phase"] = row.get("storage_phase", "unspecified")
        trial.update(row.get("dimensions", {}))
        trial["solver_status"] = ";".join(r["status"] for r in row.get("reports", []))
        trials.append(trial)
    write_csv(output / "trials.csv", trials)
    metadata = json.loads((source / "metadata.json").read_text())
    (output / "summary.json").write_text(json.dumps(dict(
        source_commit=metadata["commit"], verification=verified,
        timing=metadata["timing"], conditions=stats, initialization_profiles=profiles,
        initialization_note="Supplemental cold-process measurements; not added to original pipeline timers",
        storage_note="I/O and pipeline metrics from different storage phases are not pooled"), indent=2))
    colors = dict(zip(sorted({r["pattern"] for r in rows}), plt.get_cmap("tab10").colors))
    budgets = sorted({r["budget"] for r in rows})
    fig, axes = plt.subplots(1, len(budgets), figsize=(13, 5.5), squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        for pattern, color in colors.items():
            for mode, style in (("lifecycle", "-"), ("flat", "--")):
                points = sorted((r for r in stats if r["series"] == "scan_and_boundary" and
                                 r["pattern"] == pattern and r["budget"] == budget and
                                 r["mode"] == mode and "planning_seconds_median" in r),
                                key=lambda r: r["active_experiments"])
                ax.plot([r["active_experiments"] for r in points],
                        [r["planning_seconds_median"] for r in points],
                        color=color, linestyle=style, linewidth=1)
                for point in points:
                    success = point["on_time"] == point["repetitions"]
                    failed = point["on_time"] + point["late"] < point["repetitions"]
                    ax.plot(point["active_experiments"], point["planning_seconds_median"],
                            color=color, marker="x" if failed else "o", markersize=4,
                            markerfacecolor=color if success else "none")
                    if point["repetitions"] > 1:
                        median = point["planning_seconds_median"]
                        ax.errorbar(point["active_experiments"], median,
                                    yerr=[[median-point["planning_seconds_min"]],
                                          [point["planning_seconds_max"]-median]],
                                    color=color, capsize=2, linewidth=.8)
        ax.axhline(budget, color="black", linestyle=":", linewidth=1)
        ax.set(xscale="log", yscale="log", xlabel="Active child experiments",
               ylabel="Total planning seconds (median)", title=f"{budget:g}s total-time condition")
        ax.grid(True, alpha=.2)
    handles = [Line2D([0], [0], color=color, label=pattern) for pattern, color in colors.items()]
    handles += [Line2D([0], [0], color="black", linestyle=style, label=mode)
                for mode, style in (("lifecycle", "-"), ("flat", "--"))]
    handles += [Line2D([0], [0], color="black", linestyle="none", marker=marker,
                       markerfacecolor=face, label=label)
                for marker, face, label in (
                    ("o", "black", "all on time"),
                    ("o", "none", "valid, late / mixed timing"),
                    ("x", "none", "failure / mixed outcome"))]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .84))
    with matplotlib.rc_context({"svg.hashsalt": "gems-issue24-lifecycle-capacity"}):
        for suffix in ("png", "svg"):
            options = {"metadata": {"Date": None}} if suffix == "svg" else {}
            fig.savefig(output / f"planning.{suffix}", dpi=160, **options)
    plt.close(fig)
    write_report(source, output, stats, verified, metadata, profiles)
    print(json.dumps(dict(trials=len(rows), conditions=len(stats), source_commit=metadata["commit"])))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.directory, args.output)
