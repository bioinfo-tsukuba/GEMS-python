"""Paired, scheduler-only study across task load, penalties, and machine counts."""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ortools
import simanneal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task as MultiTask, TaskGroup as MultiGroup,
)
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    Experiments as MultiExperiments,
)
from gems_python.one_machine_problem_interval_task.task_info import (
    Task as SingleTask, TaskGroup as SingleGroup,
)
from gems_python.one_machine_problem_interval_task.transition_manager import (
    Experiments as SingleExperiments,
)
from gems_python.one_machine_problem_simple_task.task_info import OneMachineTask
from gems_python.one_machine_problem_simple_task.transition_manager import (
    Experiments as SimpleExperiments,
)
from scripts.compare_schedulers import (
    METHOD_COLOR, SOLVER_TIME_LIMIT_SECONDS, VARIANTS, make_penalty, run_manager,
)


DENSITIES = ("clustered", "spread")
PENALTY_PATTERNS = ("linear", "range", "rest_linear", "mixed")
MIXED_ORDER = (
    "CyclicalRestPenaltyWithLinear", "NonePenalty", "LinearPenalty",
    "LinearWithRangePenalty", "CyclicalRestPenalty",
)
PATTERN_TYPE = {
    "linear": "LinearPenalty",
    "range": "LinearWithRangePenalty",
    "rest_linear": "CyclicalRestPenaltyWithLinear",
}
COUNTS = {"simple": (4, 8), "single_interval": (3, 6), "multi_interval": (3, 6)}


def all_cases(variants):
    cases = []
    for variant in variants:
        for count in COUNTS[variant]:
            for density in DENSITIES:
                for penalty_pattern in PENALTY_PATTERNS:
                    machine_counts = (1, 2) if variant == "multi_interval" else (1,)
                    for machine_count in machine_counts:
                        cases.append({
                            "case_id": f"{variant}_{count}_{density}_{penalty_pattern}_m{machine_count}",
                            "variant": variant,
                            "count": count,
                            "density": density,
                            "penalty_pattern": penalty_pattern,
                            "machine_count": machine_count,
                        })
    return cases


def penalty_for(case, index):
    pattern = case["penalty_pattern"]
    kind = (
        MIXED_ORDER[index % len(MIXED_ORDER)] if pattern == "mixed"
        else PATTERN_TYPE[pattern]
    )
    return make_penalty(case["variant"], kind)


def make_case(case, directory):
    variant = case["variant"]
    clustered = case["density"] == "clustered"
    if variant == "simple":
        stride = 2 if clustered else 8
        durations = (6, 4, 7, 5, 3)
        manager = SimpleExperiments([], directory)
        manager.tasks = [
            OneMachineTask(
                12 + index * stride, durations[index % len(durations)],
                penalty_for(case, index), f"task{index}", "pattern", "fixed",
            )
            for index in range(case["count"])
        ]
        return manager
    if variant == "single_interval":
        stride = 3 if clustered else 14
        groups = [
            SingleGroup(
                12 + index * stride,
                penalty_for(case, index),
                [
                    SingleTask(3 + index % 3, f"g{index}a", interval=1 + index % 2),
                    SingleTask(3 + index % 2, f"g{index}b", interval=2 + index % 2),
                ],
            )
            for index in range(case["count"])
        ]
        return SingleExperiments([], directory, task_groups=groups)
    stride = 2 if clustered else 8
    groups = [
        MultiGroup(
            12 + index * stride,
            penalty_for(case, index),
            [
                MultiTask(4 + index % 2, f"g{index}a", 0, interval=2),
                MultiTask(3 + index % 3, f"g{index}b", 0, interval=2),
            ],
        )
        for index in range(case["count"])
    ]
    machines = MachineList([
        Machine(0, f"m{index}") for index in range(case["machine_count"])
    ])
    return MultiExperiments(
        [], directory, machine_list=machines, task_groups=groups,
    )


def input_fingerprint(manager, variant):
    if variant == "simple":
        specification = [
            {
                "optimal": task.optimal_time,
                "duration": task.processing_time,
                "penalty": type(task.penalty_type).__name__,
                "parameters": task.penalty_type.to_dict(),
            }
            for task in manager.tasks
        ]
    else:
        specification = {
            "machines": (
                [(machine.machine_id, machine.machine_type)
                 for machine in manager.machine_list.machines]
                if variant == "multi_interval" else ["m0"]
            ),
            "groups": [
                {
                    "optimal": group.optimal_start_time,
                    "penalty": type(group.penalty_type).__name__,
                    "parameters": group.penalty_type.to_dict(),
                    "tasks": [
                        (
                            task.processing_time, task.interval,
                            task.optimal_machine_type
                            if variant == "multi_interval" else 0,
                        )
                        for task in group.tasks
                    ],
                }
                for group in manager.task_groups
            ],
        }
    payload = json.dumps(specification, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def run_case(case, method, seed):
    with tempfile.TemporaryDirectory(prefix="gems-pattern-") as directory:
        manager = make_case(case, Path(directory))
        fingerprint = input_fingerprint(manager, case["variant"])
        return {
            **case,
            "input_sha256": fingerprint,
            **run_manager(manager, case["variant"], method, seed),
        }


def is_feasible(row):
    return (
        row["status"] in ("OPTIMAL", "FEASIBLE", "FINISHED")
        and row.get("overlap_minutes") == 0
        and row.get("assignment_errors") == 0
        and row.get("interval_errors") == 0
    )


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent * len(ordered)) - 1)]


def aggregate(case_results):
    cp_rows = [cp for cp, _ in case_results]
    anneal_rows = [sa for _, results in case_results for sa in results]
    matched = [
        (cp, sa) for cp, results in case_results for sa in results
        if is_feasible(cp) and is_feasible(sa)
    ]
    gaps = [
        sa["timing_penalty"] - cp["timing_penalty"] for cp, sa in matched
    ]
    return {
        "cases": len(case_results),
        "cp_optimal": sum(row["status"] == "OPTIMAL" for row in cp_rows),
        "cp_feasible": sum(row["status"] == "FEASIBLE" for row in cp_rows),
        "cp_invalid": sum(not is_feasible(row) for row in cp_rows),
        "sa_runs": len(anneal_rows),
        "sa_feasible": sum(is_feasible(row) for row in anneal_rows),
        "cp_better": sum(gap > 0 for gap in gaps),
        "equal_score": sum(gap == 0 for gap in gaps),
        "sa_better": sum(gap < 0 for gap in gaps),
        "median_gap": statistics.median(gaps) if gaps else None,
        "p90_gap": percentile(gaps, 0.9),
        "median_cp_seconds": (
            statistics.median(row["seconds"] for row in cp_rows) if cp_rows else None
        ),
        "median_sa_seconds": (
            statistics.median(row["seconds"] for row in anneal_rows)
            if anneal_rows else None
        ),
    }


def format_number(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def summary_table(case_results, field, heading):
    groups = defaultdict(list)
    for pair in case_results:
        groups[pair[0][field]].append(pair)
    lines = [
        f"### {heading}",
        "",
        "| 条件 | ケース | CP最適証明 | SA実行可能/全回 | CP低点/同点/SA低点 | スコア差中央値 | 90%点 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, pairs in sorted(groups.items(), key=lambda item: str(item[0])):
        result = aggregate(pairs)
        lines.append(
            f"| {label} | {result['cases']} | {result['cp_optimal']} | "
            f"{result['sa_feasible']}/{result['sa_runs']} | "
            f"{result['cp_better']}/{result['equal_score']}/{result['sa_better']} | "
            f"{format_number(result['median_gap'])} | "
            f"{format_number(result['p90_gap'])} |"
        )
    return "\n".join(lines)


def write_summary(output_dir, case_results, seeds):
    result = aggregate(case_results)
    lines = [
        "# Scheduler-only pattern study",
        "",
        "この文書は `scripts/study_scheduler_patterns.py` が集計した結果です。",
        "旧方式は公開APIの焼きなまし `s`、新方式は OR-Tools `o`。",
        "スケジューリングのみを実行し、シミュレーターは呼んでいません。",
        "",
        "## 実行条件",
        "",
        f"- Python {sys.version.split()[0]}, OR-Tools {ortools.__version__}, "
        f"simanneal {simanneal.__version__}, Matplotlib {matplotlib.__version__}。",
        f"- 実行環境: {platform.platform()}。",
        f"- ケース数 {result['cases']}。焼きなましシード: {seeds}。",
        f"- CP-SAT: {SOLVER_TIME_LIMIT_SECONDS}秒制限、1ワーカー、シード0。"
        " 焼きなまし: 10万ステップ、Tmax 25000、Tmin 1（既存実装）。",
        "- 各方式の入力は毎回作り直し、スケジュール前のSHA-256が同じことを検査。",
        "- 単一タスク型は4/8タスク、1機械。グループ型は3/6グループで、"
        "各グループに2タスク。複数機械型は同タイプの機械を1/2台。",
        "- 最適開始時刻は12分から一定間隔。単一タスク型の集中/分散は2/8分間隔、"
        "単一機械グループ型は3/14分間隔、複数機械グループ型は2/8分間隔。"
        "処理時間と先行間隔の生成式はスクリプトに固定。",
        "- ペナルティ構成は Linear、Range、Rest+Linear の各単独型と、"
        "全5型を順番に使う Mixed。係数・休止周期は "
        "`scripts/compare_schedulers.py` の `make_penalty()` に固定。",
        "- 以上のパターン数・間隔・シードは比較のために選んだ合成条件であり、"
        "実験運用データから推定した値ではない。",
        "",
        "## 指標",
        "",
        "- スコアは各タスク/グループの既存 `calculate_penalty()` の合計。"
        "低いほどよい。資源制約と割当・先行間隔を満たす実行だけをスコア比較する。",
        "- `CP低点/同点/SA低点` は同一ケース・シードの実行可能な対について数える。"
        "スコア差は `SA − CP` で、正ならCPが低い。"
        "90%点は最も近い順位の値。",
        "- CPの `OPTIMAL` と `FEASIBLE` を区別する。"
        "実行時間はスケジューリング呼出しの壁時計時間。",
        "",
        "## 結果",
        "",
        f"- CP: {result['cp_optimal']}件で最適証明、"
        f"{result['cp_feasible']}件で時間内の実行可能解、"
        f"{result['cp_invalid']}件は実行不能または失敗。",
        f"- 焼きなまし: {result['sa_feasible']}/{result['sa_runs']}実行が"
        "実機ID単位で実行可能。",
        f"- 実行可能な{len([1 for cp, rs in case_results for sa in rs if is_feasible(cp) and is_feasible(sa)])}対の"
        f"うち、CP低点 {result['cp_better']}、同点 {result['equal_score']}、"
        f"SA低点 {result['sa_better']}。",
        f"- スコア差中央値 {format_number(result['median_gap'])}、"
        f"90%点 {format_number(result['p90_gap'])}。",
        f"- 壁時計時間中央値: CP {format_number(result['median_cp_seconds'])}秒、"
        f"焼きなまし {format_number(result['median_sa_seconds'])}秒。",
        "",
        summary_table(case_results, "variant", "系統別"),
        "",
        summary_table(case_results, "penalty_pattern", "ペナルティ構成別"),
        "",
        summary_table(case_results, "density", "希望開始時刻の集中度別"),
        "",
        summary_table(case_results, "count", "タスク・グループ数別"),
        "",
        "複数機械型だけの機械台数別:",
        "",
        summary_table(
            [pair for pair in case_results if pair[0]["variant"] == "multi_interval"],
            "machine_count", "機械台数別",
        ),
        "",
        "## 読み方と制約",
        "",
        "焼きなましで実機IDが重複するスケジュールの低いスコアは、"
        "実行可能な解として勝敗へ加えない。CP-SATが時間制限内に"
        "最適証明できなかったケースは、得た解のスコア比較だけに使う。",
        "合成入力と少数の乱数シードによる1環境の結果であり、"
        "一般的な速度・解品質の保証や実細胞培養の予測ではない。",
        "",
        "再実行時にはCSV/JSONを新しい出力ディレクトリへ保存し、"
        "この要約と図を生成結果と照合する。生データをGitに追加しなくても、"
        "生成スクリプト、入力SHA-256、条件、集計式を追跡できる。",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(output_dir, case_results):
    feasible_pairs = [
        (cp, sa) for cp, rows in case_results for sa in rows
        if is_feasible(cp) and is_feasible(sa)
    ]
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    colors = {"simple": "#2878b5", "single_interval": "#4b9a44", "multi_interval": "#d2691e"}
    for variant in VARIANTS:
        pairs = [(cp, sa) for cp, sa in feasible_pairs if cp["variant"] == variant]
        if pairs:
            ax.scatter(
                [cp["timing_penalty"] for cp, _ in pairs],
                [sa["timing_penalty"] for _, sa in pairs],
                label=variant.replace("_", " "), color=colors[variant],
                alpha=0.65, s=35,
            )
    bound = max(
        [1] + [
            max(cp["timing_penalty"], sa["timing_penalty"])
            for cp, sa in feasible_pairs
        ]
    )
    ax.plot([0, bound], [0, bound], color="black", linestyle="--", linewidth=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("OR-Tools timing penalty")
    ax.set_ylabel("Annealing timing penalty")
    ax.set_title("Feasible paired scores (dashed: equal)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "paired_scores.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for index, variant in enumerate(VARIANTS):
        cases = [pair for pair in case_results if pair[0]["variant"] == variant]
        runs = [sa for _, results in cases for sa in results]
        fraction = (
            sum(is_feasible(sa) for sa in runs) / len(runs) if runs else 0
        )
        axes[0].bar(index, fraction * 100, color=colors[variant])
    axes[0].set_xticks(range(len(VARIANTS)), [x.replace("_", " ") for x in VARIANTS])
    axes[0].set_ylabel("Feasible annealing runs (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.2)
    for index, variant in enumerate(VARIANTS):
        cases = [pair for pair in case_results if pair[0]["variant"] == variant]
        cp_times = [cp["seconds"] for cp, _ in cases]
        sa_times = [sa["seconds"] for _, results in cases for sa in results]
        axes[1].scatter(
            [index - 0.12] * len(cp_times), cp_times, s=22,
            color=METHOD_COLOR["o"], alpha=0.65,
        )
        axes[1].scatter(
            [index + 0.12] * len(sa_times), sa_times, s=22,
            color=METHOD_COLOR["s"], alpha=0.35,
        )
    axes[1].set_xticks(range(len(VARIANTS)), [x.replace("_", " ") for x in VARIANTS])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Wall time per call (seconds, log)")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", label="OR-Tools",
                       markerfacecolor=METHOD_COLOR["o"]),
            plt.Line2D([0], [0], marker="o", color="w", label="Annealing",
                       markerfacecolor=METHOD_COLOR["s"]),
        ]
    )
    fig.savefig(output_dir / "feasibility_runtime.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=VARIANTS)
    parser.add_argument("--case-limit", type=int, default=None)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    cases = all_cases(args.variants)
    if args.case_limit is not None:
        if args.case_limit < 1:
            parser.error("--case-limit must be positive")
        cases = cases[:args.case_limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_results = []
    rows = []
    for index, case in enumerate(cases, start=1):
        cp = run_case(case, "o", None)
        anneal = [run_case(case, "s", seed) for seed in args.seeds]
        if any(row["input_sha256"] != cp["input_sha256"] for row in anneal):
            raise RuntimeError(f"Input mismatch in {case['case_id']}")
        case_results.append((cp, anneal))
        rows.extend((cp, *anneal))
        print(
            f"{index}/{len(cases)} {case['case_id']}: "
            f"CP={cp['status']}:{cp.get('timing_penalty')} "
            f"SA={[(row['seed'], row['status'], row.get('timing_penalty'), row.get('overlap_minutes')) for row in anneal]}",
            flush=True,
        )

    with (args.output_dir / "runs.csv").open("w", newline="") as file:
        columns = (
            "case_id", "input_sha256", "variant", "count", "density",
            "penalty_pattern", "machine_count", "method", "seed", "status",
            "timing_penalty", "overlap_minutes", "assignment_errors",
            "interval_errors", "seconds",
        )
        writer = csv.DictWriter(
            file, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "runs.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    write_summary(args.output_dir, case_results, args.seeds)
    make_plots(args.output_dir, case_results)
    print(f"Wrote {len(rows)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
