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
    best_case_gaps = [
        min(sa["timing_penalty"] for sa in results if is_feasible(sa))
        - cp["timing_penalty"]
        for cp, results in case_results
        if is_feasible(cp) and any(is_feasible(sa) for sa in results)
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
        "cases_with_feasible_sa": len(best_case_gaps),
        "case_cp_better": sum(gap > 0 for gap in best_case_gaps),
        "case_equal_score": sum(gap == 0 for gap in best_case_gaps),
        "case_sa_better": sum(gap < 0 for gap in best_case_gaps),
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
        "| Condition | Cases | CP proven optimal | SA feasible/all runs | CP lower/equal/SA lower | Median score gap | 90th percentile |",
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
    clustered = aggregate([
        pair for pair in case_results if pair[0]["density"] == "clustered"
    ])
    spread = aggregate([
        pair for pair in case_results if pair[0]["density"] == "spread"
    ])
    multi_one = aggregate([
        pair for pair in case_results
        if pair[0]["variant"] == "multi_interval" and pair[0]["machine_count"] == 1
    ])
    multi_two = aggregate([
        pair for pair in case_results
        if pair[0]["variant"] == "multi_interval" and pair[0]["machine_count"] == 2
    ])
    lines = [
        "# Scheduler-only pattern study",
        "",
        "This document reports results aggregated by `scripts/study_scheduler_patterns.py`.",
        "The previous method is simulated annealing `s` through the public API; the new method is OR-Tools `o`.",
        "Only scheduling was run; the simulator was not called.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/gems-mpl poetry run python scripts/study_scheduler_patterns.py \\",
        f"  --output-dir /tmp/gems-pattern-study-recheck --seeds {' '.join(map(str, seeds))}",
        "```",
        "",
        "The script generates CSV/JSON, this summary and figures in a new output directory.",
        "To reaggregate only, using saved per-run JSON:",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/gems-mpl poetry run python scripts/study_scheduler_patterns.py \\",
        "  --from-runs /tmp/gems-pattern-study-recheck/runs.json \\",
        "  --output-dir /tmp/gems-pattern-study-reanalysis",
        "```",
        "",
        "[Score scatter plot for feasible pairs](paired_scores.png) and "
        "[feasibility and runtime figure](feasibility_runtime.png).",
        "",
        "## Execution conditions",
        "",
        f"- Python {sys.version.split()[0]}, OR-Tools {ortools.__version__}, "
        f"simanneal {simanneal.__version__}, Matplotlib {matplotlib.__version__}.",
        f"- Execution environment: {platform.platform()}.",
        f"- {result['cases']} cases. Simulated annealing seeds: {seeds}.",
        f"- CP-SAT: {SOLVER_TIME_LIMIT_SECONDS}-second limit, one worker, seed 0."
        " Simulated annealing: 100,000 steps, Tmax 25000, Tmin 1 (existing implementation).",
        "- Inputs are rebuilt for each method and checked for identical SHA-256 hashes before scheduling.",
        "- The simple-task variant has 4/8 tasks and one machine. Group variants have 3/6 groups, "
        "each containing two tasks. The multi-machine variant has 1/2 machines of the same type.",
        "- Optimal start times begin at 12 minutes with fixed spacing. Clustered/spread spacing is 2/8 minutes for simple tasks, "
        "3/14 minutes for single-machine groups, and 2/8 minutes for multi-machine groups. "
        "Processing-time and preceding-interval generation formulas are fixed in the script.",
        "- Penalty patterns use Linear, Range or Rest+Linear alone, or "
        "Mixed, cycling through all five types. Coefficients and rest cycles are fixed in "
        "`make_penalty()` in `scripts/compare_schedulers.py`.",
        "- These pattern counts, intervals and seeds are synthetic conditions selected for comparison; "
        "they are not estimated from operational experiment data.",
        "",
        "## Metrics",
        "",
        "- Scores sum the existing `calculate_penalty()` for each task/group. "
        "Lower is better. Score comparisons include only runs satisfying resource, assignment and preceding-interval constraints.",
        "- `CP lower/equal/SA lower` counts feasible pairs for the same case and seed. "
        "The score gap is `SA − CP`; a positive value means CP is lower. "
        "The 90th percentile uses the nearest-rank value.",
        "- CP `OPTIMAL` and `FEASIBLE` are distinguished. "
        "Runtime is wall-clock time for the scheduling call.",
        "",
        "## Results",
        "",
        f"- CP: {result['cp_optimal']} proven optimal, "
        f"{result['cp_feasible']} feasible within the time limit, "
        f"{result['cp_invalid']} infeasible or failed.",
        f"- Simulated annealing: {result['sa_feasible']}/{result['sa_runs']} runs "
        "feasible at the individual machine-ID level.",
        f"- Among {len([1 for cp, rs in case_results for sa in rs if is_feasible(cp) and is_feasible(sa)])} feasible pairs, "
        f"CP lower {result['cp_better']}, equal {result['equal_score']}, "
        f"SA lower {result['sa_better']}.",
        f"- Per case, across {len(seeds)} simulated annealing seeds, "
        "compared with the best feasible result, "
        f"among {result['cases_with_feasible_sa']} cases: CP lower "
        f"{result['case_cp_better']}, equal {result['case_equal_score']}, "
        f"SA lower {result['case_sa_better']}. The remaining "
        f"{result['cases'] - result['cases_with_feasible_sa']} cases "
        "have no feasible simulated annealing solution.",
        f"- Median score gap {format_number(result['median_gap'])}, "
        f"90th percentile {format_number(result['p90_gap'])}.",
        f"- Median wall-clock time: CP {result['median_cp_seconds']:.6f} seconds, "
        f"simulated annealing {result['median_sa_seconds']:.6f} seconds.",
        "",
        summary_table(case_results, "variant", "By variant"),
        "",
        summary_table(case_results, "penalty_pattern", "By penalty pattern"),
        "",
        summary_table(case_results, "density", "By clustering of preferred start times"),
        "",
        summary_table(case_results, "count", "By task/group count"),
        "",
        summary_table(
            [pair for pair in case_results if pair[0]["variant"] == "multi_interval"],
            "machine_count", "By machine count",
        ),
        "",
        "## Observations by condition",
        "",
        f"- Clustered conditions: among {clustered['cp_better'] + clustered['equal_score'] + clustered['sa_better']} feasible pairs, "
        f"CP is lower in {clustered['cp_better']}; spread conditions: among "
        f"{spread['cp_better'] + spread['equal_score'] + spread['sa_better']} feasible pairs, "
        f"CP is lower in {spread['cp_better']}.",
        f"- Multi-machine simulated annealing feasibility is "
        f"{multi_one['sa_feasible']}/{multi_one['sa_runs']} with one machine of the type, and "
        f"{multi_two['sa_feasible']}/{multi_two['sa_runs']} with two machines.",
        "- The overall 90th-percentile score gap is affected by "
        "the 1,000,000-point "
        "cyclical rest penalty applied when a start time falls within the rest range. See the penalty-pattern table above "
        "for separate linear and range results.",
        "",
        "## Interpretation and limitations",
        "",
        "Low simulated annealing scores from schedules with overlapping individual machine IDs "
        "do not count as feasible wins. Where CP-SAT cannot "
        "prove optimality within the time limit, only the score of the obtained solution is compared.",
        "Results use synthetic inputs and few random seeds in one environment; "
        "they do not guarantee general speed or solution quality, or predict actual cell culture.",
        "",
        "When rerunning, save CSV/JSON in a new output directory and "
        "compare generated results with this summary and figures. Without adding raw data to Git, "
        "the generation script, input SHA-256 hashes, conditions and aggregation formulas remain traceable.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def load_case_results(path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of scheduler runs")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    case_results = []
    for case_id, group in grouped.items():
        cp = [row for row in group if row["method"] == "o"]
        sa = [row for row in group if row["method"] == "s"]
        if len(cp) != 1 or not sa or len({row["seed"] for row in sa}) != len(sa):
            raise ValueError(f"Invalid method/seed records for {case_id}")
        if len({row["input_sha256"] for row in group}) != 1:
            raise ValueError(f"Input mismatch for {case_id}")
        case_results.append((cp[0], sorted(sa, key=lambda row: row["seed"])))
    seeds = sorted({row["seed"] for row in rows if row["method"] == "s"})
    return case_results, seeds


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
    parser.add_argument(
        "--from-runs", type=Path, default=None,
        help="Regenerate summary and plots from a prior runs.json without scheduling",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_runs is not None:
        case_results, seeds = load_case_results(args.from_runs)
        write_summary(args.output_dir, case_results, seeds)
        make_plots(args.output_dir, case_results)
        print(f"Reanalyzed {len(case_results)} cases from {args.from_runs}")
        return
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    cases = all_cases(args.variants)
    if args.case_limit is not None:
        if args.case_limit < 1:
            parser.error("--case-limit must be positive")
        cases = cases[:args.case_limit]
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
