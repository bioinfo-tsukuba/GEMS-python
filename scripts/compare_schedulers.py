"""Reproducible comparison of the public OR-Tools, annealing and greedy schedulers."""

import argparse
import contextlib
import csv
import io
import json
import logging
import random
import re
import sys
import tempfile
import time
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ortools
import simanneal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems_python.common.cp_sat_scheduler import LOGGER, SOLVER_TIME_LIMIT_SECONDS
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


VARIANTS = ("simple", "single_interval", "multi_interval")
PENALTIES = (
    "NonePenalty", "LinearPenalty", "LinearWithRangePenalty",
    "CyclicalRestPenalty", "CyclicalRestPenaltyWithLinear",
)
PENALTY_SHORT = {
    "NonePenalty": "None",
    "LinearPenalty": "Linear",
    "LinearWithRangePenalty": "Range",
    "CyclicalRestPenalty": "Rest",
    "CyclicalRestPenaltyWithLinear": "Rest+Linear",
}
METHOD_LABEL = {"o": "OR-Tools", "s": "Annealing", "f": "Sequential"}
METHOD_COLOR = {"o": "#2878b5", "s": "#d2691e", "f": "#6c757d"}


def make_penalty(variant, kind):
    module_name = {
        "simple": "one_machine_problem_simple_task",
        "single_interval": "one_machine_problem_interval_task",
        "multi_interval": "multi_machine_problem_interval_task",
    }[variant]
    module = import_module(f"gems_python.{module_name}.penalty.penalty_class")
    cls = getattr(module, kind)
    if kind == "NonePenalty":
        return cls()
    if kind == "LinearPenalty":
        return cls(3)
    if kind == "LinearWithRangePenalty":
        return cls(-2, 4, 2, 2)
    if kind == "CyclicalRestPenalty":
        return cls(0, 20, [(10, 14)])
    return cls(0, 20, [(10, 14)], 3)


def make_case(variant, kind, directory):
    penalty = lambda: make_penalty(variant, kind)
    if variant == "simple":
        manager = SimpleExperiments([], directory)
        manager.tasks = [
            OneMachineTask(optimal, duration, penalty(), f"t{index}", "benchmark", "fixed")
            for index, (optimal, duration) in enumerate(
                ((10, 6), (12, 5), (15, 4), (18, 5))
            )
        ]
        return manager
    if variant == "single_interval":
        groups = [
            SingleGroup(
                optimal_start_time=optimal,
                penalty_type=penalty(),
                tasks=[
                    SingleTask(5, f"g{index}a", interval=2),
                    SingleTask(4, f"g{index}b", interval=3),
                ],
            )
            for index, optimal in enumerate((10, 12, 14))
        ]
        return SingleExperiments([], directory, task_groups=groups)
    groups = [
        MultiGroup(
            optimal_start_time=optimal,
            penalty_type=penalty(),
            tasks=[
                MultiTask(5, f"g{index}a", 0, interval=2),
                MultiTask(3, f"g{index}b", 0, interval=2),
            ],
        )
        for index, optimal in enumerate((10, 11, 12, 13))
    ]
    return MultiExperiments(
        [], directory,
        machine_list=MachineList([Machine(0, "m0"), Machine(0, "m1")]),
        task_groups=groups,
    )


def overlap_minutes(intervals):
    events = []
    for start, end in intervals:
        events.extend(((start, 1), (end, -1)))
    if not events:
        return 0
    events.sort(key=lambda event: (event[0], event[1]))
    active = 0
    last = events[0][0]
    total = 0
    for minute, delta in events:
        total += (minute - last) * max(active - 1, 0)
        active += delta
        last = minute
    return total


def evaluate(manager, variant):
    details = []
    timing = 0
    interval_errors = 0
    assignment_errors = 0
    by_machine = {}
    if variant == "simple":
        for index, task in enumerate(manager.tasks):
            if task.scheduled_time is None:
                raise ValueError(f"Task {index} has no scheduled time")
            timing += task.penalty_type.calculate_penalty(
                task.scheduled_time, task.optimal_time
            )
            details.append({
                "group": index, "task": index, "start": task.scheduled_time,
                "duration": task.processing_time, "machine": "m0",
                "optimal": task.optimal_time,
            })
            by_machine.setdefault("m0", []).append(
                (task.scheduled_time, task.scheduled_time + task.processing_time)
            )
    else:
        machine_by_id = (
            {machine.machine_id: machine for machine in manager.machine_list.machines}
            if variant == "multi_interval" else {}
        )
        for index, group in enumerate(manager.task_groups):
            if not group.tasks or group.tasks[0].scheduled_time is None:
                raise ValueError(f"Group {index} has no first scheduled task")
            group_start = group.tasks[0].scheduled_time - group.tasks[0].interval
            timing += group.penalty_type.calculate_penalty(
                group_start, group.optimal_start_time
            )
            for task_index, task in enumerate(group.tasks):
                if task.scheduled_time is None:
                    raise ValueError(f"Group {index} task {task_index} is unscheduled")
                expected = (
                    group_start + task.interval if task_index == 0
                    else group.tasks[task_index - 1].scheduled_time
                    + group.tasks[task_index - 1].processing_time + task.interval
                )
                interval_errors += task.scheduled_time != expected
                machine = "m0" if variant == "single_interval" else task.allocated_machine_id
                if variant == "multi_interval":
                    allocated = machine_by_id.get(machine)
                    assignment_errors += (
                        allocated is None
                        or allocated.machine_type != task.optimal_machine_type
                    )
                details.append({
                    "group": group.task_group_id, "task": task_index,
                    "start": task.scheduled_time, "duration": task.processing_time,
                    "machine": machine, "optimal": group.optimal_start_time,
                })
                if machine is not None:
                    by_machine.setdefault(machine, []).append(
                        (task.scheduled_time, task.scheduled_time + task.processing_time)
                    )
    return {
        "timing_penalty": timing,
        "overlap_minutes": sum(overlap_minutes(v) for v in by_machine.values()),
        "assignment_errors": assignment_errors,
        "interval_errors": interval_errors,
        "tasks": details,
    }


class SolverStatusHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.status = None

    def emit(self, record):
        match = re.search(r"status=(OPTIMAL|FEASIBLE)", record.getMessage())
        if match:
            self.status = match.group(1)


def run_manager(manager, variant, method, seed):
    """Run scheduling only and evaluate its result using the same independent metrics."""
    if seed is not None:
        random.seed(seed)
    status_handler = SolverStatusHandler()
    previous_level = LOGGER.level
    LOGGER.addHandler(status_handler)
    LOGGER.setLevel(logging.INFO)
    output = io.StringIO()
    start = time.perf_counter()
    error = None
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            manager.execute_scheduling(method)
        metrics = evaluate(manager, variant)
    except Exception as exc:
        metrics = {}
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - start
        LOGGER.removeHandler(status_handler)
        LOGGER.setLevel(previous_level)
    return {
        "variant": variant, "method": method, "seed": seed,
        "seconds": round(elapsed, 6),
        "status": error or (status_handler.status if method == "o" else "FINISHED"),
        **metrics,
    }


def run_case(variant, kind, method, seed):
    with tempfile.TemporaryDirectory(prefix="gems-scheduler-") as directory:
        manager = make_case(variant, kind, Path(directory))
        return {
            "penalty": kind,
            **run_manager(manager, variant, method, seed),
        }


def make_charts(rows, output_dir, seeds):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for ax, variant in zip(axes, VARIANTS):
        for index, kind in enumerate(PENALTIES):
            selected = [
                row for row in rows if row["variant"] == variant
                and row["penalty"] == kind and "timing_penalty" in row
            ]
            for method, offset in (("o", -0.22), ("s", 0), ("f", 0.22)):
                values = [row["timing_penalty"] for row in selected if row["method"] == method]
                if not values:
                    continue
                if method == "s":
                    ax.scatter(
                        [index + offset] * len(values), values, s=23,
                        color=METHOD_COLOR[method], marker="o", zorder=3,
                    )
                else:
                    ax.scatter(
                        index + offset, values[0], s=55, color=METHOD_COLOR[method],
                        marker="D" if method == "o" else "s", zorder=4,
                    )
        ax.set_title(variant.replace("_", " "))
        ax.set_xticks(range(len(PENALTIES)), [PENALTY_SHORT[x] for x in PENALTIES], rotation=32)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=-0.1)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Timing penalty (symlog)")
    fig.legend(
        handles=[mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
                 for m in ("o", "s", "f")],
        loc="outside upper center", ncol=3,
    )
    fig.savefig(output_dir / "penalties.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for ax, variant in zip(axes, VARIANTS):
        for index, kind in enumerate(PENALTIES):
            selected = [row for row in rows if row["variant"] == variant and row["penalty"] == kind]
            for method, offset in (("o", -0.22), ("s", 0), ("f", 0.22)):
                values = [row["seconds"] for row in selected if row["method"] == method]
                if values:
                    ax.scatter(
                        [index + offset] * len(values), values, s=28,
                        color=METHOD_COLOR[method],
                        marker="D" if method == "o" else "o", zorder=3,
                    )
        ax.set_title(variant.replace("_", " "))
        ax.set_xticks(range(len(PENALTIES)), [PENALTY_SHORT[x] for x in PENALTIES], rotation=32)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Wall time (seconds, log)")
    fig.legend(
        handles=[mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
                 for m in ("o", "s", "f")],
        loc="outside upper center", ncol=3,
    )
    fig.savefig(output_dir / "runtime.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for ax, variant in zip(axes, VARIANTS):
        for index, kind in enumerate(PENALTIES):
            selected = [
                row for row in rows if row["variant"] == variant
                and row["penalty"] == kind and "overlap_minutes" in row
            ]
            for method, offset in (("o", -0.22), ("s", 0), ("f", 0.22)):
                values = [
                    row["overlap_minutes"] for row in selected
                    if row["method"] == method
                ]
                if values:
                    ax.scatter(
                        [index + offset] * len(values), values, s=32,
                        color=METHOD_COLOR[method],
                        marker="D" if method == "o" else "o", zorder=3,
                    )
        ax.set_title(variant.replace("_", " "))
        ax.set_xticks(
            range(len(PENALTIES)), [PENALTY_SHORT[x] for x in PENALTIES],
            rotation=32,
        )
        maximum = max(
            row.get("overlap_minutes", 0) for row in rows
            if row["variant"] == variant
        )
        ax.set_ylim(-0.1, max(1, maximum) + 0.5)
        ax.set_yticks(range(max(1, maximum) + 1))
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Physical-machine overlap (minutes)")
    fig.legend(
        handles=[mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m])
                 for m in ("o", "s", "f")],
        loc="outside upper center", ncol=3,
    )
    fig.savefig(output_dir / "overlap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout="constrained")
    for ax, variant in zip(axes, VARIANTS):
        selected = [
            row for row in rows if row["variant"] == variant
            and row["penalty"] == "LinearPenalty"
            and (row["method"] != "s" or row["seed"] == seeds[0])
            and "tasks" in row
        ]
        lanes = []
        for row in selected:
            machines = (
                ("m0", "m1") if variant == "multi_interval" else ("m0",)
            )
            for machine in machines:
                lanes.append((row, machine))
        for lane_index, (row, machine) in enumerate(lanes):
            for task in row["tasks"]:
                if task["machine"] != machine:
                    continue
                ax.barh(
                    lane_index, task["duration"], left=task["start"],
                    height=0.68, color=plt.get_cmap("tab10")(task["group"] % 10),
                    edgecolor="black", linewidth=0.6, alpha=0.85,
                )
                ax.text(
                    task["start"] + task["duration"] / 2, lane_index,
                    str(task["group"]), va="center", ha="center", fontsize=7,
                )
        ax.set_yticks(
            range(len(lanes)),
            [
                f"{METHOD_LABEL[row['method']]} {machine}"
                if variant == "multi_interval" else METHOD_LABEL[row["method"]]
                for row, machine in lanes
            ],
        )
        ax.set_title(f"{variant.replace('_', ' ')} / LinearPenalty")
        ax.set_xlabel("Minute from reference")
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()
    fig.savefig(output_dir / "gantt.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        for kind in PENALTIES:
            for method in ("o", "f", "s") if variant != "simple" else ("o", "s"):
                for seed in (args.seeds if method == "s" else [None]):
                    row = run_case(variant, kind, method, seed)
                    rows.append(row)
                    print(
                        variant, PENALTY_SHORT[kind], method, seed, row["status"],
                        row.get("timing_penalty"), row.get("overlap_minutes"),
                        row["seconds"], flush=True,
                    )
    with (args.output_dir / "runs.csv").open("w", newline="") as file:
        columns = (
            "variant", "penalty", "method", "seed", "status", "timing_penalty",
            "overlap_minutes", "assignment_errors", "interval_errors", "seconds",
        )
        writer = csv.DictWriter(
            file, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "python": sys.version.split()[0],
        "ortools": ortools.__version__,
        "simanneal": simanneal.__version__,
        "matplotlib": matplotlib.__version__,
        "solver_time_limit_seconds": SOLVER_TIME_LIMIT_SECONDS,
        "annealing_steps": 100000,
        "annealing_tmax": 25000.0,
        "annealing_tmin": 1.0,
        "reference_minute": 0,
        "seeds": args.seeds,
        "cases": {
            "simple": {"optimal": [10, 12, 15, 18], "durations": [6, 5, 4, 5]},
            "single_interval": {
                "optimal": [10, 12, 14], "groups": [[(2, 5), (3, 4)]] * 3,
            },
            "multi_interval": {
                "optimal": [10, 11, 12, 13], "machines": ["m0", "m1"],
                "groups": [[(2, 5), (2, 3)]] * 4,
            },
        },
        "penalty_parameters": {
            kind: make_penalty("simple", kind).to_dict() for kind in PENALTIES
        },
        "rows": rows,
    }
    (args.output_dir / "runs.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    make_charts(rows, args.output_dir, args.seeds)
    print(f"Wrote {len(rows)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
