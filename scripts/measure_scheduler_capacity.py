"""Deterministic capacity study with fixed CP-SAT search limits, not total budgets.

Each trial uses a fresh process. Inputs and outputs survive failures; summaries
can be reconstructed from results.jsonl without running a solver again.
"""

import argparse
import contextlib
import copy
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import logging
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ortools
import pandas
import polars

from gems_python.common import dataframes
from gems_python.common.scheduling import LOGGER, validate_groups
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import (
    CyclicalRestPenaltyWithLinear, LinearPenalty, LinearWithRangePenalty,
)
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task, TaskGroup, TaskGroupStatus, TaskStatus,
)
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    Experiment, Experiments, State,
)


DEFAULT_CONFIG = Path(__file__).with_name("scheduler_capacity_config.json")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class CapacityState(State):
    """Three deterministic group cycles, with fixed observations and no noise."""

    def __init__(self, pattern, index, config):
        self.pattern, self.index, self.config = pattern, index, config

    def task_generator(self, history):
        cfg = self.config
        if len(history) >= cfg["group_cycles"] * len(cfg["processing_minutes"]):
            return TaskGroup(0, LinearPenalty(3), [])
        latest = int(history["time"].iloc[-1]) if len(history) else 10
        penalty = LinearPenalty(3)
        if self.pattern == "mixed_penalties":
            if self.index % 3 == 1:
                penalty = LinearWithRangePenalty(-2, 4, 2, 3)
            elif self.index % 3 == 2:
                penalty = CyclicalRestPenaltyWithLinear(
                    0, cfg["rest_cycle_minutes"], [tuple(r) for r in cfg["rest_ranges"]], 3)
        types = [0, 1, 2] if self.pattern == "heterogeneous" else [0, 0, 0]
        tasks = [Task(duration, f"operation_{j}", machine_type, interval=interval)
                 for j, (duration, interval, machine_type) in enumerate(zip(
                     cfg["processing_minutes"], cfg["interval_minutes"], types))]
        return TaskGroup(latest + self.index % 4, penalty, tasks)

    def transition_function(self, history):
        return self.state_name

    def dummy_output(self, history, task_group_id, task_id):
        raise RuntimeError("Capacity study requires its fixed observation provider")


# Use the importable class itself so saved JSON and pickle have the same identity.
if __name__ == "__main__":
    from scripts.measure_scheduler_capacity import CapacityState


def observe(state, history, group, task, event_time):
    return dataframes.make_frame({"time": [event_time], "value": [len(history) + 1]})


def make_case(pattern, size, directory, config):
    machines = MachineList([
        Machine(kind, f"type{kind}_m{index}")
        for kind, count in enumerate(config["machine_counts"][pattern])
        for index in range(count)
    ])
    with contextlib.redirect_stdout(io.StringIO()):
        experiments = [
            Experiment(f"experiment_{i}", [CapacityState(pattern, i, config)], "CapacityState",
                       experiment_uuid=f"capacity-{pattern}-{i:04d}")
            for i in range(size)
        ]
        # Capacity budgets are fixed per solve, even though runtime now retries UNKNOWN.
        manager = Experiments(experiments, directory, machine_list=machines, unknown_policy="stop")
    if pattern == "replanning":
        reference = 21 + size * 20
        manager.reference_time = reference
        for i, group in enumerate(manager.task_groups):
            if i % 4:
                group.optimal_start_time = reference + i % 4
                continue
            group.tasks[1].interval += size * 20
            group.schedule_tasks(10 + i * 20)
            group.tasks[0].task_status = TaskStatus.COMPLETED
            group.status = TaskGroupStatus.IN_PROGRESS
            for task in group.tasks:
                task.allocated_machine_id = "type0_m0"
                if task.scheduled_time <= reference < task.scheduled_time + task.processing_time:
                    task.task_status = TaskStatus.IN_PROGRESS
            # This snapshot follows one already completed observation.
            manager.experiments[i].shared_variable_history = dataframes.make_frame({
                "time": [group.tasks[0].scheduled_time + group.tasks[0].processing_time],
                "value": [1],
            })
            manager.experiments[i].current_task_group = copy.deepcopy(group)
    return manager


def snapshot(manager):
    value = manager.to_dict()
    # Exclude transient output-directory names from paired input hashes.
    value["attributes"]["parent_dir_path"] = {
        "_gems_serialization": 1, "kind": "path", "class": "pathlib:PosixPath",
        "value": "/tmp/gems-capacity-replay",
    }
    return value


def dimensions(manager):
    groups = [g for g in manager.task_groups if not g.is_completed()]
    tasks = [t for g in groups for t in g.tasks if not t.completed]
    machines = manager.machine_list.machines
    return {
        "active_experiments": len({g.experiment_uuid for g in groups}),
        "group_count": len(groups), "task_count": len(tasks),
        "machine_count": len(machines),
        "machine_type_count": len({m.machine_type for m in machines}),
        "assignment_candidates": sum(sum(m.machine_type == t.optimal_machine_type for m in machines)
                                     for t in tasks),
        "fixed_task_count": sum(not t.completed for g in groups if not g.is_not_started()
                                for t in g.tasks),
        "in_progress_task_count": sum(t.task_status == TaskStatus.IN_PROGRESS for t in tasks),
        "history_rows": sum(len(e.shared_variable_history) for e in manager.experiments),
    }


def percentile(values, q=0.95):
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    low, high = math.floor(at), math.ceil(at)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def quality(manager):
    groups = [g for g in manager.task_groups if g.tasks and not g.is_completed()]
    delays, penalty, movable_penalty = [], 0, 0
    intervals = []
    for group in groups:
        start = group.tasks[0].scheduled_time - group.tasks[0].interval
        cost = group.penalty_type.calculate_penalty(start, group.optimal_start_time)
        penalty += cost
        if group.is_not_started():
            movable_penalty += cost
            delays.append(max(0, start - group.optimal_start_time))
        intervals += [(t.scheduled_time, t.scheduled_time + t.processing_time)
                      for t in group.tasks if not t.completed]
    span = max((end for _, end in intervals), default=manager.reference_time) - min(
        (start for start, _ in intervals), default=manager.reference_time)
    occupied = sum(end - start for start, end in intervals)
    return {
        "timing_penalty": penalty, "movable_penalty": movable_penalty,
        "delay_median_minutes": statistics.median(delays) if delays else 0,
        "delay_p95_minutes": percentile(delays) if delays else 0,
        "delay_max_minutes": max(delays, default=0),
        "schedule_span_minutes": span,
        "machine_utilization": occupied / (span * len(manager.machine_list.machines)) if span else 0,
    }


class Reports(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        if hasattr(record, "scheduling_report"):
            self.rows.append(copy.deepcopy(record.scheduling_report))


def classify(reports, error=None):
    if error and not reports:
        return "EXECUTION_ERROR"
    for report in reports:
        if not report.get("valid"):
            return {
                "UNKNOWN": "NO_SOLUTION", "INFEASIBLE": "INFEASIBLE",
                "MODEL_INVALID": "INVALID_MODEL", "INVALID_INPUT": "INVALID_INPUT",
                "INVALID_OUTPUT": "INVALID_OUTPUT",
            }.get(report["status"], "EXECUTION_ERROR")
    if error:
        return "EXECUTION_ERROR"
    return "VALID"


def run_trial(request, directory):
    manager = Experiments.from_dict(request["input"])
    restored_input_hash = fingerprint(snapshot(manager))
    manager.parent_dir_path = directory / "workspace"
    manager.parent_dir_path.mkdir(parents=True, exist_ok=True)
    limit = request["search_limit_seconds"]
    handler, previous = Reports(), LOGGER.level
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    started = time.perf_counter()
    updates, error = [], None
    io_times = {}
    before = dimensions(manager)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if request["phase"] != "operation":
                manager.execute_scheduling(solver_time_limit_seconds=limit)
                validate_groups(manager.task_groups, manager.reference_time, manager.machine_list)
                if request["phase"] == "io":
                    saving = time.perf_counter()
                    manager.save_all(save_dir=directory / "saved")
                    io_times["save_seconds"] = time.perf_counter() - saving
                    draw_dir = directory / "drawing"
                    draw_dir.mkdir()
                    drawing = time.perf_counter()
                    manager.generate_gantt_chart(save_dir=draw_dir)
                    io_times["draw_seconds"] = time.perf_counter() - drawing
                    import matplotlib.pyplot as plt
                    plt.close("all")
            else:
                cfg = request["config"]
                for step in range(cfg["operation_steps"]):
                    checking = snapshot(manager)
                    call_start = len(handler.rows)
                    update_started = time.perf_counter()
                    additions = []
                    if step in cfg["operation_add_at_steps"]:
                        for _ in range(cfg["operation_add_count"]):
                            i = len(manager.experiments)
                            experiment = Experiment(
                                f"experiment_{i}", [CapacityState(request["pattern"], i, cfg)],
                                "CapacityState", experiment_uuid=f"capacity-{request['pattern']}-{i:04d}",
                            )
                            additions.append(experiment.experiment_uuid)
                            manager.add_experiment(experiment, solver_time_limit_seconds=limit,
                                                   save_step=False)
                    result = manager.simulate_one(
                        observation_provider=observe, solver_time_limit_seconds=limit)
                    elapsed = time.perf_counter() - update_started
                    record = {
                        "step": step, "input_hash": fingerprint(checking), "input": checking,
                        "dimensions": dimensions(manager), "result": result,
                        "added_experiments": additions, "update_seconds": elapsed,
                        "calls": copy.deepcopy(handler.rows[call_start:]),
                    }
                    updates.append(record)
                    if result["status"] == "no_tasks":
                        break
                    if result["status"] != "ok":
                        raise RuntimeError(result)
                    validate_groups(manager.task_groups, manager.reference_time, manager.machine_list)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (directory / "error.txt").write_text(traceback.format_exc())
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(previous)
    row = {
        key: request[key] for key in
        ("pattern", "size", "phase", "repetition", "search_limit_seconds", "input_hash")
    }
    row.update(
        outcome=classify(handler.rows, error), error=error,
        actual_input_hash=restored_input_hash, dimensions=before,
        total_seconds=time.perf_counter() - started,
        memory_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        reports=handler.rows, updates=updates,
        **io_times,
    )
    if row["outcome"] == "VALID":
        row.update(quality(manager))
    (directory / "output.json").write_text(manager.to_json())
    (directory / "result.json").write_text(json.dumps(row, indent=2))
    return row


def environment():
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                if line.startswith("model name")), platform.processor())
    memory = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    return {
        "commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
        "python": sys.version, "executable": sys.executable,
        "ortools": ortools.__version__, "pandas": pandas.__version__, "polars": polars.__version__,
        "platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
        "memory": memory, "load_average": os.getloadavg(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dependencies": {name: version(name) for name in
                         ("ortools", "pandas", "polars", "pyarrow", "simanneal", "matplotlib")},
        "lock_sha256": hashlib.sha256(Path("poetry.lock").read_bytes()).hexdigest(),
        "memory_measurement": "fresh worker ru_maxrss; includes imports; operation peak is cumulative",
    }


def summarize(output):
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    summary = []
    keys = sorted({(r["pattern"], r["size"], r["phase"], r["search_limit_seconds"]) for r in rows})
    for pattern, size, phase, limit in keys:
        selected = [r for r in rows if (r["pattern"], r["size"], r["phase"], r["search_limit_seconds"])
                    == (pattern, size, phase, limit)]
        searches = [sum(x["search_seconds"] for x in r.get("reports", [])) for r in selected]
        totals = [r["total_seconds"] for r in selected]
        summary.append({
            "pattern": pattern, "size": size, "phase": phase, "search_limit_seconds": limit,
            "repetitions": len(selected), "valid": sum(r["outcome"] == "VALID" for r in selected),
            "outcomes": {k: sum(r["outcome"] == k for r in selected) for k in sorted({r["outcome"] for r in selected})},
            "search_median_seconds": statistics.median(searches),
            "search_p95_seconds": percentile(searches), "search_max_seconds": max(searches),
            "total_median_seconds": statistics.median(totals),
            "total_p95_seconds": percentile(totals), "total_max_seconds": max(totals),
            "peak_memory_mib": max(r.get("memory_peak_mib", 0) for r in selected),
            "statuses": sorted({x["status"] for r in selected for x in r.get("reports", [])}),
            "dimensions": selected[0].get("dimensions", {}),
        })
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    columns = ["pattern", "size", "phase", "search_limit_seconds", "repetitions", "valid",
               "search_median_seconds", "search_p95_seconds", "search_max_seconds",
               "total_median_seconds", "total_p95_seconds", "total_max_seconds", "peak_memory_mib",
               "statuses", "outcomes", "dimensions"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    table = [
        "# Fixed-search-limit capacity study",
        "",
        "Limits apply to each CP-SAT search. Construction, application and validation times are "
        "reference measurements and do not determine success. FEASIBLE is not an optimality proof. "
        "UNKNOWN is not an infeasibility proof. Delays use virtual minutes; computation uses real seconds.",
        "",
        "| Pattern | Size | Phase | Limit s | Valid / runs | Search median / p95 / max s | Status |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in summary:
        table.append(
            f"| {row['pattern']} | {row['size']} | {row['phase']} | {row['search_limit_seconds']:g} | "
            f"{row['valid']}/{row['repetitions']} | {row['search_median_seconds']:.3f} / "
            f"{row['search_p95_seconds']:.3f} / {row['search_max_seconds']:.3f} | "
            f"{', '.join(row['statuses'])} |")
    table += ["", "p95 uses linear interpolation; five repetitions do not establish an operational guarantee.",
              "Memory includes the Python/OR-Tools imports. Utilization describes the pending schedule, "
              "not a calibrated facility throughput model.", ""]
    (output / "summary.md").write_text("\n".join(table))
    return summary


def study(output, config):
    if config["seed"] != 0 or config["workers"] != 1:
        raise ValueError("This study holds the repository solver seed=0 and workers=1 fixed")
    if any(len(config[key]) != 3 for key in ("processing_minutes", "interval_minutes")):
        raise ValueError("The five workload definitions require three tasks per group")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "environment.json").write_text(json.dumps(environment(), indent=2))
    (output / "results.jsonl").touch()
    started = time.monotonic()
    deadline = started + config["total_runtime_seconds"]
    rows, skipped = [], []

    def trial(pattern, size, limit, phase, repetition):
        # The six-hour controller budget is independent of the search limit.
        if time.monotonic() + limit + 10 > deadline:
            skipped.append({"pattern": pattern, "size": size, "limit": limit,
                            "phase": phase, "repetition": repetition})
            return None
        name = f"{len(rows):04d}-{phase}-{pattern}-{size}-{limit:g}-{repetition}"
        directory = output / "trials" / name
        directory.mkdir(parents=True)
        manager = make_case(pattern, size, directory / "workspace", config)
        input_value = snapshot(manager)
        request = dict(pattern=pattern, size=size, phase=phase, repetition=repetition,
                       search_limit_seconds=limit, input_hash=fingerprint(input_value),
                       input=input_value, config=config)
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request, indent=2))
        timeout = min(deadline - time.monotonic(),
                      (limit * (config["operation_steps"] * 3 + 1) + 120)
                      if phase == "operation" else limit + 120)
        wall = time.perf_counter()
        with (directory / "worker.log").open("w") as log:
            try:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False,
                )
                if process.returncode or not (directory / "result.json").exists():
                    raise RuntimeError(f"Worker exit={process.returncode}; see worker.log")
                result = json.loads((directory / "result.json").read_text())
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                result = {key: request[key] for key in
                          ("pattern", "size", "phase", "repetition", "search_limit_seconds", "input_hash")}
                result.update(outcome="WORKER_TIMEOUT" if isinstance(exc, subprocess.TimeoutExpired)
                              else "EXECUTION_ERROR", error=str(exc), reports=[],
                              total_seconds=time.perf_counter() - wall, dimensions=dimensions(manager))
        result["artifact_directory"] = str(directory.relative_to(output))
        result["worker_process_seconds"] = time.perf_counter() - wall
        rows.append(result)
        with (output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
        summarize(output)
        print(f"{name}: {result['outcome']} statuses="
              f"{[r['status'] for r in result.get('reports', [])]} "
              f"seconds={result['total_seconds']:.3f}", flush=True)
        return result

    # Baseline ladder: one trial per size/limit, stopping one size above first failure.
    evaluated = {}
    for pattern in config["patterns"]:
        failures = {}
        for index, size in enumerate(config["sizes"]):
            for limit in config["search_limits_seconds"]:
                if limit in failures and index > failures[limit] + 1:
                    continue
                row = trial(pattern, size, limit, "scan", 0)
                if row is None:
                    continue
                evaluated[(pattern, size, limit)] = row
                if row["outcome"] != "VALID":
                    failures.setdefault(limit, index)
    # Repeat each limit's first-failure neighborhood, or last two sizes if no failure.
    for pattern in config["patterns"]:
        for limit in config["search_limits_seconds"]:
            observed = [s for s in config["sizes"] if (pattern, s, limit) in evaluated]
            if not observed:
                continue
            first = next((i for i, s in enumerate(observed)
                          if evaluated[(pattern, s, limit)]["outcome"] != "VALID"), None)
            neighborhood = observed[-2:] if first is None else observed[max(0, first - 1):first + 2]
            for size in neighborhood:
                for repetition in range(1, config["boundary_repetitions"]):
                    trial(pattern, size, limit, "scan", repetition)
    # Operational runs use the smaller of the limits' all-valid measured sizes.
    # Save/draw reference costs are separate small trials, not operational latency.
    for pattern in config["patterns"]:
        trial(pattern, min(config["sizes"]), config["search_limits_seconds"][0], "io", 0)
    for pattern in config["patterns"]:
        best = []
        for limit in config["search_limits_seconds"]:
            valid = [s for s in config["sizes"] if any(
                r["pattern"] == pattern and r["size"] == s and r["phase"] == "scan"
                and r["search_limit_seconds"] == limit for r in rows)
                and all(r["outcome"] == "VALID" for r in rows if
                        r["pattern"] == pattern and r["size"] == s and r["phase"] == "scan"
                        and r["search_limit_seconds"] == limit)]
            best.append(max(valid, default=min(config["sizes"])))
        size = min(best)
        for limit in config["search_limits_seconds"]:
            trial(pattern, size, limit, "operation", 0)
    (output / "completion.json").write_text(json.dumps({
        "elapsed_seconds": time.monotonic() - started, "trial_count": len(rows),
        "skipped": skipped, "fully_measured": not skipped,
        "note": "512 success is a censored bound, not a maximum. Nonmonotonic behavior is possible.",
    }, indent=2))
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_trial(json.loads(args.worker.read_text()), args.worker.parent)
    elif args.summarize:
        summarize(args.summarize)
    elif args.output:
        study(args.output, json.loads(args.config.read_text()))
    else:
        parser.error("--output or --summarize is required")


if __name__ == "__main__":
    main()
