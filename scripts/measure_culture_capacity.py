"""Execute complete empirical ROUND3 culture runs, retaining every outcome.

One initial plan, then one production update/replan per received observation.
No redundant pre-receipt solve is performed on an already valid schedule.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import gzip
import hashlib
from importlib.metadata import version
import json
import logging
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

# Independent trials each use one solver/BLAS thread.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "POLARS_MAX_THREADS", "ARROW_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gems30-mpl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gems_python.common.scheduling import LOGGER, validate_groups
from gems_python.multi_machine_problem_interval_task.task_info import TaskStatus
from scripts.culture_capacity import (
    CultureObservations, _predict_from_observations, build_manager, draw_cultures,
)
from scripts.extract_round3_culture_data import write_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class Reports(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        if hasattr(record, "scheduling_report"):
            self.rows.append(dict(record.scheduling_report))


def advance(manager, observations):
    """Consume the valid plan with the same event semantics as simulate_one."""
    group, task = min(
        ((g, t) for g in manager.runnable_groups() for t in g.tasks if not t.completed),
        key=lambda pair: (pair[1].scheduled_time+pair[1].processing_time,
                          pair[1].scheduled_time, pair[0].task_group_id, pair[1].task_id))
    experiment = next(e for e in manager.experiments if e.experiment_uuid == group.experiment_uuid)
    state = experiment.states[experiment.current_state_index]
    event_time = max(manager.reference_time, task.scheduled_time+task.processing_time)
    frame = observations(state, experiment.history_for_state(), group, task, event_time)
    for existing in manager.runnable_groups():
        if existing.is_not_started():
            for operation in existing.tasks:
                if not operation.completed and operation.scheduled_time <= event_time:
                    operation.task_status = TaskStatus.IN_PROGRESS
            existing.status_update()
    manager.reference_time = event_time
    manager.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
        group.task_group_id, task.task_id, frame, scheduling_method="o",
        optimal_time_reference_time=event_time,
        task_group_uuid=group.task_group_uuid,
        result_id=f"culture-simulation:{group.task_group_uuid}:{task.task_id}")


def fixed_snapshot(manager):
    return {(g.task_group_uuid, t.task_id): (t.scheduled_time, t.allocated_machine_id)
            for g in manager.task_groups if not g.is_not_started()
            for t in g.tasks if not t.completed}


def independent_event_check(events):
    """Inspect actual completed occupancy and fixed group intervals."""
    errors = []
    occupied = sorted(((e["start"], e["end"], e) for e in events if e["duration"] > 0),
                      key=lambda row: (row[0], row[1]))
    previous_end = -1
    for start, end, event in occupied:
        if start < previous_end:
            errors.append(f"Machine overlap at {event['group_uuid']}:{event['task_id']}")
        if end-start != event["duration"]:
            errors.append("Duration mismatch")
        if event["machine"] != "ROUND3-shared-resource":
            errors.append("Incorrect machine assignment")
        previous_end = max(end, previous_end)
    groups = {}
    for event in events:
        groups.setdefault(event["group_uuid"], []).append(event)
    for group_events in groups.values():
        group_events.sort(key=lambda event: event["task_id"])
        prior = None
        for event in group_events:
            expected = (event["group_start"]+event["interval"] if prior is None else
                        prior["end"]+event["interval"])
            if event["start"] != expected:
                errors.append("Group interval mismatch")
            if prior is not None and event["group_start"] != prior["group_start"]:
                errors.append("Started group moved")
            prior = event
    return errors


def run_trial(request):
    hek, ips, repetition = request["case"]
    directory = Path(request["output"]) / f"h{hek:02d}_i{ips:02d}_r{repetition:02d}"
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    config, model = request["config"], request["model"]
    _predict_from_observations.cache_clear()
    cultures = draw_cultures(model, config, hek, ips, repetition)
    write_json(directory / "input.json", {
        "case": [hek, ips, repetition], "cultures": cultures,
        "config_hash": fingerprint(config), "model_hash": fingerprint(model),
    })
    provider = CultureObservations(cultures, config)
    handler = Reports()
    level = LOGGER.level
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    start = time.perf_counter()
    outcome, failure = None, None
    fixed_errors = []
    manager = None
    write_json(directory / "progress.json", {"pid": os.getpid(), "stage": "starting"})
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), (
                contextlib.redirect_stderr(sink)):
            manager = build_manager(cultures, config, directory / "workspace")
            manager.execute_scheduling()
            for step in range(config["max_events_per_run"]):
                if step % 20 == 0:
                    write_json(directory / "progress.json", {
                        "pid": os.getpid(), "stage": "running", "events": len(provider.events),
                        "virtual_minutes": manager.reference_time,
                        "wall_seconds": time.perf_counter()-start,
                        "solver_statuses": dict(Counter(r["status"] for r in handler.rows)),
                    })
                if manager.manager_status == "STOPPED":
                    outcome = (manager.stop_diagnostic or {}).get("kind", "SOLVER_STOP")
                    failure = manager.stop_diagnostic
                    break
                if not any(e.experiment_uuid != "refill" for e in manager.experiments):
                    outcome = "COMPLETED"
                    break
                if manager.reference_time > config["max_virtual_days"]*1440:
                    outcome = "VIRTUAL_HORIZON_CENSORED"
                    break
                if time.perf_counter()-start > config["max_wall_seconds_per_run"]:
                    outcome = "WALL_CENSORED"
                    break
                before = fixed_snapshot(manager)
                advance(manager, provider)
                after = fixed_snapshot(manager)
                fixed_errors += [str(key) for key in before.keys() & after.keys()
                                 if before[key] != after[key]]
                if manager.manager_status != "STOPPED":
                    validate_groups(manager.task_groups, manager.reference_time,
                                    manager.machine_list)
            else:
                outcome = "EVENT_COUNT_CENSORED"
    except Exception as error:
        diagnostic = manager.stop_diagnostic if manager else None
        category = diagnostic.get("kind") if diagnostic else None
        outcome = (category if category in {"UNKNOWN", "INFEASIBLE", "MODEL_INVALID"}
                   else "EXECUTION_ERROR")
        failure = {"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(),
                   "stop_diagnostic": manager.stop_diagnostic if manager else None}
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(level)
    events = provider.events
    checks = independent_event_check(events)
    completed = sorted({e["culture_id"] for e in events if e["operation"].endswith("Sampling")})
    targets = [e for e in events if e["target"] is not None]
    complete = outcome == "COMPLETED" and len(completed) == hek+ips
    if complete and len(targets) != 3*(hek+ips):
        checks.append("Completed run has an incorrect count of targeted operations")
    statuses = Counter(row["status"] for row in handler.rows)
    if outcome == "SOLVER_STOP" or outcome is None:
        outcome = "UNKNOWN" if "UNKNOWN" in statuses else "SOLVER_STOP"
    primary = complete and all(e["primary_pass"] for e in targets)
    supplementary = complete and all(e["supplementary_pass"] for e in targets)
    end = manager.reference_time if manager else 0
    occupied = sum(e["duration"] for e in events)
    occupied_cells = sum(e["duration"] for e in events if e["cell_type"] != "refill")
    density_errors = [abs(e["error"]) for e in targets]
    output = {
        "hek": hek, "ips": ips, "repetition": repetition, "outcome": outcome,
        "completed": complete, "completed_cultures": len(completed),
        "primary_accepted": primary, "supplementary_accepted": supplementary,
        "primary_violations": sum(not e["primary_pass"] for e in targets),
        "supplementary_violations": sum(not e["supplementary_pass"] for e in targets),
        "scored_operations": len(targets), "events": len(events),
        "virtual_minutes": end, "wall_seconds": time.perf_counter()-start,
        "machine_utilization": occupied/end if end else 0,
        "culture_occupancy_minutes": occupied_cells,
        "refill_occupancy_minutes": occupied-occupied_cells,
        "density_abs_error_mean": float(np.mean(density_errors)) if targets else None,
        "density_abs_error_max": max(density_errors, default=None),
        "solver_calls": len(handler.rows), "solver_statuses": dict(statuses),
        "search_seconds": sum(row.get("search_seconds", 0) for row in handler.rows),
        "event_check_errors": checks, "fixed_reservation_errors": fixed_errors,
        "input_hash": fingerprint(cultures), "failure": failure,
    }
    if checks or fixed_errors:
        output["outcome"] = "INVALID_OUTPUT"
        output["primary_accepted"] = output["supplementary_accepted"] = False
    with gzip.open(directory / "events.json.gz", "wt") as stream:
        json.dump(events, stream)
    with gzip.open(directory / "solver_calls.json.gz", "wt") as stream:
        json.dump(handler.rows, stream)
    write_json(directory / "result.json", output)
    write_json(directory / "progress.json", {
        "pid": os.getpid(), "stage": "recorded", "outcome": output["outcome"],
        "wall_seconds": output["wall_seconds"], "events": output["events"],
    })
    return output


def environment():
    root = Path(__file__).resolve().parents[1]
    files = [root / "scripts" / name for name in [
        "culture_capacity.py", "measure_culture_capacity.py",
        "culture_capacity_config.json", "calibrate_round3_growth.py"]]
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in
                     ["numpy", "scipy", "pandas", "polars", "ortools", "matplotlib", "pyarrow"]},
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in files},
        "solver_threads": 1, "solver_seed": 0,
        "biology": "30 independent repetitions per condition; paired prefixes across count conditions",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name(
        "culture_capacity_config.json"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", nargs=3, type=int, action="append",
                        help="HEK iPS repetition; pilot/debug only")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--wall-budget-seconds", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model = json.loads(args.model.read_text())
    cases = args.case or [(h, i, r) for h in config["hek_counts"]
                         for i in config["ips_counts"] for r in range(config["repetitions"])]
    if len({tuple(case) for case in cases}) != len(cases):
        raise ValueError("Duplicate run cases")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a fresh output directory or explicitly --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "model_hash": fingerprint(model),
                "config_hash": fingerprint(config), "cases": cases,
                "environment": environment(),
                "wall_budget_seconds": args.wall_budget_seconds,
                "scope": "pilot" if args.case else "main"}
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        for key in ("model_hash", "config_hash", "cases"):
            if fingerprint(previous[key]) != fingerprint(metadata[key]):
                raise ValueError(f"Resume {key} mismatch")
        if previous["environment"]["source_hashes"] != metadata["environment"]["source_hashes"]:
            raise ValueError("Resume implementation source hash mismatch")
    else:
        write_json(metadata_path, metadata)
    pending = []
    for case in cases:
        h, i, r = case
        directory = args.output / f"h{h:02d}_i{i:02d}_r{r:02d}"
        if not (directory / "result.json").exists():
            if directory.exists():
                raise ValueError(f"Incomplete directory requires inspection before resume: {directory}")
            pending.append(case)
    started = time.perf_counter()
    workers = args.workers or config["process_workers"]
    write_json(args.output / "process.json", {"pid": os.getpid(), "workers": workers,
                                            "started_epoch_seconds": time.time()})
    completed = len(cases)-len(pending)
    # At most workers submissions outstanding, so a total budget can censor
    # undispatched work without quietly executing an already-queued full grid.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        iterator = iter(pending)
        active = {}
        for _ in range(workers):
            case = next(iterator, None)
            if case is not None:
                future = pool.submit(run_trial, {"case": case, "config": config,
                                     "model": model, "output": str(args.output)})
                active[future] = case
        while active:
            future = next(as_completed(active))
            case = active.pop(future)
            result = future.result()
            completed += 1
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            if completed % 10 == 0 or result["outcome"] != "COMPLETED":
                print(f"{completed}/{len(cases)} case={case} outcome={result['outcome']} "
                      f"accepted={result['primary_accepted']} wall={result['wall_seconds']:.2f}s",
                      flush=True)
            if time.perf_counter()-started < args.wall_budget_seconds:
                following = next(iterator, None)
                if following is not None:
                    future = pool.submit(run_trial, {"case": following, "config": config,
                                         "model": model, "output": str(args.output)})
                    active[future] = following
    write_json(args.output / "execution.json", {
        "requested_cases": len(cases), "recorded_cases": completed,
        "all_cases_recorded": completed == len(cases),
        "wall_seconds": time.perf_counter()-started, "workers": workers,
        "wall_budget_seconds": args.wall_budget_seconds,
    })
    print(f"Recorded {completed}/{len(cases)} outcomes", flush=True)


if __name__ == "__main__":
    main()
