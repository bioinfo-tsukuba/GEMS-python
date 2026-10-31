"""Measure received experiment result -> updated schedule JSON, without preplanning."""

import argparse
import contextlib
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems_python.common import dataframes
from gems_python.common.scheduling import LOGGER, validate_groups
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from scripts.analyze_scheduler_capacity import describe
from scripts.measure_scheduler_capacity import (
    Reports, classify, dimensions, environment, fingerprint, snapshot,
)


ARCHIVE_COMMIT = "bf457334a0bddc3a75ff6560726a80e3c42dbf80"
ARCHIVE_DIRECTORY = "docs/benchmarks/scheduler_capacity"
PATTERN_SIZES = {
    "contention": 256, "heterogeneous": 512, "many_candidates": 128,
    "mixed_penalties": 128, "replanning": 256,
}


def result_event(manager):
    """Prepare a received result using the first completion of the retained plan."""
    group, task = min(
        ((group, task) for group in manager.task_groups for task in group.tasks
         if not task.completed),
        key=lambda pair: (pair[1].scheduled_time + pair[1].processing_time,
                          pair[1].scheduled_time, pair[0].task_group_id, pair[1].task_id),
    )
    experiment = next(e for e in manager.experiments if e.experiment_uuid == group.experiment_uuid)
    return {
        "task_group_id": group.task_group_id, "task_id": task.task_id,
        "experiment_uuid": group.experiment_uuid,
        "event_time": max(manager.reference_time, task.scheduled_time + task.processing_time),
        "value": len(experiment.shared_variable_history) + 1,
        "completes_group": all(t.completed or t.task_id == task.task_id for t in group.tasks),
    }


def freeze_begun_work(manager, event_time):
    """Apply progress observed when the result arrived, as simulate_one does."""
    for group in manager.task_groups:
        if group.is_not_started():
            for task in group.tasks:
                if not task.completed and task.scheduled_time <= event_time:
                    task.task_status = type(task.task_status).IN_PROGRESS
            group.status_update()
    manager.reference_time = event_time


def schedule_payload(manager):
    return {
        "reference_time": manager.reference_time,
        "tasks": [
            {
                "experiment_uuid": group.experiment_uuid,
                "task_group_id": group.task_group_id, "task_id": task.task_id,
                "allocated_machine_id": task.allocated_machine_id,
                "scheduled_time": task.scheduled_time,
                "processing_time": task.processing_time,
                "task_status": task.task_status.value,
            }
            for group in manager.task_groups for task in group.tasks
        ],
    }


def measure_response(manager, event, limit):
    # The incoming result is already a DataFrame when the application receives it.
    received = dataframes.make_frame({"time": [event["event_time"]], "value": [event["value"]]})
    handler, old_level = Reports(), LOGGER.level
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    output_log = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(output_log), contextlib.redirect_stderr(output_log):
            start = time.perf_counter()
            freeze_begun_work(manager, event["event_time"])
            after_progress = time.perf_counter()
            try:
                manager.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
                    task_group_id=event["task_group_id"], task_id=event["task_id"],
                    new_result_of_experiment=received, update_type="a", scheduling_method="o",
                    optimal_time_reference_time=event["event_time"],
                    solver_time_limit_seconds=limit,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                output_log.write(traceback.format_exc())
            after_update = time.perf_counter()
            outcome = classify(handler.rows, error)
            response = json.dumps(
                schedule_payload(manager) if outcome == "VALID"
                else {"outcome": outcome, "error": error},
                separators=(",", ":"), ensure_ascii=False,
            )
            finish = time.perf_counter()
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(old_level)
    row = {
        "outcome": outcome, "error": error,
        "response_seconds": finish - start,
        "progress_seconds": after_progress - start,
        "update_api_seconds": after_update - after_progress,
        "json_seconds": finish - after_update,
        "response_bytes": len(response.encode()), "solver_calls": len(handler.rows),
        "reports": handler.rows, "event": event,
    }
    row["search_seconds"] = sum(r["search_seconds"] for r in handler.rows)
    row["non_search_seconds"] = row["response_seconds"] - row["search_seconds"]
    # Independent checks occur after the response timer, outside the measured API.
    if outcome == "VALID":
        validate_groups(manager.task_groups, manager.reference_time, manager.machine_list)
        actual_experiment = next(
            e for e in manager.experiments if e.experiment_uuid == event["experiment_uuid"])
        assert len(actual_experiment.shared_variable_history) == event["value"]
        assert json.loads(response) == schedule_payload(manager)
    assert len(handler.rows) == 1, f"Expected one scheduling call; got {len(handler.rows)}"
    assert all(r["solver_time_limit_seconds"] == limit for r in handler.rows)
    return row, response, output_log.getvalue()


def worker(request_path):
    request = json.loads(request_path.read_text())
    manager = Experiments.from_dict(request["input"])
    assert fingerprint(snapshot(manager)) == request["input_hash"]
    manager.parent_dir_path = request_path.parent / "workspace"
    manager.parent_dir_path.mkdir(exist_ok=True)
    row, response, log = measure_response(manager, request["event"], request["search_limit_seconds"])
    row.update({key: request[key] for key in
                ("pattern", "size", "repetition", "search_limit_seconds", "input_hash")})
    row["response_sha256"] = hashlib.sha256(response.encode()).hexdigest()
    (request_path.parent / "response.json").write_text(response)
    (request_path.parent / "update.log").write_text(log)
    (request_path.parent / "result.json").write_text(json.dumps(row, indent=2))


def retained_inputs(commit, output):
    raw = subprocess.run(
        ["git", "show", f"{commit}:{ARCHIVE_DIRECTORY}/raw_artifacts.tar.gz"],
        capture_output=True, check=True,
    ).stdout
    csv_text = subprocess.run(
        ["git", "show", f"{commit}:{ARCHIVE_DIRECTORY}/trial_metrics.csv"],
        capture_output=True, text=True, check=True,
    ).stdout
    trials = list(csv.DictReader(io.StringIO(csv_text)))
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        names = archive.getnames()
        for pattern, size in PATTERN_SIZES.items():
            trial = next(row for row in trials if row["phase"] == "scan"
                         and row["pattern"] == pattern and int(row["size"]) == size
                         and float(row["search_limit_seconds"]) == 1
                         and row["outcome"] == "VALID")
            suffix = f"/{trial['artifact_directory']}/output.json"
            source = next(name for name in names if name.endswith(suffix))
            value = json.load(archive.extractfile(source))
            manager = Experiments.from_dict(value)
            validate_groups(manager.task_groups, manager.reference_time, manager.machine_list)
            typed = snapshot(manager)
            result[pattern] = {
                "input": typed, "input_hash": fingerprint(typed),
                "event": result_event(manager), "size": size,
                "source_output": source, "source_input_hash": trial["input_hash"],
                "dimensions": dimensions(manager),
            }
    (output / "inputs.json").write_text(json.dumps(result, indent=2))
    return result


def summarize(output):
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    summary = []
    lines = [
        "# Time from experiment result receipt to schedule JSON generation",
        "",
        "Timing starts when a received DataFrame is passed to the application and covers fixing started work, updating history, completion processing, "
        "required state transitions and task generation, CP-SAT replanning, and schedule JSON generation.",
        "Process startup, input preparation, hardware communication, rendering and file saving are excluded.",
        "No replanning occurs before result receipt. CP-SAT runs once per response.",
        "",
        "The first result event for the previously saved valid plan is replayed. "
        "Each condition has three repetitions with identical input and seed. These are descriptive response-time statistics, not a measurement of eight continued updates.",
        "For conditions whose first event is not at the end of a group, state-transition and next-group-generation branches do not execute.",
        "",
        "| Workload | Scale | Search limit (seconds) | Valid schedules/trials | Successful response median / p95 / maximum (seconds) | Failed response median (seconds) |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for pattern in PATTERN_SIZES:
        for limit in [1.0, 60.0]:
            selected = [r for r in rows if r["pattern"] == pattern
                        and r["search_limit_seconds"] == limit]
            if not selected:
                continue
            valid = [r for r in selected if r["outcome"] == "VALID"]
            failures = [r for r in selected if r["outcome"] != "VALID"]
            stats = describe(r["response_seconds"] for r in valid)
            failed_stats = describe(r["response_seconds"] for r in failures)
            item = {
                "pattern": pattern, "size": selected[0]["size"], "search_limit_seconds": limit,
                "runs": len(selected), "valid": len(valid), "response_seconds": stats,
                "failed_response_seconds": failed_stats,
                "all_attempts_response_seconds": describe(r["response_seconds"] for r in selected),
                "search_seconds": describe(r["search_seconds"] for r in selected),
                "non_search_seconds": describe(r["non_search_seconds"] for r in selected),
                "event": selected[0]["event"],
            }
            summary.append(item)
            text = (" / ".join(f"{stats[k]:.3f}" for k in ["median", "p95", "max"])
                    if valid else "—")
            failed = f"{failed_stats['median']:.3f}" if failures else "—"
            lines.append(f"| {pattern} | {item['size']} | {limit:g} | {len(valid)}/{len(selected)} | "
                         f"{text} | {failed} |")
    lines += [
        "",
        "Failed responses do not count as successful schedule returns. UNKNOWN means no solution was found; "
        "it does not prove infeasibility. The p95 is linearly interpolated from three points and does not guarantee future performance.",
        "JSON includes all task IDs, machine IDs, start times, processing times, progress and reference time. "
        "Formatting debug output from the return API is included in timing.",
        "",
        "Inputs, events, all search records, successful/failed JSON and environment information are retained. "
        "Operational times from the previous evaluation include search before result receipt and are not pooled into this table.",
        "",
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "README.md").write_text("\n".join(lines))
    return rows, summary


def run(output, archive_commit, repetitions):
    output.mkdir(parents=True, exist_ok=False)
    env = environment()
    env["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    env["source_archive_commit"] = archive_commit
    env["repetitions"] = repetitions
    env["boundary"] = "received DataFrame -> complete schedule JSON in memory"
    (output / "environment.json").write_text(json.dumps(env, indent=2))
    inputs = retained_inputs(archive_commit, output)
    for repetition in range(repetitions):
        for limit in [1.0, 60.0]:
            for pattern, base in inputs.items():
                directory = output / f"{pattern}-{limit:g}-{repetition}"
                directory.mkdir()
                request = {
                    "pattern": pattern, "size": base["size"], "repetition": repetition,
                    "search_limit_seconds": limit, "input_hash": base["input_hash"],
                    "input": base["input"], "event": base["event"],
                }
                request_path = directory / "request.json"
                request_path.write_text(json.dumps(request, indent=2))
                with (directory / "worker.log").open("w") as log:
                    subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
                        stdout=log, stderr=subprocess.STDOUT, timeout=limit + 120, check=True,
                    )
                row = json.loads((directory / "result.json").read_text())
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(f"{pattern} / {limit:g} s / repeat {repetition}: "
                      f"{row['outcome']}, response={row['response_seconds']:.6f} s", flush=True)
                summarize(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive-commit", default=ARCHIVE_COMMIT)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.output and args.repetitions > 0:
        run(args.output, args.archive_commit, args.repetitions)
    else:
        parser.error("Specify --output and a positive repetition count, or --worker")
