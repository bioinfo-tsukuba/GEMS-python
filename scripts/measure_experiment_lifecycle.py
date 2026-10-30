"""Issue #24: measure split preparation and total planning, with preserved paired inputs.

Planning timers include manager input checks, model, solve, output checks and
application. Search limits remain 1/60s, retry is disabled; a valid late output
is never counted as an on-time success. Preconditioning observations are
synthetic completed inputs, not timings of biological execution.
"""
import argparse
import ast
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems_python.common import dataframes as frames
from gems_python.common.persistence import save_snapshot
from gems_python.common.scheduling import LOGGER, validate_groups
from gems_python.multi_machine_problem_interval_task.lifecycle import ChildSpec, fingerprint
from gems_python.multi_machine_problem_interval_task.transition_manager import State, Experiment, Experiments
from gems_python.multi_machine_problem_interval_task.task_info import (
    TaskGroup, TaskStatus, TaskGroupStatus, Machine, MachineList,
)
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from scripts.measure_scheduler_capacity import CapacityState, Reports, dimensions, quality

DEFAULT_CONFIG = Path(__file__).with_name("experiment_lifecycle_capacity_config.json")
SCHEDULER_CONFIG = Path(__file__).with_name("scheduler_capacity_config.json")


class ParentState(State):
    def task_generator(self, history):
        return TaskGroup(0, LinearPenalty(1), [])

    def transition_function(self, history):
        return self.state_name


class ChildState(CapacityState):
    def task_generator(self, history):
        # This static capacity fixture never ends due to inherited row count.
        return super().task_generator(history)


def prepare(job, directory):
    scheduler = json.loads(SCHEDULER_CONFIG.read_text())
    scheduler["group_cycles"] = 1000000
    machines = MachineList([
        Machine(kind, f"type{kind}_m{i}")
        for kind, count in enumerate(scheduler["machine_counts"][job["pattern"]])
        for i in range(count)
    ])
    parents = [Experiment(f"parent-{i}", [ParentState()], "ParentState",
                          experiment_uuid=f"lifecycle-{job['pattern']}-{i}",
                          shared_variable_history=frames.make_frame({
                              "time": list(range(job["group_interval"] * 3)),
                              "value": [i] * (job["group_interval"] * 3)}))
               for i in range(job["parents"])]
    lab = Experiments(parents, directory, machines, enable_split=True, unknown_policy="stop")
    preparation_started = time.perf_counter()
    initial_generation_started = preparation_started
    for generation in range(job["generations"]):
        lab.reference_time = generation * 100 + job["group_interval"] * 3
        current = list(lab.experiments)
        for i, parent in enumerate(current):
            if generation:
                extra = frames.make_frame({
                    "time": list(range(generation * 100, generation * 100 + job["group_interval"] * 3)),
                    "value": [i] * (job["group_interval"] * 3)})
                parent.shared_variable_history = frames.append(parent.shared_variable_history, extra)
                for group in lab.task_groups:
                    if group.experiment_uuid == parent.experiment_uuid:
                        for task in group.tasks:
                            task.task_status = TaskStatus.COMPLETED
                        group.status_update()
            index = i * job["children"]
            children = [
                ChildSpec(str(j), f"child-{generation}-{index+j}",
                          [ChildState(job["pattern"], index+j, scheduler)], "ChildState",
                          frames.make_frame({"time": [generation * 100 + job["group_interval"] * 3],
                                             "value": [0]}),
                          {"container_id": f"{generation}-{index+j}"},
                          {"container_id": "capacity-fixture:explicit"})
                for j in range(job["children"])
            ]
            request = f"split:{generation}:{parent.experiment_uuid}"
            lab.request_split(parent.experiment_uuid, request, children)
            lab.confirm_split(request, replan=False)
    preparation_seconds = time.perf_counter() - preparation_started
    if job["pattern"] == "replanning":
        size = len(lab.experiments)
        base_time = lab.reference_time
        lab.reference_time = base_time + 21 + size * 20
        for i, group in enumerate(lab.task_groups):
            if i % 4:
                group.optimal_start_time = lab.reference_time + i % 4
                continue
            group.tasks[1].interval += size * 20
            group.schedule_tasks(base_time + 10 + i * 20)
            group.tasks[0].task_status = TaskStatus.COMPLETED
            group.status = TaskGroupStatus.IN_PROGRESS
            for task in group.tasks:
                task.allocated_machine_id = "type0_m0"
                if task.scheduled_time <= lab.reference_time < task.scheduled_time + task.processing_time:
                    task.task_status = TaskStatus.IN_PROGRESS
            child = lab._experiment(group.experiment_uuid)
            child.shared_variable_history = frames.append(child.shared_variable_history, frames.make_frame({
                "time": [group.tasks[0].scheduled_time + group.tasks[0].processing_time], "value": [1]}))
    if job["mode"] == "flat":
        experiments = copy.deepcopy(lab.experiments)
        for old, new in zip(lab.experiments, experiments):
            new.shared_variable_history = old.history_for_state()
            new._inherited_history = None
            new.parent_uuid = None
        lab = Experiments(experiments, directory, copy.deepcopy(lab.machine_list),
                          task_groups=copy.deepcopy(lab.task_groups),
                          reference_time=lab.reference_time, unknown_policy="stop")
    lab.parent_dir_path = Path("/tmp/gems-issue24-capacity-replay")
    return lab, preparation_seconds, time.perf_counter() - initial_generation_started


def conditions(lab):
    return (lab.reference_time,
            [(m.machine_type, m.machine_id) for m in lab.machine_list.machines],
            [(g.optimal_start_time, g.penalty_type, g.status.name,
              [(t.processing_time, t.interval, t.optimal_machine_type, t.scheduled_time,
                t.allocated_machine_id, t.task_status.name) for t in g.tasks])
             for g in lab.task_groups])


def worker(job_file):
    job = json.loads(job_file.read_text())
    directory = job_file.parent
    reports = Reports()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(reports)
    result = dict(job)
    pipeline_started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lab, preparation_seconds, fixture_seconds = prepare(job, directory / "lab")
        result.update(preparation_seconds=preparation_seconds if job["mode"] == "lifecycle" else None,
                      fixture_seconds=fixture_seconds,
                      dimensions=dimensions(lab), input_hash=fingerprint(lab),
                      condition_hash=fingerprint(conditions(lab)),
                      input_json_bytes=len(lab.to_json().encode()))
        (directory / "input.json").write_text(lab.to_json())
        started = time.perf_counter()
        error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                outcome = lab.execute_scheduling(solver_time_limit_seconds=job["budget"])
        except Exception as err:
            error = f"{type(err).__name__}: {err}"
            outcome = {"status": "ERROR"}
        planning_seconds = time.perf_counter() - started
        result.update(planning_seconds=planning_seconds, reports=reports.rows,
                      error=error, manager_status=lab.manager_status)
        if outcome["status"] == "READY":
            original = Experiments.from_json((directory / "input.json").read_text())
            fixed = {id(new): (old.scheduled_time, old.allocated_machine_id)
                     for old_group, new_group in zip(original.task_groups, lab.task_groups)
                     if not old_group.is_not_started()
                     for old, new in zip(old_group.tasks, new_group.tasks)}
            validate_groups(lab.task_groups, lab.reference_time, lab.machine_list, fixed)
            result["outcome"] = "VALID_ON_TIME" if planning_seconds <= job["budget"] else "VALID_LATE"
            result.update(quality(lab))
        else:
            result["outcome"] = lab.stop_diagnostic["kind"] if lab.stop_diagnostic else "ERROR"
            (directory / "error.json").write_text(json.dumps(lab.stop_diagnostic or outcome))
        (directory / "output.json").write_text(lab.to_json())
        save_started = time.perf_counter()
        save_snapshot(lab, directory / "saved")
        result["save_seconds"] = time.perf_counter() - save_started
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["total_pipeline_seconds"] = time.perf_counter() - pipeline_started
        result["measured_phase_seconds"] = fixture_seconds + planning_seconds + result["save_seconds"]
        result["own_history_rows"] = sum(len(e.shared_variable_history) for e in lab.experiments)
        result["inherited_history_rows"] = sum(len(e.inherited_history) if e._inherited_history is not None
                                              else 0 for e in lab.experiments)
        result["archived_history_rows"] = sum(len(e.history) for e in lab.ended_experiments.values())
        result["archived_experiments"] = len(lab.ended_experiments)
    except Exception:
        result.update(outcome="WORKER_ERROR", error=traceback.format_exc())
        (directory / "error.txt").write_text(result["error"])
    (directory / "result.json").write_text(json.dumps(result, indent=2))


def jobs(config):
    for pattern in config["patterns"]:
        for budget in config["planning_budgets_seconds"]:
            for parents in config["parent_counts"]:
                yield dict(series="scan", pattern=pattern, budget=budget, parents=parents,
                           children=2, group_interval=2, generations=1)
    for pattern in config["patterns"]:
        for budget in config["planning_budgets_seconds"]:
            for children in config["variation_child_counts"]:
                yield dict(series="child_count", pattern=pattern, budget=budget,
                           parents=config["variation_parents"], children=children,
                           group_interval=2, generations=1)
            for interval in config["variation_group_intervals"]:
                yield dict(series="frequency", pattern=pattern, budget=budget,
                           parents=config["variation_parents"], children=2,
                           group_interval=interval, generations=1)
            yield dict(series="two_generations", pattern=pattern, budget=budget,
                       parents=config["variation_parents"], children=2,
                       group_interval=2, generations=2)


def measurement_signature(job):
    return tuple(job[k] for k in ("series", "pattern", "budget", "parents", "children",
                                  "group_interval", "generations", "mode", "repetition"))


def verify_resume_source(output):
    """Keep generation, timed work and dependencies fixed across driver recovery."""
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((output / "metadata.json").read_text())
    commit = metadata["commit"]
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit],
                                    cwd=root, text=True).splitlines()
    fixed = [f for f in files if f.startswith("gems_python/")] + [
        "scripts/measure_scheduler_capacity.py", "scripts/scheduler_capacity_config.json"]
    hashes = {}
    for name in fixed:
        original = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        current = (root / name).read_bytes()
        if original != current:
            raise ValueError(f"Measurement source changed: {name}")
        hashes[name] = hashlib.sha256(current).hexdigest()
    original = subprocess.check_output(
        ["git", "show", f"{commit}:scripts/measure_experiment_lifecycle.py"], cwd=root, text=True)
    names = {"ParentState", "ChildState", "prepare", "conditions", "worker", "jobs"}
    def fixed_ast(source):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(source).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names}
    expected = fixed_ast(original)
    actual = fixed_ast(Path(__file__).read_text())
    if actual != expected or set(actual) != names:
        raise ValueError("Input generation or timed measurement changed")
    versions = {p: version(p) for p in metadata["versions"]}
    if versions != metadata["versions"] or sys.version != metadata["python"]:
        raise ValueError("Measurement Python or dependency versions changed")
    environment = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    if set(environment) != set((output / "environment.txt").read_text().splitlines()):
        raise ValueError("Frozen measurement environment changed")
    return dict(source_commit=commit, source_hashes=hashes, versions=versions,
                timed_ast_sha256=hashlib.sha256(
                    json.dumps(actual, sort_keys=True).encode()).hexdigest(),
                driver_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())


def recover_measurement(output, config):
    if json.loads((output / "config.json").read_text()) != config:
        raise ValueError("Resume must use the original measurement config")
    source = verify_resume_source(output)
    ledger = output / "results.jsonl"
    if not ledger.exists():
        checkpoints = sorted(output.glob("checkpoint-*.jsonl"))
        if not checkpoints:
            raise ValueError("No preserved result ledger")
        ledger.write_bytes(checkpoints[-1].read_bytes())
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if len({measurement_signature(row) for row in rows}) != len(rows):
        raise ValueError("Duplicated trial in preserved ledger")
    for row in rows:
        artifact = output / row["artifact_directory"]
        recorded = json.loads((artifact / "result.json").read_text())
        if any(row[k] != v for k, v in recorded.items()):
            raise ValueError(f"Preserved trial does not match ledger: {artifact.name}")
        row.setdefault("storage_phase", "initial_tmpfs")
    metadata = json.loads((output / "metadata.json").read_text())
    budget = output / "study_budget.json"
    deadline = (datetime.fromisoformat(json.loads(budget.read_text())["deadline"])
                if budget.exists() else
                datetime.fromtimestamp(datetime.fromisoformat(metadata["started"]).timestamp()
                                       + config["total_runtime_seconds"], timezone.utc))
    index = len(list(output.glob("resume-*.json"))) + 1
    recovery = dict(source, started=datetime.now(timezone.utc).isoformat(),
                    deadline=deadline.isoformat(), preserved_trials=len(rows),
                    storage_phase="resume_workspace",
                    storage_note="Initial 54 trials used /tmp tmpfs; resumed trials use workspace filesystem")
    (output / f"resume-{index:02d}.json").write_text(json.dumps(recovery, indent=2))
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows, max(0, deadline.timestamp() - time.time())


def profile_initialization(job_file):
    """Supplemental cold initialization; never added to the original pipeline timer."""
    job = json.loads(job_file.read_text())
    expected = json.loads(job_file.with_name("result.json").read_text())
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        lab, _, fixture_seconds = prepare(job, job_file.parent / "initialization-lab")
    elapsed = time.perf_counter() - started
    assert fingerprint(lab) == expected["input_hash"], "Supplemental input differs from measured input"
    result = dict(initialization_seconds=elapsed-fixture_seconds,
                  fixture_seconds=fixture_seconds, total_generation_seconds=elapsed,
                  input_hash=expected["input_hash"], supplemental=True,
                  started=datetime.now(timezone.utc).isoformat(),
                  note="Fresh process, no solver; recorded separately from the measured trial")
    job_file.with_name("initialization_profile.json").write_text(json.dumps(result, indent=2))


def profile_all_initializations(output):
    source = verify_resume_source(output)
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    budget = json.loads((output / "study_budget.json").read_text())
    deadline = datetime.fromisoformat(budget["deadline"]).timestamp()
    completed, skipped, failures = [], [], []
    for row in rows:
        directory = output / row["artifact_directory"]
        if row["outcome"] in ("WORKER_ERROR", "WORKER_TIMEOUT_OR_ERROR"):
            skipped.append(directory.name)
            continue
        if (directory / "initialization_profile.json").exists():
            completed.append(directory.name)
            continue
        if time.time() >= deadline:
            skipped.append(directory.name)
            continue
        with (directory / "initialization_profile.log").open("w") as log:
            try:
                subprocess.run(
                    [sys.executable, "-m", "scripts.measure_experiment_lifecycle",
                     "--profile-initialization", str((directory / "job.json").resolve())],
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                    timeout=min(120, deadline-time.time()))
                completed.append(directory.name)
            except Exception as err:
                failures.append(dict(directory=directory.name, error=str(err)))
    summary = dict(source, completed=completed, skipped=skipped, failures=failures,
                   deadline=budget["deadline"], finished=datetime.now(timezone.utc).isoformat())
    (output / "initialization_profiles.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(profiles=len(completed), skipped=len(skipped), failures=failures)))
    return summary


def measure(config, output, resume=False):
    if config["seed"] != 0 or config["workers"] != 1 or config["dataframe_backend"] != "pandas":
        raise ValueError("This study uses seed=0, workers=1 and pandas; other settings need a separate study")
    if resume:
        rows, remaining = recover_measurement(output, config)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(config, indent=2))
        metadata = dict(started=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                    versions={p: version(p) for p in ("ortools", "pandas", "polars", "pyarrow")},
                    platform=platform.platform(), processor=platform.processor(),
                    timing="manager planning entry through validated application",
                    retry=False, notes="synthetic completed observations; not physical execution")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (output / "environment.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True))
        rows, remaining = [], config["total_runtime_seconds"]
    deadline = time.monotonic() + remaining
    skipped = []
    completed = {measurement_signature(row) for row in rows}
    existing = [int(p.name.split("-")[1]) for p in output.glob("trial-*") if p.is_dir()]
    next_trial = max(existing, default=-1) + 1

    def run_pair(base, repetition):
        nonlocal next_trial
        for mode in ("lifecycle", "flat"):
            job = dict(base, mode=mode, repetition=repetition)
            if measurement_signature(job) in completed:
                continue
            if time.monotonic() >= deadline:
                skipped.append(job)
                continue
            directory = output / f"trial-{next_trial:04d}"
            next_trial += 1
            directory.mkdir()
            job_file = directory / "job.json"
            job_file.write_text(json.dumps(job))
            with (directory / "worker.log").open("w") as log:
                try:
                    subprocess.run([sys.executable, "-m", "scripts.measure_experiment_lifecycle",
                                    "--worker", str(job_file.resolve())],
                                   stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=min(deadline-time.monotonic(), job["budget"] + 120))
                    row = json.loads((directory / "result.json").read_text())
                except Exception as err:
                    row = dict(job, outcome="WORKER_TIMEOUT_OR_ERROR", error=str(err))
            row["artifact_directory"] = directory.name
            row["storage_phase"] = "resume_workspace" if resume else "initial"
            rows.append(row)
            completed.add(measurement_signature(job))
            with (output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"{len(rows)} {base['pattern']} P={base['parents']} "
                  f"C={base['children']} {base['budget']}s {mode} {row['outcome']}", flush=True)

    for job in jobs(config):
        run_pair(job, 0)
    for pattern in config["patterns"]:
        for budget in config["planning_budgets_seconds"]:
            scan = [r for r in rows if r["series"] == "scan" and r["mode"] == "lifecycle"
                    and r["pattern"] == pattern and r["budget"] == budget]
            valid = [r for r in scan if r["outcome"] == "VALID_ON_TIME"]
            boundary = max((r["parents"] for r in valid), default=None)
            selected = [boundary] if boundary is not None else []
            failure = next((r["parents"] for r in scan if
                            (boundary is None or r["parents"] > boundary) and
                            r["outcome"] != "VALID_ON_TIME"), None)
            if failure is not None:
                selected.append(failure)
            for parents in selected:
                base = dict(series="boundary", pattern=pattern, budget=budget, parents=parents,
                            children=2, group_interval=2, generations=1)
                for repetition in range(1, config["boundary_repetitions"]):
                    run_pair(base, repetition)
    (output / "completion.json").write_text(json.dumps(dict(
        fully_measured=not skipped, skipped=skipped, trial_count=len(rows),
        finished=datetime.now(timezone.utc).isoformat()), indent=2))
    return rows


if __name__ == "__main__":
    # Persist importable class names, not __main__:Class.
    from scripts.measure_experiment_lifecycle import ParentState, ChildState
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--profile-initialization", type=Path)
    parser.add_argument("--profile-all-initializations", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.profile_initialization:
        profile_initialization(args.profile_initialization)
    elif args.profile_all_initializations:
        profile_all_initializations(args.profile_all_initializations)
    else:
        if args.output is None:
            parser.error("--output is required")
        measure(json.loads(args.config.read_text()), args.output, resume=args.resume)
