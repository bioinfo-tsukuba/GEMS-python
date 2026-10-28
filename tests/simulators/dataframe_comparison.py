"""Paired virtual experiments exercising DataFrame reads and live OR-Tools."""

import contextlib
from dataclasses import dataclass
import io
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile

from gems_python.common import dataframes as frames
from gems_python.common.cp_sat_scheduler import LOGGER
from gems_python.common.class_dumper import recursive_to_dict
from gems_python.one_machine_problem_simple_task import transition_manager as simple
from gems_python.one_machine_problem_simple_task.task_info import OneMachineTaskLocalInformation
from gems_python.one_machine_problem_simple_task.penalty.penalty_class import LinearPenalty as SimplePenalty
from gems_python.one_machine_problem_interval_task import transition_manager as single
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import LinearPenalty as SinglePenalty
from gems_python.multi_machine_problem_interval_task import transition_manager as multi
from gems_python.multi_machine_problem_interval_task.task_info import Machine, MachineList, Task, TaskGroup
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from scripts.compare_schedulers import evaluate
from tests.simulators import cell_culture, reaction_search


@dataclass
class SimpleState(simple.State):
    def task_generator(self, df):
        start = 0 if not len(df) else int(frames.last_value(df, "time")) + 3
        return OneMachineTaskLocalInformation(start, 2, SimplePenalty(1), "measure")

    def transition_function(self, df):
        return "SimpleState"


class SingleState(single.State):
    def task_generator(self, df):
        start = 0 if not len(df) else int(frames.last_value(df, "time")) + 3
        return single.TaskGroup(
            optimal_start_time=start, penalty_type=SinglePenalty(1),
            tasks=[single.Task(2, "prepare", interval=1), single.Task(1, "measure", interval=2)],
        )

    def transition_function(self, df):
        return "SingleState"

    def dummy_output(self, df, task_group_id, task_id):
        # Deliberately use the other backend: the dummy_output path must convert.
        return frames.make_frame(
            {"value": [len(df) + 1], "task_group_id": [task_group_id], "task_id": [task_id],
             "time": [0 if not len(df) else int(frames.last_value(df, "time")) + 6]},
            backend="polars" if frames.backend_of(df) == "pandas" else "pandas",
        )


class TimedState(multi.State):
    def __init__(self, duration=20, machine_type=0, first_start=0):
        self.duration = duration
        self.machine_type = machine_type
        self.first_start = first_start

    def task_generator(self, df):
        start = self.first_start if not len(df) else int(frames.last_value(df, "time")) + 100
        return TaskGroup(
            optimal_start_time=start, penalty_type=LinearPenalty(1),
            tasks=[Task(self.duration, "work", self.machine_type)],
        )

    def transition_function(self, df):
        return "TimedState"

    def to_dict(self):
        return {
            "_class": f"{type(self).__module__}.{type(self).__name__}",
            "duration": self.duration, "machine_type": self.machine_type,
            "first_start": self.first_start,
        }


class SolverTrace(logging.Handler):
    def __init__(self):
        super().__init__()
        self.calls = []

    def emit(self, record):
        match = re.search(r"status=(\w+) penalty=(\S+) objective=(\S+) best_bound=(\S+)", record.getMessage())
        if match:
            self.calls.append({
                "status": match[1], "penalty": float(match[2]),
                "objective": float(match[3]), "best_bound": float(match[4]),
            })


def cases():
    return [
        {"name": "cell_culture", "steps": 16},
        {"name": "reaction_grid", "steps": 27, "strategy": "grid", "seed": 0},
        *({"name": f"reaction_random_{seed}", "steps": 27, "strategy": "random", "seed": seed}
          for seed in (0, 7, 42)),
        {"name": "concurrent", "steps": 2},
        {"name": "single_interval", "steps": 12},
        {"name": "single_dummy", "steps": 12},
        {"name": "simple", "steps": 12},
    ]


def _build(case, directory, backend):
    name = case["name"]
    if name == "cell_culture":
        return cell_culture.build_experiments(directory, backend), cell_culture.observe, "multi_interval"
    if name.startswith("reaction_"):
        candidates = reaction_search.candidate_order(case["strategy"], case["seed"])
        return (reaction_search.build_experiments(directory, candidates, backend),
                reaction_search.ReactionObserver(candidates), "multi_interval")
    if name == "concurrent":
        experiments = [
            multi.Experiment("slow", [TimedState()], "TimedState", experiment_uuid="slow",
                             dataframe_backend=backend),
            multi.Experiment("fast", [TimedState(1, 1, 1)], "TimedState", experiment_uuid="fast",
                             dataframe_backend=backend),
        ]
        manager = multi.Experiments(
            experiments, directory, machine_list=MachineList([
                Machine(0, machine_id="slow_machine"), Machine(1, machine_id="fast_machine"),
            ]),
        )
        return manager, _observe, "multi_interval"
    if name.startswith("single_"):
        experiment = single.Experiment(
            "single", [SingleState()], "SingleState",
            experiment_uuid="single", dataframe_backend=backend,
        )
        return single.Experiments([experiment], directory), _observe, "single_interval"
    experiment = simple.Experiment(
        "simple", [SimpleState()], "SimpleState",
        experiment_uuid="simple", dataframe_backend=backend,
    )
    return simple.Experiments([experiment], directory), None, "simple"


def _observe(state, history, group, task, minute):
    return frames.make_frame({
        "time": [minute], "value": [len(history) + 1],
        "task_group_id": [group.task_group_id], "task_id": [task.task_id],
    }, frames.backend_of(history))


def run_case(case, backend):
    trace = SolverTrace()
    old_level = LOGGER.level
    LOGGER.addHandler(trace)
    LOGGER.setLevel(logging.INFO)
    snapshots, events = [], []
    try:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            manager, observer, variant = _build(case, Path(directory), backend)
            inputs = {
                "case": case,
                "experiments": [
                    {"name": e.experiment_name, "uuid": e.experiment_uuid,
                     "state": e.current_state_name,
                     "columns": list(e.shared_variable_history.columns),
                     "rows": frames.records(e.shared_variable_history)}
                    for e in manager.experiments
                ],
                "tasks": [recursive_to_dict(t) for t in
                          (manager.tasks if variant == "simple" else manager.task_groups)],
                "machines": recursive_to_dict(manager.machine_list) if variant == "multi_interval" else None,
            }
            input_sha256 = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
            manager.execute_scheduling()
            for _ in range(case["steps"]):
                snapshots.append(evaluate(manager, variant))
                if variant == "simple":
                    task = min(manager.tasks, key=lambda t: (t.scheduled_time, t.task_id))
                    minute = task.scheduled_time + task.processing_time
                    experiment = next(e for e in manager.experiments if e.experiment_uuid == task.experiment_uuid)
                    old_state = experiment.current_state_name
                    manager.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
                        task.task_id,
                        frames.make_frame({"time": [minute], "value": [len(experiment.shared_variable_history) + 1]}, backend),
                        scheduling_method="o", optimal_time_reference_time=minute,
                    )
                    result = {"status": "ok", "experiment_name": experiment.experiment_name,
                              "from_state": old_state, "to_state": experiment.current_state_name,
                              "simulated_time": minute, "simulated_task_id": task.task_id}
                else:
                    result = manager.simulate_one(
                        scheduling_method="o",
                        observation_provider=None if case["name"] == "single_dummy" else observer,
                    )
                if result.get("status") != "ok":
                    raise RuntimeError(f"{case['name']} ({backend}) failed: {result}")
                events.append(result)
            snapshots.append(evaluate(manager, variant))
            histories = {
                e.experiment_name: {"columns": list(e.shared_variable_history.columns),
                                    "rows": frames.records(e.shared_variable_history)}
                for e in manager.experiments
            }
        if not trace.calls or any(call["status"] != "OPTIMAL" for call in trace.calls):
            raise AssertionError(f"Small acceptance cases must be proven optimal: {trace.calls}")
        if any(s["overlap_minutes"] or s["assignment_errors"] or s["interval_errors"] for s in snapshots):
            raise AssertionError(f"Invalid schedule in {case['name']} ({backend})")
        return {"input_sha256": input_sha256, "events": events, "histories": histories,
                "schedules": snapshots, "solver": trace.calls}
    finally:
        LOGGER.removeHandler(trace)
        LOGGER.setLevel(old_level)


def assert_same(case, pandas_run, polars_run):
    for key in ("input_sha256", "events", "histories", "schedules", "solver"):
        if pandas_run[key] != polars_run[key]:
            raise AssertionError(f"Backend mismatch in {case['name']}: {key}")
