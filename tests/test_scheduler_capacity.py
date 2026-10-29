"""Regression checks for physical assignments, limits and finite updates."""

import copy
from contextlib import nullcontext
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ortools.sat.python import cp_model

from gems_python.common import dataframes
from gems_python.common.cp_sat_scheduler import schedule_task_groups
from gems_python.common.scheduling import (
    LOGGER, assign_machine_ids, validate_groups,
)
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task, TaskGroup, TaskGroupStatus, TaskStatus,
)
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    State, Experiment, Experiments,
)
from scripts.compare_schedulers import make_case, run_case


class FiniteState(State):
    def task_generator(self, df):
        return TaskGroup(
            int(df["time"].iloc[-1]) if len(df) else 0, LinearPenalty(1),
            [] if len(df) >= 4 else [Task(3, "first", 0), Task(1, "second", 0, interval=1)],
        )

    def transition_function(self, df):
        return self.state_name

    def dummy_output(self, df, task_group_id, task_id):
        return dataframes.make_frame({"time": [len(df) + 1]})


def observe(state, history, group, task, event_time):
    return dataframes.make_frame({"time": [event_time]})


class Reports(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        if hasattr(record, "scheduling_report"):
            self.rows.append(copy.deepcopy(record.scheduling_report))


class TestSchedulerCapacity(unittest.TestCase):
    def test_original_overlap_reproduction_now_valid(self):
        for method, seed in [("f", 0), ("s", 0), ("s", 1), ("s", 2), ("o", 0)]:
            with self.subTest(method=method, seed=seed):
                result = run_case("multi_interval", "LinearPenalty", method, seed)
                self.assertEqual(result["overlap_minutes"], 0)
                self.assertEqual(result["assignment_errors"], 0)
                self.assertEqual(result["interval_errors"], 0)

    def test_small_sequential_counts_and_zero_duration(self):
        for count in range(1, 5):
            for capacity in (1, 2):
                groups = [TaskGroup(5 + i, LinearPenalty(1), [
                    Task(2, "a", 0), Task(0, "zero", 0), Task(1, "b", 0, interval=1)
                ]) for i in range(count)]
                machines = MachineList([Machine(0, f"m{i}") for i in range(capacity)])
                result = TaskGroup.schedule_task_groups(groups, machines, 5)
                validate_groups(result, 5, machines)

    def test_fixed_reservations_and_simultaneous_end_start(self):
        machines = MachineList([Machine(0, "m0"), Machine(0, "m1")])
        fixed = TaskGroup(0, LinearPenalty(1), [
            Task(10, "fixed", 0, task_status=TaskStatus.IN_PROGRESS,
                 allocated_machine_id="m0", scheduled_time=0),
            Task(2, "fixed_future", 0, allocated_machine_id="m1", scheduled_time=10),
        ], status=TaskGroupStatus.IN_PROGRESS)
        for method in ("f", "s", "o"):
            groups = [copy.deepcopy(fixed), TaskGroup(5, LinearPenalty(1), [Task(5, "new", 0)])]
            if method == "o":
                result = schedule_task_groups(groups, 5, machines)
            elif method == "f":
                result = TaskGroup.schedule_task_groups(groups, machines, 5)
            else:
                # Exercise the real finalization with a controlled annealing result.
                with patch("simanneal.Annealer.anneal", lambda self: (self.state, 0)):
                    result = TaskGroup.schedule_task_groups_simulated_annealing(groups, machines, 5)
            self.assertEqual([(t.scheduled_time, t.allocated_machine_id) for t in result[0].tasks],
                             [(0, "m0"), (10, "m1")])
            validate_groups(result, 5, machines)
        touching = [TaskGroup(0, LinearPenalty(1), [Task(5, "a", 0)]),
                    TaskGroup(5, LinearPenalty(1), [Task(2, "b", 0)])]
        result = TaskGroup.schedule_task_groups(touching, MachineList([Machine(0, "m0")]), 0)
        self.assertEqual([g.tasks[0].scheduled_time for g in result], [0, 5])

    def test_invalid_assignments_and_annealing_fallback(self):
        machines = MachineList([Machine(0, "m0")])
        with self.assertRaises(ValueError):
            TaskGroup.schedule_task_groups([TaskGroup(0, LinearPenalty(1), [Task(1, "a", 1)])],
                                           machines, 0)
        def invalid_result(annealer):
            for group in annealer.state:
                group.schedule_tasks(0)
            return annealer.state, 0
        groups = [TaskGroup(0, LinearPenalty(1), [Task(5, "a", 0)]),
                  TaskGroup(0, LinearPenalty(1), [Task(5, "b", 0)])]
        with patch("simanneal.Annealer.anneal", invalid_result), self.assertLogs(
                "gems_python.multi_machine_problem_interval_task.task_info", level="WARNING"):
            result = TaskGroup.schedule_task_groups_simulated_annealing(groups, machines, 0)
        validate_groups(result, 0, machines)
        self.assertEqual(sorted(g.tasks[0].scheduled_time for g in result), [0, 5])
        # Both assigner and validator must preserve/reject conflicting fixed IDs.
        for g in result:
            g.status = TaskGroupStatus.IN_PROGRESS
            g.tasks[0].scheduled_time = 0
        with self.assertRaises(ValueError):
            assign_machine_ids(result, machines)
        with self.assertRaises(ValueError):
            schedule_task_groups(result, 0, machines)

    def test_time_limit_settings_override_and_legacy_restore_all_variants(self):
        for variant in ("simple", "single_interval", "multi_interval"):
            with tempfile.TemporaryDirectory() as directory:
                manager = make_case(variant, "LinearPenalty", Path(directory))
                self.assertEqual(manager.solver_time_limit_seconds, 5.0)
                manager.solver_time_limit_seconds = 2.5
                limits = []
                original = cp_model.CpSolver.solve
                def solve(solver, model, *args, **kwargs):
                    limits.append(solver.parameters.max_time_in_seconds)
                    return original(solver, model, *args, **kwargs)
                with patch.object(cp_model.CpSolver, "solve", solve):
                    manager.execute_scheduling()
                    manager.execute_scheduling(solver_time_limit_seconds=1)
                self.assertEqual(limits, [2.5, 1.0])
                self.assertEqual(manager.solver_time_limit_seconds, 2.5)
                for restored in (type(manager).from_json(manager.to_json()),
                                 type(manager).from_pickle(dumped=manager.to_pickle())):
                    self.assertEqual(restored.solver_time_limit_seconds, 2.5)
                del manager.solver_time_limit_seconds
                saved = manager.to_dict()
                saved["attributes"].pop("solver_time_limit_seconds", None)
                self.assertEqual(type(manager).from_dict(saved).solver_time_limit_seconds, 5.0)
                manager.solver_time_limit_seconds = 2.5
                for bad in (True, False, 0, -1, float("nan"), float("inf"), "1"):
                    before = [(t.scheduled_time, getattr(t, "allocated_machine_id", None))
                              for t in (manager.tasks if variant == "simple" else
                                        [t for g in manager.task_groups for t in g.tasks])]
                    with self.assertRaises(ValueError):
                        manager.execute_scheduling(solver_time_limit_seconds=bad)
                    after = [(t.scheduled_time, getattr(t, "allocated_machine_id", None))
                             for t in (manager.tasks if variant == "simple" else
                                       [t for g in manager.task_groups for t in g.tasks])]
                    self.assertEqual(before, after)

    def test_unknown_is_recorded_and_tasks_rolled_back(self):
        groups = [TaskGroup(0, LinearPenalty(1), [Task(2, "a", 0)])]
        groups[0].tasks[0].scheduled_time = 99
        groups[0].tasks[0].allocated_machine_id = "original"
        handler = Reports()
        old_level = LOGGER.level
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        try:
            with patch.object(cp_model.CpSolver, "solve", return_value=cp_model.UNKNOWN), \
                 patch.object(cp_model.CpSolver, "wall_time", property(lambda s: 0.0)), \
                 patch.object(cp_model.CpSolver, "objective_value", property(lambda s: 0.0)), \
                 patch.object(cp_model.CpSolver, "best_objective_bound", property(lambda s: 0.0)):
                with self.assertRaisesRegex(RuntimeError, "UNKNOWN"):
                    schedule_task_groups(groups, 0, MachineList([Machine(0, "m0")]),
                                         solver_time_limit_seconds=60)
            self.assertEqual(handler.rows[-1]["status"], "UNKNOWN")
            self.assertFalse(handler.rows[-1]["valid"])
            self.assertEqual(handler.rows[-1]["solver_time_limit_seconds"], 60)
            self.assertEqual((groups[0].tasks[0].scheduled_time, groups[0].tasks[0].allocated_machine_id),
                             (99, "original"))
        finally:
            LOGGER.removeHandler(handler)
            LOGGER.setLevel(old_level)

    def test_finite_update_limits_and_no_internal_legacy_scheduler(self):
        for method in ("o", "f", "s"):
            with tempfile.TemporaryDirectory() as directory:
                manager = Experiments([
                    Experiment("finite", [FiniteState()], "FiniteState", experiment_uuid="finite")
                ], Path(directory), MachineList([Machine(0, "m0")]),
                    solver_time_limit_seconds=2)
                limits = []
                original = cp_model.CpSolver.solve
                def solve(solver, model, *args, **kwargs):
                    limits.append(solver.parameters.max_time_in_seconds)
                    return original(solver, model, *args, **kwargs)
                with patch.object(cp_model.CpSolver, "solve", solve), \
                     patch("simanneal.Annealer.anneal", lambda self: (self.state, 0)), \
                     (patch.object(TaskGroup, "schedule_task_groups",
                                   side_effect=AssertionError("Internal f call"))
                      if method == "o" else nullcontext()):
                    results = manager.simulate(8, scheduling_method=method,
                                               observation_provider=observe,
                                               solver_time_limit_seconds=1)
                self.assertEqual([r["status"] for r in results], ["ok"] * 4 + ["no_tasks"])
                self.assertEqual(manager.task_groups, [])
                self.assertEqual(manager.solver_time_limit_seconds, 2)
                if method == "o":
                    self.assertTrue(limits)
                    self.assertEqual(set(limits), {1.0})

    def test_simultaneous_completions_preserve_the_other_task(self):
        machines = MachineList([Machine(0, "m0"), Machine(0, "m1")])
        with tempfile.TemporaryDirectory() as directory:
            manager = Experiments([
                Experiment(name, [FiniteState()], "FiniteState", experiment_uuid=name)
                for name in ("first", "second")
            ], Path(directory), machines)
            manager.execute_scheduling()
            before = [(g.tasks[0].scheduled_time, g.tasks[0].allocated_machine_id)
                      for g in manager.task_groups]
            manager.simulate_one(observation_provider=observe)
            # Both first tasks finish at 3; the second completion must remain at 3.
            other = manager.task_groups[1].tasks[0]
            self.assertEqual((other.scheduled_time, other.allocated_machine_id), before[1])
            self.assertEqual(other.task_status, TaskStatus.IN_PROGRESS)
            result = manager.simulate_one(observation_provider=observe)
            self.assertEqual(result["simulated_time"], 3)
            remaining = manager.simulate(10, observation_provider=observe)
            self.assertEqual([r["status"] for r in remaining], ["ok"] * 6 + ["no_tasks"])

