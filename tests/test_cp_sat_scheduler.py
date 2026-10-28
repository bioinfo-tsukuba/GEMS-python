"""Behavioral checks for the OR-Tools default scheduler."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ortools.sat.python import cp_model

from gems_python.common.cp_sat_scheduler import (
    _cost, schedule_simple_tasks, schedule_task_groups,
)
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import (
    CyclicalRestPenaltyWithLinear as MultiCycle,
    LinearPenalty as MultiLinear,
)
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task as MultiTask, TaskGroup as MultiGroup,
    TaskGroupStatus, TaskStatus,
)
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    Experiments as MultiExperiments,
)
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import (
    CyclicalRestPenalty as SingleCycle,
    LinearPenalty as SingleLinear,
)
from gems_python.one_machine_problem_interval_task.task_info import (
    Task as SingleTask, TaskGroup as SingleGroup,
)
from gems_python.one_machine_problem_interval_task.transition_manager import (
    Experiments as SingleExperiments,
)
from gems_python.one_machine_problem_simple_task.penalty.penalty_class import (
    CyclicalRestPenaltyWithLinear as SimpleCycle,
    LinearPenalty as SimpleLinear,
)
from gems_python.one_machine_problem_simple_task.task_info import OneMachineTask
from gems_python.one_machine_problem_simple_task.transition_manager import (
    Experiments as SimpleExperiments,
)


class TestDefaultScheduler(unittest.TestCase):
    def test_defaults_and_simple_task_optimum(self):
        for experiment_type in (
            SimpleExperiments, SingleExperiments, MultiExperiments
        ):
            self.assertEqual(
                inspect.signature(experiment_type.execute_scheduling)
                .parameters["scheduling_method"].default,
                "o",
            )
        tasks = [
            OneMachineTask(0, 5, SimpleLinear(1), "a", "a", "a"),
            OneMachineTask(0, 5, SimpleLinear(1), "b", "b", "b"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            experiments = SimpleExperiments([], Path(directory))
            experiments.tasks = tasks
            experiments.execute_scheduling()
        times = sorted(task.scheduled_time for task in tasks)
        self.assertEqual(times, [0, 5])
        self.assertEqual(sum(abs(task.scheduled_time) for task in tasks), 5)
        self.assertEqual(
            min(
                a + b for a in range(11) for b in range(11)
                if abs(a - b) >= 5
            ),
            5,
        )

    def test_explicit_annealing_option_remains_available(self):
        task = OneMachineTask(0, 1, SimpleLinear(1), "a", "a", "a")
        task.scheduled_time = 0
        with tempfile.TemporaryDirectory() as directory:
            experiments = SimpleExperiments([], Path(directory))
            experiments.tasks = [task]
            with patch.object(
                OneMachineTask, "simulated_annealing_schedule", return_value=[task]
            ) as anneal:
                experiments.execute_scheduling("s")
                anneal.assert_called_once()

    def test_simple_cycle_uses_absolute_minutes(self):
        task = OneMachineTask(
            100, 1, SimpleCycle(100, 10, [(0, 2)], 1),
            "a", "a", "a",
        )
        schedule_simple_tasks([task], 100)
        self.assertEqual(task.scheduled_time, 103)

    def test_single_machine_intervals_and_group_start_penalty(self):
        groups = [
            SingleGroup(
                optimal_start_time=0,
                penalty_type=SingleLinear(1),
                tasks=[
                    SingleTask(3, "a", interval=2),
                    SingleTask(4, "b", interval=5),
                ],
            ),
            SingleGroup(
                optimal_start_time=0,
                penalty_type=SingleLinear(1),
                tasks=[SingleTask(5, "c", interval=0)],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            experiments = SingleExperiments(
                experiments=[], parent_dir_path=Path(directory), task_groups=groups
            )
            experiments.execute_scheduling()
        a, b = groups[0].tasks
        self.assertEqual(b.scheduled_time, a.scheduled_time + 3 + 5)
        intervals = [
            (task.scheduled_time, task.scheduled_time + task.processing_time)
            for group in groups for task in group.tasks
        ]
        for index, (start, end) in enumerate(intervals):
            for other_start, other_end in intervals[index + 1:]:
                self.assertTrue(end <= other_start or other_end <= start)
        self.assertEqual(
            SingleGroup.eval_schedule_penalty(groups),
            sum(
                abs(group.tasks[0].scheduled_time - group.tasks[0].interval)
                for group in groups
            ),
        )

    def test_multi_machine_assignment_and_fixed_task(self):
        machines = MachineList([
            Machine(0, machine_id="m0"), Machine(0, machine_id="m1")
        ])
        groups = [
            MultiGroup(0, MultiLinear(1), [MultiTask(10, "a", 0)]),
            MultiGroup(0, MultiLinear(1), [MultiTask(10, "b", 0)]),
        ]
        schedule_task_groups(groups, 0, machines)
        self.assertEqual([group.tasks[0].scheduled_time for group in groups], [0, 0])
        self.assertEqual(
            {group.tasks[0].allocated_machine_id for group in groups},
            {"m0", "m1"},
        )
        fixed = MultiGroup(
            0, MultiLinear(1),
            [MultiTask(
                10, "fixed", 0, task_status=TaskStatus.IN_PROGRESS,
                allocated_machine_id="m0", scheduled_time=0,
            )],
            status=TaskGroupStatus.IN_PROGRESS,
        )
        movable = MultiGroup(0, MultiLinear(1), [MultiTask(10, "new", 0)])
        schedule_task_groups(
            [fixed, movable], 0, MachineList([Machine(0, machine_id="m0")])
        )
        self.assertEqual(fixed.tasks[0].scheduled_time, 0)
        self.assertEqual(fixed.tasks[0].allocated_machine_id, "m0")
        self.assertGreaterEqual(movable.tasks[0].scheduled_time, 10)

    def test_multi_machine_default_and_missing_resource(self):
        group = MultiGroup(0, MultiLinear(1), [MultiTask(3, "a", 0)])
        with tempfile.TemporaryDirectory() as directory:
            experiments = MultiExperiments(
                experiments=[], parent_dir_path=Path(directory),
                task_groups=[group],
                machine_list=MachineList([Machine(0, machine_id="m0")]),
            )
            experiments.execute_scheduling()
        self.assertEqual(group.tasks[0].scheduled_time, 0)
        self.assertEqual(group.tasks[0].allocated_machine_id, "m0")
        with self.assertRaisesRegex(ValueError, "No machine of type 2"):
            schedule_task_groups(
                [MultiGroup(0, MultiLinear(1), [MultiTask(3, "bad", 2)])],
                0, MachineList([Machine(0, machine_id="m0")]),
            )

    def test_infeasible_fixed_assignments_report_error(self):
        groups = [
            MultiGroup(
                0, MultiLinear(1),
                [MultiTask(
                    10, "fixed", 0, task_status=TaskStatus.IN_PROGRESS,
                    allocated_machine_id="m0", scheduled_time=0,
                )],
                status=TaskGroupStatus.IN_PROGRESS,
            )
            for _ in range(2)
        ]
        groups.append(MultiGroup(0, MultiLinear(1), [MultiTask(1, "new", 0)]))
        with self.assertRaisesRegex(RuntimeError, "INFEASIBLE"):
            schedule_task_groups(
                groups, 0, MachineList([Machine(0, machine_id="m0")])
            )

    def test_cycle_cost_matches_existing_penalty_at_boundaries(self):
        for penalty in (
            SingleCycle(100, 10, [(0, 2), (9, 10)]),
            MultiCycle(100, 10, [(0, 2), (9, 10)], 3),
        ):
            for minute in (99, 100, 102, 103, 109, 110, 111):
                with self.subTest(penalty=type(penalty).__name__, minute=minute):
                    model = cp_model.CpModel()
                    start = model.new_int_var(minute, minute, "start")
                    cost, _ = _cost(
                        model, start, minute, minute, 105, penalty, "cycle"
                    )
                    solver = cp_model.CpSolver()
                    self.assertEqual(solver.solve(model), cp_model.OPTIMAL)
                    self.assertEqual(
                        solver.value(cost),
                        penalty.calculate_penalty(minute, 105),
                    )

    def test_invalid_method_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            experiments = SimpleExperiments([], Path(directory))
            experiments.tasks = [
                OneMachineTask(0, 1, SimpleLinear(1), "a", "a", "a")
            ]
            with self.assertRaisesRegex(ValueError, "Unexpected scheduling_method"):
                experiments.execute_scheduling("unknown")


if __name__ == "__main__":
    unittest.main()
