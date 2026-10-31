import tempfile
import unittest
from unittest.mock import patch

from gems_python.common.cp_sat_scheduler import schedule_task_groups
from gems_python.common.planning_errors import PlanningError, SolverFailure
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment
from gems_python.multi_machine_problem_interval_task.task_info import Machine, TaskStatus, Task, TaskGroup
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from tests.lifecycle_fixtures import manager, FiniteState, observe


class TestPlanningPolicy(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.lab = manager(self.directory.name)

    def test_machine_wait_is_local_and_registration_automatically_replans(self):
        other = Experiment("other", [FiniteState(machine_type=1)], "FiniteState",
                           experiment_uuid="other")
        with self.assertLogs(level="WARNING"):
            self.lab.add_experiment(other, save_step=False)
        self.assertEqual(other.lifecycle_status, "WAITING_MACHINE")
        self.assertEqual(self.lab.simulate_one(observation_provider=observe)["status"], "ok")
        with patch.object(self.lab, "proceed_to_next_step"):
            self.lab.add_machine(Machine(1, "imager"))
        self.assertEqual(other.lifecycle_status, "ACTIVE")
        self.assertEqual(other.experiment_uuid, "other")

    def test_registration_checks_later_tasks_in_first_group_before_start(self):
        group = TaskGroup(0, LinearPenalty(1), [Task(1, "available", 0), Task(1, "missing", 99)])
        other = Experiment("two-task", [FiniteState()], "FiniteState",
                           experiment_uuid="two-task", current_task_group=group)
        with self.assertLogs(level="WARNING"):
            self.lab.add_experiment(other, save_step=False)
        self.assertEqual(other.lifecycle_status, "WAITING_MACHINE")
        registered = self.lab._resolve_group(group_uuid=group.task_group_uuid)
        self.assertTrue(all(t.scheduled_time is None for t in registered.tasks))
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            self.lab.start_task(task_group_uuid=group.task_group_uuid, task_id=0)
        self.assertEqual(self.lab.simulate_one(observation_provider=observe)["status"], "ok")
        with patch.object(self.lab, "proceed_to_next_step"):
            self.lab.add_machine(Machine(99, "new-kind"))
        self.assertEqual(other.lifecycle_status, "ACTIVE")
        self.assertTrue(all(t.allocated_machine_id is not None for t in registered.tasks))

    def _unknown(self, limits, success_after=None):
        def solver(groups, reference, machines, **kwargs):
            limits.append(kwargs["solver_time_limit_seconds"])
            if success_after is not None and len(limits) > success_after:
                return schedule_task_groups(groups, reference, machines, **kwargs)
            raise SolverFailure("UNKNOWN", {"status": "UNKNOWN"})
        return patch("gems_python.common.cp_sat_scheduler.schedule_task_groups", solver)

    def test_unknown_doubles_and_keeps_registration_then_accepts_plan(self):
        limits = []
        with self._unknown(limits, 2), self.assertLogs(level="WARNING"):
            self.lab.execute_scheduling()
        self.assertEqual(limits, [5, 10, 20])
        self.assertEqual(len(self.lab.experiments), 1)
        self.assertEqual(self.lab.solver_time_limit_seconds, 5)
        self.assertEqual(self.lab.manager_status, "RUNNING")
        self.assertEqual(self.lab.experiments[0].lifecycle_status, "ACTIVE")
        self.assertEqual(len(self.lab.planning_progress.reports), 3)

    def test_retry_limit_and_configured_stop_warn_and_block_new_tasks(self):
        for policy, expected in (("retry", [5, 10, 20, 40]), ("stop", [5])):
            lab = manager(self.directory.name)
            limits = []
            lab.unknown_policy = policy
            original_lab, self.lab = self.lab, lab
            with self._unknown(limits), self.assertLogs(level="WARNING"):
                if policy == "retry":
                    with self.assertRaisesRegex(PlanningError, "retry limit"):
                        lab.execute_scheduling()
                else:
                    self.assertEqual(lab.execute_scheduling()["status"], "STOPPED")
            self.assertEqual(limits, expected)
            self.assertEqual(lab.manager_status, "STOPPED")
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                lab.start_task(task_group_uuid=lab.task_groups[0].task_group_uuid, task_id=0)
            self.assertEqual(lab.simulate_one()["status"], "stopped")
            lab.resume()
            self.assertEqual(lab.manager_status, "RUNNING")
            self.lab = original_lab

    def test_negative_duration_and_fixed_overlap_stop_with_input_details(self):
        self.lab.task_groups[0].tasks[0].processing_time = -1
        with self.assertRaisesRegex(PlanningError, "processing_time"):
            self.lab.execute_scheduling()
        self.assertEqual(self.lab.manager_status, "STOPPED")
        self.assertIn("input", self.lab.stop_diagnostic)
        self.lab.task_groups[0].tasks[0].processing_time = 1
        self.lab.resume()
        original = self.lab.task_groups[0]
        self.lab.start_task(task_group_uuid=original.task_group_uuid, task_id=0)
        e = Experiment("other", [FiniteState()], "FiniteState", experiment_uuid="other")
        self.lab.experiments.append(e)
        g = e.current_task_group
        g.tasks[0].scheduled_time = original.tasks[0].scheduled_time
        g.tasks[0].allocated_machine_id = "m0"
        g.tasks[0].task_status = TaskStatus.IN_PROGRESS
        g.status_update()
        self.lab.task_groups.append(g)
        with self.assertRaisesRegex(PlanningError, "Overlapping.*m0"):
            self.lab.execute_scheduling()
        self.assertEqual(self.lab.manager_status, "STOPPED")

    def test_infeasible_is_not_retried(self):
        with patch("gems_python.common.cp_sat_scheduler.schedule_task_groups",
                   side_effect=SolverFailure("INFEASIBLE", {"status": "INFEASIBLE"})) as solver, \
                self.assertRaisesRegex(PlanningError, "INFEASIBLE"):
            self.lab.execute_scheduling()
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(self.lab.manager_status, "STOPPED")
        self.assertIn("not identified", self.lab.stop_diagnostic["reason"])

    def test_stopped_accepts_in_progress_result_without_transition_until_resume(self):
        self.lab.execute_scheduling()
        g = self.lab.task_groups[0]
        self.lab.start_task(task_group_uuid=g.task_group_uuid, task_id=0)
        self.lab._stop("TEST", "injected stop")
        history = observe(None, self.lab.experiments[0].history_for_state(), g, g.tasks[0], 1)
        self.lab.record_task_result(g.task_group_uuid, 0, "received", history)
        self.assertTrue(g.is_completed())
        self.assertEqual(self.lab.task_groups, [g])
        self.assertEqual(len(self.lab.experiments[0].shared_variable_history), 1)
        self.lab.resume()
        self.assertNotEqual(self.lab.task_groups[0].task_group_uuid, g.task_group_uuid)

    def test_busy_machine_is_scheduled_in_future_not_held(self):
        self.lab.execute_scheduling()
        group = self.lab.task_groups[0]
        group.tasks[0].processing_time = 100
        self.lab.start_task(task_group_uuid=group.task_group_uuid, task_id=0)
        exp = Experiment("new", [FiniteState()], "FiniteState", experiment_uuid="new")
        self.lab.add_experiment(exp, save_step=False)
        self.assertEqual(exp.lifecycle_status, "ACTIVE")
        self.assertGreaterEqual(self.lab.task_groups[1].tasks[0].scheduled_time, 100)
