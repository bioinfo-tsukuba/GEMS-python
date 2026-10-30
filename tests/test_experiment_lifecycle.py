import copy
import tempfile
import unittest
from unittest.mock import patch

from gems_python.common import dataframes as frames
from gems_python.common.planning_errors import PlanningError
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, Experiments
from gems_python.multi_machine_problem_interval_task.task_info import Machine, MachineList
from tests.lifecycle_fixtures import (
    manager, child_specs, advance, observe, FiniteState, EmptyState, LegacyEmptyState,
)


class TestExperimentLifecycle(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_self_delete_only_after_explicit_completed_protocol(self):
        lab = manager(self.directory.name)
        advance(lab)
        self.assertEqual(len(lab.experiments), 1)
        advance(lab)
        self.assertEqual(lab.experiments, [])
        self.assertEqual(lab.task_groups, [])
        ending = lab.ended_experiments["parent"]
        self.assertEqual(len(ending.history), 2)
        self.assertEqual(ending.reason, "protocol complete")
        self.assertEqual(lab.simulate_one()["status"], "no_tasks")
        self.assertIs(lab.request_finish("parent", ending.request_id, ending.reason), ending)
        with self.assertRaises(ValueError):
            lab.request_finish("parent", ending.request_id, "different")

    def test_empty_group_does_not_imply_termination(self):
        for cls, expected in ((EmptyState, 0), (LegacyEmptyState, 1)):
            lab = Experiments([Experiment(cls.__name__, [cls()], cls.__name__)],
                              self.directory.name)
            lab.execute_scheduling()
            self.assertEqual(len(lab.experiments), expected)
            self.assertEqual(lab.task_groups, [])

    def test_unfinished_tasks_block_finish_and_split_confirmation(self):
        lab = manager(self.directory.name, split=True)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            lab.request_finish("parent", "end", "done")
        record = lab.request_split("parent", "split", child_specs())
        with self.assertRaisesRegex(ValueError, "before parent tasks finish"):
            lab.confirm_split(record.request_id)
        self.assertEqual(len(lab.experiments), 1)
        self.assertFalse(lab.ended_experiments)

    def test_split_disabled_and_conflicting_requests(self):
        lab = manager(self.directory.name)
        with self.assertRaisesRegex(ValueError, "disabled"):
            lab.request_split("parent", "split", child_specs())
        lab.enable_split = True
        record = lab.request_split("parent", "split", child_specs())
        self.assertIs(record, lab.request_split("parent", "split", child_specs()))
        other = child_specs()
        other[0].initial_conditions["volume"] = 6
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            lab.request_split("parent", "split", other)
        lab.confirm_split("split", False, reason="operation failed")
        self.assertEqual(record.status, "FAILED")
        self.assertEqual(len(lab.experiments), 1)
        self.assertFalse(lab.ended_experiments)

    def test_history_order_and_independent_children_both_backends(self):
        for backend in ("pandas", "polars"):
            with self.subTest(backend=backend):
                lab = manager(self.directory.name, backend, True)
                advance(lab)
                lab.simulate_one(observation_provider=observe)
                event = next(iter(lab.split_requests.values()))
                self.assertEqual(event.status, "PENDING")
                self.assertEqual(len(lab.experiments), 1)
                lab.confirm_split(event.request_id)
                a, b = lab.experiments
                self.assertEqual(frames.convert(a.history_for_state(), "polars")["value"].to_list(),
                                 [1, 2, 0])
                parent_before = lab.ended_experiments["parent"].history
                sibling_before = b.to_json()
                a.shared_variable_history = frames.append(a.shared_variable_history,
                                                         frames.make_frame({"time": [3], "value": [1]}, backend))
                self.assertEqual(b.to_json(), sibling_before)
                self.assertEqual(frames.serialize(parent_before),
                                 frames.serialize(lab.ended_experiments["parent"].history))
                copied = a.inherited_history
                if backend == "pandas":
                    copied.iloc[0, 1] = 999
                self.assertEqual(frames.last_value(a.inherited_history, "value"), 2)
                self.assertIs(lab.confirm_split(event.request_id), event)
                self.assertEqual(len(lab.experiments), 2)
                self.assertNotEqual(a.experiment_uuid, b.experiment_uuid)
                self.assertIsNot(a.states[0], b.states[0])
                self.assertEqual(a.task_group_sequence, 1)
                self.assertEqual(a.parent_uuid, "parent")

    def test_invalid_second_child_and_commit_fault_leave_no_partial_registration(self):
        for injected in ("spec", "commit"):
            lab = manager(self.directory.name, split=True)
            advance(lab)
            lab.simulate_one(observation_provider=observe)
            event = next(iter(lab.split_requests.values()))
            if injected == "spec":
                event.children[1].states[0].duration = -1
                with self.assertRaises(PlanningError):
                    lab.confirm_split(event.request_id)
            else:
                def broken(experiments, groups, endings, record):
                    lab.experiments = experiments[:1]
                    raise RuntimeError("injected mid-commit fault")
                with patch.object(lab, "_commit_split", broken), self.assertRaises(PlanningError):
                    lab.confirm_split(event.request_id)
            self.assertEqual([e.experiment_uuid for e in lab.experiments], ["parent"])
            self.assertEqual(event.status, "PENDING")
            self.assertFalse(lab.ended_experiments)
            self.assertEqual(lab.manager_status, "STOPPED")

    def test_split_with_missing_machine_retains_all_children_and_runs_other_child(self):
        lab = manager(self.directory.name, split=True)
        advance(lab)
        lab.simulate_one(observation_provider=observe)
        previous = next(iter(lab.split_requests.values()))
        lab.confirm_split(previous.request_id, False)
        specs = child_specs(time=lab.reference_time)
        specs[1].states[0].machine_type = 99
        event = lab.request_split("parent", "missing-machine", specs)
        with self.assertLogs(level="WARNING"):
            lab.confirm_split(event.request_id)
        self.assertEqual(len(lab.experiments), 2)
        self.assertEqual([e.lifecycle_status for e in lab.experiments],
                         ["ACTIVE", "WAITING_MACHINE"])
        lab.simulate_one(observation_provider=observe)
        self.assertEqual(lab.experiments[1].shared_variable_history.shape[0], 1)

    def test_resend_last_result_after_parent_deleted_and_id_renumbered(self):
        lab = manager(self.directory.name)
        lab.execute_scheduling()
        advance(lab)
        group = lab.task_groups[0]
        result = frames.make_frame({"time": [2], "value": [2]})
        lab.record_task_result(group.task_group_uuid, 0, "final-result", result)
        before = lab.to_json()
        lab.record_task_result(group.task_group_uuid, 0, "final-result", result)
        self.assertEqual(lab.to_json(), before)
        with self.assertRaisesRegex(ValueError, "Conflicting result"):
            lab.record_task_result(group.task_group_uuid, 0, "final-result",
                                   frames.make_frame({"time": [2], "value": [99]}))

    def test_other_experiment_and_started_reservation_survive_parent_deletion(self):
        lab = manager(self.directory.name)
        lab.machine_list.add_machine(Machine(1, "other-machine"))
        other = Experiment("other", [FiniteState(limit=100, duration=10, machine_type=1)], "FiniteState",
                           experiment_uuid="other")
        lab.add_experiment(other, save_step=False)
        lab.start_task(task_group_uuid=lab.task_groups[1].task_group_uuid, task_id=0)
        fixed = lab.task_groups[1]
        before = (fixed.tasks[0].scheduled_time, fixed.tasks[0].allocated_machine_id)
        advance(lab)
        advance(lab)
        self.assertIn("parent", lab.ended_experiments)
        self.assertEqual((fixed.tasks[0].scheduled_time, fixed.tasks[0].allocated_machine_id), before)
        self.assertEqual(len(other.shared_variable_history), 0)
