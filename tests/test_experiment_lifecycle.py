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
        self.assertEqual(ending.settings["protocol_states"][0].limit, 2)
        self.assertEqual(ending.settings["max_unknown_retries"], 3)
        settings_copy = ending.settings
        settings_copy["protocol_states"][0].limit = 99
        self.assertEqual(ending.settings["protocol_states"][0].limit, 2)
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

    def test_planning_stop_after_split_commit_retains_all_children_and_lineage(self):
        from gems_python.common.planning_errors import SolverFailure
        for status in ("UNKNOWN", "INFEASIBLE"):
            with self.subTest(status=status):
                lab = manager(self.directory.name, split=True)
                lab.unknown_policy = "stop"
                advance(lab)
                lab.simulate_one(observation_provider=observe)
                event = next(iter(lab.split_requests.values()))
                with patch("gems_python.common.cp_sat_scheduler.schedule_task_groups",
                           side_effect=SolverFailure(status, {"status": status})):
                    if status == "UNKNOWN":
                        lab.confirm_split(event.request_id)
                    else:
                        with self.assertRaises(PlanningError):
                            lab.confirm_split(event.request_id)
                self.assertEqual(event.status, "COMMITTED")
                self.assertEqual(lab.manager_status, "STOPPED")
                self.assertEqual({e.experiment_uuid for e in lab.experiments},
                                 set(event.child_uuids))
                self.assertEqual(set(lab.ended_experiments), {"parent"})
                before = lab.to_json()
                self.assertIs(lab.confirm_split(event.request_id), event)
                self.assertEqual(lab.to_json(), before)
                for restored in (Experiments.from_json(before),
                                 Experiments.from_pickle(dumped=lab.to_pickle())):
                    self.assertEqual(restored.manager_status, "STOPPED")
                    self.assertEqual({e.experiment_uuid for e in restored.experiments},
                                     set(event.child_uuids))

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
        lab.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
            None, 0, result, task_group_uuid=group.task_group_uuid,
            result_id="final-result", optimal_time_reference_time=999)
        self.assertEqual(lab.to_json(), before)

    def test_file_result_resend_after_deletion_does_not_advance_step_or_clock(self):
        import json
        from pathlib import Path
        for backend in ("pandas", "polars"):
            with self.subTest(backend=backend):
                lab = manager(Path(self.directory.name) / backend, backend)
                advance(lab)
                group = lab.task_groups[0]
                directory = lab.save_dir()
                directory.mkdir(parents=True, exist_ok=True)
                csv = directory / "result.csv"
                frames.write_csv(frames.make_frame({"time": [2], "value": [2]}, backend), csv)
                response = dict(task_response="Completed", task_group_uuid=group.task_group_uuid,
                                task_id=0, result_id="final-file", result_path=str(csv),
                                schedule_reference_time=2)
                (directory / "experiment_result.json").write_text(json.dumps(response))
                with patch.object(lab, "proceed_to_next_step") as proceed:
                    lab.auto_load()
                    self.assertEqual(proceed.call_count, 1)
                self.assertFalse(lab.experiments)
                response["schedule_reference_time"] = 999
                (directory / "experiment_result.json").write_text(json.dumps(response))
                before = lab.to_json()
                with patch.object(lab, "proceed_to_next_step") as proceed:
                    lab.auto_load()
                    proceed.assert_not_called()
                self.assertEqual(lab.to_json(), before)

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

    def test_invalid_child_task_rejected_before_request_is_accepted(self):
        lab = manager(self.directory.name, split=True)
        specs = child_specs()
        specs[1].states[0].duration = -1
        with self.assertRaisesRegex(PlanningError, "processing_time"):
            lab.request_split("parent", "bad", specs)
        self.assertEqual(len(lab.experiments), 1)
        self.assertFalse(lab.split_requests)
        self.assertFalse(lab.ended_experiments)

    def test_second_generation_inherits_each_observation_once(self):
        lab = manager(self.directory.name, split=True)
        advance(lab)
        advance(lab)
        first = lab.experiments[0]
        parent_values = frames.convert(first.history_for_state(), "polars")["value"].to_list()
        for group in lab.task_groups:
            if group.experiment_uuid == first.experiment_uuid:
                for task in group.tasks:
                    task.task_status = type(task.task_status).COMPLETED
                group.status_update()
        request = lab.request_split(first.experiment_uuid, "second-generation", child_specs(time=2))
        lab.confirm_split(request.request_id)
        for uuid in request.child_uuids:
            child = lab._experiment(uuid)
            self.assertEqual(frames.convert(child.inherited_history, "polars")["value"].to_list(),
                             parent_values)
            self.assertEqual(len(child.history_for_state()), len(parent_values) + 1)

    def test_finite_culture_model_and_explicit_aliquot_initial_conditions(self):
        from tests.simulators.lifecycle_cell_culture import build, run
        from tests.simulators.cell_culture import logistic_growth
        for backend in ("pandas", "polars"):
            lab = build(self.directory.name, backend)
            logs = run(lab)
            self.assertEqual(logs[-1]["status"], "no_tasks")
            self.assertEqual(len(lab.ended_experiments), 3)
            event = next(iter(lab.split_requests.values()))
            self.assertEqual(event.status, "COMMITTED")
            for i, uuid in enumerate(event.child_uuids):
                record = lab.ended_experiments[uuid]
                history = frames.convert(record.history, "polars")
                own = history.filter(history["generation"] == 1)
                self.assertEqual(len(own), 3)
                self.assertEqual(own["confluence"][0], (0.12, 0.18)[i])
                expected = logistic_growth(own["confluence"][-2],
                                          (own["time"][-1] - own["time"][-2]) / 60)
                self.assertAlmostEqual(own["confluence"][-1], expected)
                self.assertEqual(record.initial_conditions["volume"], 5.0)
                self.assertEqual(record.settings["protocol_states"][0].generation, 1)
