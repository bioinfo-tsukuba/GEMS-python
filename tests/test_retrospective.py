import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gems_python.common.persistence import load_snapshot, save_snapshot
from gems_python.common.serialization import equivalent
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from gems_python.visualization.retrospective import Renderer, review_data
from tests.lifecycle_fixtures import advance, manager, observe
from examples.experiment_retrospective import run


class TestRetrospective(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_completed_run_saved_reload_and_render_both_backends(self):
        for backend in ("pandas", "polars"):
            with self.subTest(backend=backend), contextlib.redirect_stdout(io.StringIO()):
                lab, events = run(self.root / backend, backend)
                self.assertEqual(events[-1]["status"], "no_tasks")
                self.assertFalse(lab.task_groups)
                save_snapshot(lab, self.root / backend / "snapshot")
                restored = load_snapshot(self.root / backend / "snapshot", Experiments)
                self.assertTrue(equivalent(lab, restored))
                data = review_data(restored)
                self.assertTrue(data["complete"])
                self.assertEqual(len(data["experiments"]), 8)
                self.assertEqual(len(data["tasks"]), 24)
                self.assertEqual(max(e["depth"] for e in data["experiments"]), 2)
                for record in data["experiments"]:
                    self.assertEqual(sum(t["experiment"] == record["id"] for t in data["tasks"]), 3)
                    if record["parent"]:
                        parent = next(e for e in data["experiments"] if e["id"] == record["parent"])
                        self.assertEqual(record["born"], parent["ended"])
                for machine in data["machines"]:
                    tasks = sorted([t for t in data["tasks"] if t["machine"] == machine["id"]],
                                   key=lambda t: t["start"])
                    self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(tasks, tasks[1:])))
                if backend == "pandas":
                    paths = restored.render_retrospective(self.root / "figures")
                    self.assertEqual(len(paths), 9)
                    page = (self.root / "figures" / "review.html").read_text()
                    self.assertNotIn('<script src=', page)
                    self.assertIn('id="lineage"', page)
                    self.assertIn('id="schedule"', page)

    def test_trace_idempotency_after_group_and_experiment_removed(self):
        lab = manager(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            lab.execute_scheduling()
            group = lab.task_groups[0]
            result = observe(None, lab.experiments[0].history_for_state(), group, group.tasks[0], 1)
            lab.update_shared_variable_history(group.task_group_id, 0, result,
                                               task_group_uuid=group.task_group_uuid, result_id="r")
            before = json.dumps(lab.retrospective_trace, sort_keys=True)
            lab.update_shared_variable_history(None, 0, result,
                                               task_group_uuid=group.task_group_uuid, result_id="r")
            self.assertEqual(before, json.dumps(lab.retrospective_trace, sort_keys=True))
            for _ in range(4):
                advance(lab)
        self.assertFalse(lab.experiments)
        self.assertEqual(len(lab.retrospective_trace["tasks"]), 2)
        self.assertEqual(review_data(lab)["experiments"][0]["name"], "parent")

    def test_split_failure_rolls_back_registration_and_retry_does_not_duplicate(self):
        lab = manager(self.root, split=True)
        with contextlib.redirect_stdout(io.StringIO()):
            lab.simulate_one(observation_provider=observe)
            lab.simulate_one(observation_provider=observe)
            event = next(iter(lab.split_requests.values()))
            before = json.dumps(lab.retrospective_trace, sort_keys=True)
            original = lab._commit_split
            def fail(*args):
                original(*args)
                raise RuntimeError("injected commit fault")
            with patch.object(lab, "_commit_split", side_effect=fail), self.assertRaises(RuntimeError):
                lab.confirm_split(event.request_id)
            self.assertEqual(before, json.dumps(lab.retrospective_trace, sort_keys=True))
            lab.resume()
            lab.confirm_split(event.request_id)
            after = json.dumps(lab.retrospective_trace, sort_keys=True)
            lab.confirm_split(event.request_id)
            self.assertEqual(after, json.dumps(lab.retrospective_trace, sort_keys=True))
            self.assertEqual(len(lab.retrospective_trace["experiments"]), 3)

    def test_old_snapshot_does_not_invent_completed_tasks(self):
        lab = manager(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(4):
                advance(lab)
        del lab.retrospective_trace
        for restored in (Experiments.from_json(lab.to_json()),
                         Experiments.from_pickle(dumped=lab.to_pickle())):
            self.assertFalse(restored.retrospective_trace["complete"])
            with self.assertWarnsRegex(RuntimeWarning, "Incomplete"):
                data = review_data(restored)
            self.assertFalse(data["complete"])
            self.assertEqual(data["tasks"], [])
            self.assertIsNone(data["experiments"][0]["born"])

    def test_unfinished_run_rejected_and_unknown_reservation_not_invented(self):
        lab = manager(self.root)
        with self.assertRaisesRegex(ValueError, "Finish all"):
            review_data(lab)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(4):
                advance(lab)
        task = next(iter(lab.retrospective_trace["tasks"].values()))
        task["start"] = None
        with self.assertWarnsRegex(RuntimeWarning, "Incomplete"):
            data = review_data(lab)
        self.assertEqual(data["omitted_tasks"], 1)
        self.assertEqual(len(data["tasks"]), 1)

    def test_names_and_uuid_are_generic_and_cross_page_ancestors_survive(self):
        with contextlib.redirect_stdout(io.StringIO()):
            lab, _ = run(self.root)
        for registration in lab.retrospective_trace["experiments"].values():
            registration["name"] = "同名 <sample>"
        data = review_data(lab)
        labels = [e["label"] for e in data["experiments"]]
        self.assertEqual(len(set(labels)), 8)
        self.assertTrue(all(label.startswith("同名 <sample> #") for label in labels))
        renderer = Renderer(data, None, "s")
        grandchild = next(e for e in data["experiments"] if e["depth"] == 2)
        context = renderer.with_ancestors([grandchild])
        self.assertEqual([e["depth"] for e in context], [0, 1, 2])
        self.assertEqual(context[-1]["id"], grandchild["id"])
        self.assertEqual(len(data["tasks"]), 24)

    def test_registration_uses_manager_clock_on_construction_and_add(self):
        from tests.lifecycle_fixtures import EmptyState
        from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment
        lab = Experiments([], self.root, reference_time=37)
        exp = Experiment("empty", [EmptyState()], "EmptyState", experiment_uuid="uuid-with-dashes")
        with contextlib.redirect_stdout(io.StringIO()):
            lab.add_experiment(exp, save_step=False)
        self.assertFalse(lab.experiments)
        record = review_data(lab)["experiments"][0]
        self.assertEqual(record["born"], 37)
        self.assertEqual(record["ended"], 37)
        self.assertEqual(record["id"], "uuid-with-dashes")


if __name__ == "__main__":
    unittest.main()
