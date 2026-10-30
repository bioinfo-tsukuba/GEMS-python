import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gems_python.common import dataframes as frames
from gems_python.common.persistence import load_snapshot, save_snapshot
from gems_python.common.serialization import equivalent, SerializationError
from gems_python.common.planning_errors import SolverFailure
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from tests.lifecycle_fixtures import manager, advance, observe


class TestLifecyclePersistence(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_pending_committed_and_ended_resume_match_uninterrupted(self):
        for backend in ("pandas", "polars"):
            for checkpoint in range(7):
                with self.subTest(backend=backend, checkpoint=checkpoint), \
                        contextlib.redirect_stdout(io.StringIO()):
                    source = manager(self.directory.name, backend, True)
                    # checkpoint 2 deliberately retains an unconfirmed split.
                    for i in range(checkpoint):
                        if i == 1:
                            source.simulate_one(observation_provider=observe)
                        else:
                            advance(source)
                    resumed = [Experiments.from_json(source.to_json()),
                               Experiments.from_pickle(dumped=source.to_pickle())]
                    for restored in resumed:
                        self.assertTrue(equivalent(source, restored))
                    for _ in range(9):
                        expected = advance(source)
                        for restored in resumed:
                            self.assertEqual(expected, advance(restored))
                            self.assertTrue(equivalent(source, restored))

    def test_stopped_reload_does_not_release_stop(self):
        lab = manager(self.directory.name)
        lab.execute_scheduling()
        lab._stop("TEST", "injected")
        for restored in (Experiments.from_json(lab.to_json()),
                         Experiments.from_pickle(dumped=lab.to_pickle())):
            self.assertEqual(restored.manager_status, "STOPPED")
            self.assertEqual(restored.simulate_one()["status"], "stopped")
            restored.resume()
            self.assertEqual(restored.manager_status, "RUNNING")

    def test_retry_checkpoint_keeps_remaining_budget(self):
        from gems_python.common.cp_sat_scheduler import schedule_task_groups
        lab = manager(self.directory.name)
        saved, limits = [], []
        def solver(groups, reference, machines, **kwargs):
            limits.append(kwargs["solver_time_limit_seconds"])
            if len(limits) == 1:
                raise SolverFailure("UNKNOWN", {"status": "UNKNOWN"})
            saved.append(lab.to_json())
            return schedule_task_groups(groups, reference, machines, **kwargs)
        with patch("gems_python.common.cp_sat_scheduler.schedule_task_groups", solver):
            lab.execute_scheduling()
        restored = Experiments.from_json(saved[0])
        resumed_limits = []
        def remaining(groups, reference, machines, **kwargs):
            resumed_limits.append(kwargs["solver_time_limit_seconds"])
            raise SolverFailure("UNKNOWN", {"status": "UNKNOWN"})
        with patch("gems_python.common.cp_sat_scheduler.schedule_task_groups", remaining), \
                self.assertRaisesRegex(RuntimeError, "retry limit"):
            restored.execute_scheduling()
        self.assertEqual(resumed_limits, [10, 20, 40])
        self.assertEqual(restored.manager_status, "STOPPED")

    def test_same_generation_selected_after_write_faults(self):
        import gems_python.common.persistence as persistence
        directory = Path(self.directory.name)
        lab = manager(directory)
        lab.execute_scheduling()
        save_snapshot(lab, directory)
        before = lab.to_json()
        advance(lab)
        original = persistence._write_synced
        for failure in (1, 2, 3):
            calls = []
            def fail(path, content):
                calls.append(path)
                if len(calls) == failure:
                    raise OSError("injected write interruption")
                return original(path, content)
            with patch.object(persistence, "_write_synced", fail), self.assertRaises(OSError):
                save_snapshot(lab, directory)
            self.assertEqual(load_snapshot(directory, Experiments).to_json(), before)
        original_replace = persistence.os.replace
        def fail_pointer(source, destination):
            if Path(destination).name == "snapshot_manifest.json":
                raise OSError("injected manifest interruption")
            return original_replace(source, destination)
        with patch.object(persistence.os, "replace", fail_pointer), self.assertRaises(OSError):
            save_snapshot(lab, directory)
        self.assertEqual(load_snapshot(directory, Experiments).to_json(), before)

    def test_canonical_hash_validation_and_compatibility_exports(self):
        directory = Path(self.directory.name)
        lab = manager(directory)
        save_snapshot(lab, directory)
        manifest = json.loads((directory / "snapshot_manifest.json").read_text())
        generation = directory / ".snapshot_generations" / manifest["generation"]
        (directory / "experiments.json").write_text("broken compatibility export")
        self.assertTrue(equivalent(lab, load_snapshot(directory, Experiments)))
        (generation / "experiments.json").write_text("broken")
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(equivalent(lab, load_snapshot(directory, Experiments)))
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            load_snapshot(directory, Experiments, source="json")
        (generation / "experiments.pkl").write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "json:.*pickle:"):
            load_snapshot(directory, Experiments)

    def test_missing_new_progress_fields_fail_but_old_snapshot_migrates(self):
        lab = manager(self.directory.name)
        value = lab.to_dict()
        del value["attributes"]["max_unknown_retries"]
        with self.assertRaisesRegex(SerializationError, "missing saved fields"):
            Experiments.from_dict(value)
        value = lab.to_dict()
        from gems_python.common.serialization import _LIFECYCLE_FIELDS
        def old(item):
            if isinstance(item, dict):
                name = item.get("class", "").split(":")[-1]
                if item.get("kind") == "object" and name in _LIFECYCLE_FIELDS:
                    for key in _LIFECYCLE_FIELDS[name]:
                        item["attributes"].pop(key, None)
                for child in list(item.values()):
                    old(child)
            elif isinstance(item, list):
                for child in item:
                    old(child)
        old(value)
        restored = Experiments.from_dict(value)
        self.assertEqual(restored.manager_status, "RUNNING")
        self.assertFalse(restored.enable_split)
        self.assertEqual(restored.max_unknown_retries, 3)
        self.assertEqual(restored.experiments[0].experiment_uuid, "parent")
        restored.execute_scheduling()

    def test_restore_in_fresh_process_retains_pending_split_and_history(self):
        lab = manager(self.directory.name, "polars", True)
        advance(lab)
        lab.simulate_one(observation_provider=observe)
        path = Path(self.directory.name) / "pending.json"
        path.write_text(lab.to_json())
        code = """
import sys
from pathlib import Path
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from tests.lifecycle_fixtures import advance
lab = Experiments.from_json(Path(sys.argv[1]).read_text())
assert len(lab.experiments) == 1
event = next(iter(lab.split_requests.values()))
lab.confirm_split(event.request_id)
assert len(lab.experiments) == 2
assert all(len(e.inherited_history) == 2 for e in lab.experiments)
for _ in range(8): advance(lab)
assert not lab.experiments
assert len(lab.ended_experiments) == 3
"""
        subprocess.run([sys.executable, "-c", code, str(path)], check=True, capture_output=True)
