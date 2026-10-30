import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from gems_python.common.scheduling import validate_groups
from gems_python.common.persistence import save_snapshot
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from gems_python.multi_machine_problem_interval_task.task_info import Machine
from gems_python.multi_machine_problem_interval_task.lifecycle import fingerprint
from scripts.measure_experiment_lifecycle import prepare, conditions, worker
from scripts.verify_experiment_lifecycle_capacity import verify


class TestLifecycleCapacity(unittest.TestCase):
    def test_paired_loads_match_and_two_generations_have_complete_lineage(self):
        for pattern in ("contention", "heterogeneous", "many_candidates", "mixed_penalties", "replanning"):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                job = dict(pattern=pattern, parents=2, children=2, group_interval=2, generations=2)
                source, _, _ = prepare(dict(job, mode="lifecycle"), Path(directory))
                flat, _, _ = prepare(dict(job, mode="flat"), Path(directory))
                self.assertEqual(fingerprint(conditions(source)), fingerprint(conditions(flat)))
                self.assertEqual(len(source.experiments), 8)
                self.assertEqual(len(source.ended_experiments), 6)
                self.assertEqual(len(flat.ended_experiments), 0)
                self.assertEqual(source.execute_scheduling(solver_time_limit_seconds=1)["status"], "READY")
                validate_groups(source.task_groups, source.reference_time, source.machine_list)

    def test_worker_artifacts_are_verified_without_re_solving(self):
        import json
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            rows = []
            for i, mode in enumerate(("lifecycle", "flat")):
                target = root / f"trial-{i:04d}"
                target.mkdir()
                job = dict(series="scan", pattern="contention", budget=1.0, parents=1,
                           children=2, group_interval=2, generations=1,
                           mode=mode, repetition=0)
                (target / "job.json").write_text(json.dumps(job))
                worker(target / "job.json")
                row = json.loads((target / "result.json").read_text())
                self.assertTrue(row["outcome"].startswith("VALID"), row)
                rows.append(dict(row, artifact_directory=target.name))
            (root / "results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            (root / "completion.json").write_text(json.dumps(dict(
                trial_count=2, fully_measured=True, skipped=[])))
            report = verify(root)
            self.assertEqual(report["verified_pairs"], 1)
            self.assertEqual(report["valid_outputs"], 2)
            target = root / rows[0]["artifact_directory"]
            original = (target / "output.json").read_text()
            corrupted = Experiments.from_json(original)
            corrupted.machine_list.machines.append(Machine(0, "forged-device"))
            (target / "output.json").write_text(corrupted.to_json())
            save_snapshot(corrupted, target / "saved")
            with self.assertRaisesRegex(AssertionError, "changed registered machines"):
                verify(root)
            corrupted = Experiments.from_json(original)
            corrupted.experiments[0].initial_conditions["container_id"] = "unexpected"
            (target / "output.json").write_text(corrupted.to_json())
            save_snapshot(corrupted, target / "saved")
            with self.assertRaisesRegex(AssertionError, "scientific experiment settings"):
                verify(root)
            corrupted = Experiments.from_json(original)
            experiment = corrupted.experiments[0]
            experiment.shared_variable_history = experiment.history_for_state()
            experiment._inherited_history = None
            (target / "output.json").write_text(corrupted.to_json())
            save_snapshot(corrupted, target / "saved")
            with self.assertRaisesRegex(AssertionError, "provenance or own observations"):
                verify(root)
            corrupted = Experiments.from_json(original)
            corrupted.task_groups[-1].tasks[-1].processing_time += 1
            (target / "output.json").write_text(corrupted.to_json())
            save_snapshot(corrupted, target / "saved")
            with self.assertRaisesRegex(AssertionError, "Planner changed task specification"):
                verify(root)
