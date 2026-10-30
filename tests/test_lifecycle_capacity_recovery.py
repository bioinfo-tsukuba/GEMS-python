from datetime import datetime, timedelta, timezone
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.analyze_experiment_lifecycle_capacity import summarize
from scripts.measure_experiment_lifecycle import DEFAULT_CONFIG, measure, recover_measurement


class TestCapacityRecovery(unittest.TestCase):
    def test_unstable_largest_scan_success_checks_lower_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "study"
            config = json.loads(DEFAULT_CONFIG.read_text())
            config.update(patterns=["contention"], parent_counts=[1, 2, 4],
                          planning_budgets_seconds=[1.0], total_runtime_seconds=60,
                          variation_child_counts=[], variation_group_intervals=[])
            def recorded_worker(command, **kwargs):
                path = Path(command[-1])
                job = json.loads(path.read_text())
                unstable = job["series"] == "boundary" and job["parents"] == 4
                path.with_name("result.json").write_text(json.dumps(dict(
                    job, outcome="VALID_LATE" if unstable else "VALID_ON_TIME")))
            with patch("scripts.measure_experiment_lifecycle.subprocess.run",
                       side_effect=recorded_worker), \
                    patch("scripts.measure_experiment_lifecycle.subprocess.check_output",
                          return_value="test-fixture\n"), \
                    contextlib.redirect_stdout(io.StringIO()):
                rows = measure(config, root)
            repeats = {parents: [r["outcome"] for r in rows
                                 if r["parents"] == parents and r["mode"] == "lifecycle"
                                 and r["series"] in ("scan", "boundary")]
                       for parents in (1, 2, 4)}
            self.assertEqual(repeats[4], ["VALID_ON_TIME", "VALID_LATE", "VALID_LATE"])
            self.assertEqual(repeats[2], ["VALID_ON_TIME"] * 3)
            self.assertEqual(len(repeats[1]), 1)
            self.assertTrue(json.loads((root / "completion.json").read_text())["fully_measured"])

    def test_expired_resume_keeps_completed_trials_without_running_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(DEFAULT_CONFIG.read_text())
            config.update(patterns=["contention"], parent_counts=[1],
                          variation_child_counts=[], variation_group_intervals=[],
                          boundary_repetitions=1)
            (root / "config.json").write_text(json.dumps(config))
            expired = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
            (root / "metadata.json").write_text(json.dumps(dict(started=expired)))
            (root / "study_budget.json").write_text(json.dumps(dict(deadline=expired)))
            rows = []
            for index, mode in enumerate(("lifecycle", "flat")):
                target = root / f"trial-{index:04d}"
                target.mkdir()
                job = dict(series="scan", pattern="contention", budget=1.0, parents=1,
                           children=2, group_interval=2, generations=1, mode=mode, repetition=0)
                (target / "job.json").write_text(json.dumps(job))
                result = dict(job, outcome="VALID_ON_TIME")
                (target / "result.json").write_text(json.dumps(result))
                rows.append(dict(result, artifact_directory=target.name))
            checkpoint = "".join(json.dumps(row)+"\n" for row in rows)
            (root / "checkpoint-0002.jsonl").write_text(checkpoint)
            with patch("scripts.measure_experiment_lifecycle.verify_resume_source",
                       return_value={"source_commit": "test-fixture"}), \
                    patch("scripts.measure_experiment_lifecycle.subprocess.run") as run:
                recovered = measure(config, root, resume=True)
                run.assert_not_called()
                self.assertEqual(len(recovered), 2)
                self.assertEqual(len(list(root.glob("trial-*"))), 2)
                completion = json.loads((root / "completion.json").read_text())
                self.assertFalse(completion["fully_measured"])
                self.assertTrue(completion["skipped"])
                self.assertEqual((root / "checkpoint-0002.jsonl").read_text(), checkpoint)
                # A changed result must be rejected before accepting another resume.
                changed = json.loads((root / "trial-0000" / "result.json").read_text())
                changed["parents"] = 99
                (root / "trial-0000" / "result.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "does not match ledger"):
                    recover_measurement(root, config)

    def test_storage_phases_do_not_pool_save_or_pipeline_times(self):
        base = dict(series="scan", pattern="contention", budget=1.0, mode="lifecycle",
                    parents=1, children=2, group_interval=2, generations=1,
                    outcome="VALID_ON_TIME", dimensions={"active_experiments": 2})
        rows = [
            dict(base, planning_seconds=.1, save_seconds=.01, total_pipeline_seconds=.3,
                 measured_phase_seconds=.2, storage_phase="initial_tmpfs"),
            dict(base, planning_seconds=.2, save_seconds=10, total_pipeline_seconds=11,
                 measured_phase_seconds=10.2, storage_phase="resume_workspace"),
        ]
        summary = summarize(rows)[0]
        self.assertAlmostEqual(summary["planning_seconds_median"], .15)
        self.assertNotIn("save_seconds_median", summary)
        self.assertNotIn("total_pipeline_seconds_median", summary)
        self.assertNotIn("measured_phase_seconds_median", summary)
        self.assertEqual(json.loads(summary["save_seconds_by_storage_phase"]),
                         {"initial_tmpfs": {"min": .01, "median": .01, "max": .01},
                          "resume_workspace": {"min": 10, "median": 10, "max": 10}})

    def test_worker_failure_before_dimensions_does_not_hide_valid_repeat(self):
        valid = dict(series="scan", pattern="contention", budget=1.0, mode="lifecycle",
                     parents=1, children=2, group_interval=2, generations=1,
                     outcome="VALID_ON_TIME", dimensions={"active_experiments": 2},
                     planning_seconds=.1)
        failed = {k: v for k, v in valid.items()
                  if k not in ("dimensions", "planning_seconds")}
        failed["outcome"] = "WORKER_ERROR"
        summary = summarize([failed, valid])[0]
        self.assertEqual(summary["repetitions"], 2)
        self.assertEqual(summary["on_time"], 1)
        self.assertEqual(summary["active_experiments"], 2)
        self.assertEqual(json.loads(summary["outcomes"]),
                         {"WORKER_ERROR": 1, "VALID_ON_TIME": 1})
