from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.analyze_experiment_lifecycle_capacity import summarize
from scripts.measure_experiment_lifecycle import DEFAULT_CONFIG, measure, recover_measurement


class TestCapacityRecovery(unittest.TestCase):
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
