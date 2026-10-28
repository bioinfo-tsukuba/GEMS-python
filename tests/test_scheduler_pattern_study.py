"""Checks for the paired scheduler study's case coverage and aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.study_scheduler_patterns import (
    aggregate, all_cases, input_fingerprint, load_case_results, make_case,
)


class TestSchedulerPatternStudy(unittest.TestCase):
    def test_case_grid_and_fingerprints(self):
        cases = all_cases(("simple", "single_interval", "multi_interval"))
        self.assertEqual(len(cases), 64)
        self.assertEqual(len({case["case_id"] for case in cases}), 64)
        with tempfile.TemporaryDirectory() as directory:
            first = cases[0]
            manager_a = make_case(first, Path(directory))
            manager_b = make_case(first, Path(directory))
            self.assertEqual(
                input_fingerprint(manager_a, first["variant"]),
                input_fingerprint(manager_b, first["variant"]),
            )
            other = next(
                case for case in cases
                if case["variant"] == "multi_interval"
                and case["machine_count"] == 2
            )
            manager_c = make_case(other, Path(directory))
            self.assertNotEqual(
                input_fingerprint(manager_a, first["variant"]),
                input_fingerprint(manager_c, other["variant"]),
            )

    def test_infeasible_annealing_scores_are_excluded(self):
        cp = {
            "status": "OPTIMAL", "timing_penalty": 10,
            "overlap_minutes": 0, "assignment_errors": 0,
            "interval_errors": 0, "seconds": 0.1,
        }
        annealing = [
            {
                "status": "FINISHED", "timing_penalty": score,
                "overlap_minutes": overlap, "assignment_errors": 0,
                "interval_errors": 0, "seconds": 1.0,
            }
            for score, overlap in ((5, 4), (13, 0), (10, 0), (8, 0))
        ]
        result = aggregate([(cp, annealing)])
        self.assertEqual(result["sa_feasible"], 3)
        self.assertEqual(result["cp_better"], 1)
        self.assertEqual(result["equal_score"], 1)
        self.assertEqual(result["sa_better"], 1)
        self.assertEqual(result["median_gap"], 0)
        self.assertEqual(result["p90_gap"], 3)
        self.assertEqual(result["cases_with_feasible_sa"], 1)
        self.assertEqual(result["case_sa_better"], 1)

    def test_reanalysis_rejects_changed_inputs(self):
        cp = {
            "case_id": "case1", "method": "o", "seed": None,
            "input_sha256": "original",
        }
        sa = {
            "case_id": "case1", "method": "s", "seed": 0,
            "input_sha256": "changed",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.json"
            path.write_text(json.dumps([cp, sa]))
            with self.assertRaisesRegex(ValueError, "Input mismatch"):
                load_case_results(path)


if __name__ == "__main__":
    unittest.main()
