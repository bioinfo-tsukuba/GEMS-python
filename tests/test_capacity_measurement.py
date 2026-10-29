"""Check paired snapshots, solver-only limits and operational measurement."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.measure_scheduler_capacity import (
    DEFAULT_CONFIG, classify, dimensions, fingerprint, make_case, run_trial, snapshot,
)


class TestCapacityMeasurement(unittest.TestCase):
    def test_all_workloads_have_reproducible_inputs_and_valid_small_plans(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pattern in config["patterns"]:
                first = make_case(pattern, 2, root / f"{pattern}-first", config)
                second = make_case(pattern, 2, root / f"{pattern}-second", config)
                value = snapshot(first)
                self.assertEqual(fingerprint(value), fingerprint(snapshot(second)))
                for limit in (1.0, 60.0):
                    directory = root / f"{pattern}-{limit}"
                    directory.mkdir()
                    row = run_trial(dict(
                        pattern=pattern, size=2, phase="scan", repetition=0,
                        search_limit_seconds=limit, input_hash=fingerprint(value),
                        input=value, config=config,
                    ), directory)
                    self.assertEqual(row["outcome"], "VALID", row.get("error"))
                    self.assertEqual(row["actual_input_hash"], fingerprint(value))
                    self.assertEqual(row["reports"][0]["solver_time_limit_seconds"], limit)
                    self.assertEqual(row["reports"][0]["penalty"], row["movable_penalty"])
                if pattern == "replanning":
                    self.assertGreater(dimensions(first)["in_progress_task_count"], 0)

    def test_operation_collects_every_call_including_additions(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        config["operation_steps"] = 4
        config["operation_add_at_steps"] = [2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manager = make_case("heterogeneous", 2, directory / "input", config)
            value = snapshot(manager)
            row = run_trial(dict(
                pattern="heterogeneous", size=2, phase="operation", repetition=0,
                search_limit_seconds=1.0, input_hash=fingerprint(value), input=value, config=config,
            ), directory)
            self.assertEqual(row["outcome"], "VALID", row.get("error"))
            self.assertEqual(len(row["updates"]), 4)
            self.assertEqual(len(row["updates"][2]["added_experiments"]), 1)
            self.assertGreater(len(row["updates"][2]["calls"]), 2)
            self.assertTrue(all(call.get("solver_time_limit_seconds") == 1.0
                                for call in row["reports"]))
            self.assertEqual(len(row["reports"]), sum(len(u["calls"]) for u in row["updates"]))

    def test_unknown_and_infeasible_are_distinct_and_total_time_is_not_a_gate(self):
        self.assertEqual(classify([{"status": "UNKNOWN", "valid": False}]), "NO_SOLUTION")
        self.assertEqual(classify([{"status": "INFEASIBLE", "valid": False}]), "INFEASIBLE")
        self.assertEqual(classify([{"status": "FEASIBLE", "valid": True,
                                   "total_seconds": 100, "solver_time_limit_seconds": 1}]), "VALID")
