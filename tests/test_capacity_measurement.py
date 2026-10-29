"""Check paired snapshots, solver-only limits and operational measurement."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.measure_scheduler_capacity import (
    DEFAULT_CONFIG, classify, dimensions, fingerprint, make_case, run_trial, snapshot,
)


class TestCapacityMeasurement(unittest.TestCase):
    def test_cli_worker_can_pickle_added_live_state(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        config["operation_steps"] = 3
        config["operation_add_at_steps"] = [2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = snapshot(make_case("heterogeneous", 2, directory / "input", config))
            request = directory / "request.json"
            request.write_text(json.dumps(dict(
                pattern="heterogeneous", size=2, phase="operation", repetition=0,
                search_limit_seconds=1.0, input_hash=fingerprint(value), input=value, config=config,
            )))
            # Run the actual __main__ path. JSON alone cannot detect a class whose
            # claimed module resolves to a different class when pickle imports it.
            program = """
import runpy
import sys
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
original = Experiments.to_json
def checked_json(manager, *args, **kwargs):
    restored = Experiments.from_pickle(dumped=manager.to_pickle())
    assert len(restored.experiments) == len(manager.experiments)
    return original(manager, *args, **kwargs)
Experiments.to_json = checked_json
sys.argv = ["scripts/measure_scheduler_capacity.py", "--worker", sys.argv[1]]
runpy.run_path("scripts/measure_scheduler_capacity.py", run_name="__main__")
"""
            completed = subprocess.run(
                [sys.executable, "-c", program, str(request)],
                cwd=DEFAULT_CONFIG.parent.parent, capture_output=True, text=True,
                env={**os.environ, "MPLBACKEND": "Agg"}, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            row = json.loads((directory / "result.json").read_text())
            self.assertEqual(row["outcome"], "VALID", row.get("error"))
            self.assertEqual(len(row["updates"][2]["added_experiments"]), 1)

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
