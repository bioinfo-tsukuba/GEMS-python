"""Verify the received-result timing path uses one search and a complete JSON response."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gems_python.common import dataframes
from scripts.measure_scheduler_capacity import DEFAULT_CONFIG, make_case
from scripts.measure_result_update_latency import measure_response, result_event


class TestResultUpdateLatency(unittest.TestCase):
    def test_result_to_json_uses_one_search_and_updates_history(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            manager = make_case("heterogeneous", 2, Path(temporary), config)
            manager.execute_scheduling(solver_time_limit_seconds=1.0)
            event = result_event(manager)
            with patch.object(manager, "execute_scheduling", wraps=manager.execute_scheduling) as search:
                row, response, _ = measure_response(manager, event, 1.0)
            self.assertEqual(search.call_count, 1)
            self.assertEqual(row["outcome"], "VALID", row["error"])
            self.assertEqual(row["solver_calls"], 1)
            self.assertEqual(len(json.loads(response)["tasks"]), 6)
            self.assertGreaterEqual(row["response_seconds"], row["search_seconds"])
            self.assertAlmostEqual(row["response_seconds"], sum(
                row[key] for key in ("progress_seconds", "update_api_seconds", "json_seconds")))

    def test_group_completion_generates_next_group_before_return(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            manager = make_case("heterogeneous", 1, Path(temporary), config)
            manager.execute_scheduling(solver_time_limit_seconds=1.0)
            group = manager.task_groups[0]
            for task in group.tasks[:-1]:
                task.task_status = type(task.task_status).COMPLETED
            group.status_update()
            manager.experiments[0].shared_variable_history = dataframes.make_frame({
                "time": [t.scheduled_time + t.processing_time for t in group.tasks[:-1]],
                "value": [1, 2],
            })
            event = result_event(manager)
            self.assertTrue(event["completes_group"])
            row, response, _ = measure_response(manager, event, 1.0)
            self.assertEqual(row["outcome"], "VALID", row["error"])
            self.assertEqual(len(manager.experiments[0].shared_variable_history), 3)
            self.assertEqual(len(json.loads(response)["tasks"]), 3)
            self.assertTrue(all(t["task_status"] == "Not Started"
                                for t in json.loads(response)["tasks"]))
