"""Acceptance checks at DataFrame boundaries and through live experiments."""

from datetime import datetime, timezone
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import contextlib

import pandas as pd
import polars as pl

from gems_python.common import dataframes as frames
from gems_python.common.class_dumper import auto_dataclass
from gems_python.one_machine_problem_simple_task import transition_manager as simple
from gems_python.one_machine_problem_interval_task import transition_manager as single
from gems_python.multi_machine_problem_interval_task import transition_manager as multi
from tests.simulators.dataframe_comparison import (
    SimpleState, SingleState, TimedState, assert_same, cases, run_case,
)
from scripts.compare_schedulers import PENALTIES, evaluate, make_penalty


@auto_dataclass
class FrameHolder:
    history: frames.DataFrame


class TestDataframeBackends(unittest.TestCase):
    def test_virtual_experiments_match(self):
        for case in cases():
            with self.subTest(case=case["name"]):
                assert_same(case, run_case(case, "pandas"), run_case(case, "polars"))

    def test_auto_explicit_and_invalid_backends_in_all_managers(self):
        for module, state, name in ((simple, SimpleState, "SimpleState"),
                                    (single, SingleState, "SingleState"),
                                    (multi, TimedState, "TimedState")):
            with self.subTest(module=module.__name__), contextlib.redirect_stdout(io.StringIO()):
                default = module.Experiment("test", [state()], name)
                self.assertIsInstance(default.shared_variable_history, pd.DataFrame)
                for backend in ("pandas", "polars"):
                    history = frames.make_frame({"time": [5]}, backend)
                    experiment = module.Experiment("test", [state()], name, history)
                    self.assertEqual(experiment.dataframe_backend, backend)
                    other = "polars" if backend == "pandas" else "pandas"
                    forced = module.Experiment("test", [state()], name, history, dataframe_backend=other)
                    self.assertEqual(frames.backend_of(forced.shared_variable_history), other)
                with self.assertRaises(ValueError):
                    module.Experiment("test", [state()], name, dataframe_backend="unknown")
                with self.assertRaises(TypeError):
                    module.Experiment("test", [state()], name, [])

    def test_sparse_append_numeric_promotion_overwrite_and_index(self):
        for backend in ("pandas", "polars"):
            with self.subTest(backend=backend):
                history = frames.convert(pd.DataFrame({"a": [1], "flag": [True]}, index=[99]), backend)
                other = "polars" if backend == "pandas" else "pandas"
                result = frames.make_frame({"a": [2.5], "b": ["new"]}, other)
                appended = frames.update(history, result)
                self.assertEqual(list(appended.columns), ["a", "flag", "b"])
                self.assertEqual(frames.records(appended), [
                    {"a": 1.0, "flag": True, "b": None},
                    {"a": 2.5, "flag": None, "b": "new"},
                ])
                self.assertEqual(frames.backend_of(appended), backend)
                self.assertEqual(frames.records(history), [{"a": 1, "flag": True}])
                replaced = frames.update(history, result, "o")
                self.assertEqual(frames.backend_of(replaced), backend)
                self.assertEqual(frames.records(replaced), frames.records(result))
                if backend == "pandas":
                    self.assertEqual(list(appended.index), [0, 1])
                with self.assertRaises(TypeError):
                    frames.append(history, frames.make_frame({"a": ["bad"]}, other))
                with self.assertRaises(ValueError):
                    frames.update(history, result, "unknown")

    def test_empty_and_null_histories(self):
        for backend in ("pandas", "polars"):
            for initial in (None, {"a": []}, {"a": [None]}):
                with self.subTest(backend=backend, initial=initial):
                    history = frames.make_frame(initial, backend)
                    rows = frames.records(frames.append(history, frames.make_frame({"a": [1]}, backend)))
                    self.assertEqual(rows[-1], {"a": 1})
                    self.assertEqual(len(rows), len(history) + 1)

    def test_arrow_json_pickle_and_csv(self):
        data = pd.DataFrame({
            "time": pd.to_datetime([datetime(2026, 10, 28, tzinfo=timezone.utc), None]),
            "count": pd.Series([1, None], dtype="Int64"),
            "flag": pd.Series([True, None], dtype="boolean"),
            "value": [0.12, float("nan")],
            "event": ["passage", None],
        })
        for backend in ("pandas", "polars"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                original = frames.convert(data, backend)
                holder = FrameHolder(original)
                restored = FrameHolder.from_json(holder.to_json()).history
                self.assertEqual(frames.backend_of(restored), backend)
                if backend == "pandas":
                    pd.testing.assert_frame_equal(original, restored)
                else:
                    self.assertTrue(original.equals(restored))
                    self.assertEqual(original.schema, restored.schema)
                self.assertEqual(frames.records(pickle.loads(pickle.dumps(original))), frames.records(original))
                path = Path(directory) / "history.csv"
                frames.write_csv(original, path)
                self.assertEqual(path.read_text().splitlines()[0], "time,count,flag,value,event")
                loaded = frames.read_csv(path, backend)
                self.assertEqual(frames.records(loaded), frames.records(original))
                empty = Path(directory) / "empty.csv"
                frames.write_csv(frames.make_frame(backend=backend), empty)
                self.assertEqual(len(frames.read_csv(empty, backend)), 0)

    def test_legacy_and_invalid_serialization(self):
        original = pl.DataFrame({"time": [1], "value": [0.12]})
        legacy = original.serialize(format="json")
        self.assertTrue(FrameHolder.from_dict({"history": legacy}).history.equals(original))
        self.assertTrue(frames.deserialize(original.to_dict(as_series=False)).equals(original))
        for value in ({"_dataframe": 2, "format": "arrow-ipc"}, {"_dataframe": 1, "format": "wrong"}):
            with self.assertRaises(ValueError):
                frames.deserialize(value)
        # Legacy simple experiments identify State classes only by name.
        experiment = simple.Experiment("legacy", [SimpleState()], "SimpleState",
                                       pl.DataFrame({"time": [5]}), experiment_uuid="legacy")
        legacy_experiment = {
            "experiment_name": "legacy", "states": ["SimpleState"],
            "current_state_name": "SimpleState",
            "shared_variable_history": {"time": [5]},
            "current_task": None, "experiment_uuid": "legacy",
        }
        restored = simple.Experiment.from_json(json.dumps(legacy_experiment))
        self.assertIsInstance(restored.shared_variable_history, pl.DataFrame)
        # Restore the saved absence of a task without generating a replacement.
        self.assertIsNone(restored.current_task)

    def test_experiment_save_restore_and_schedule_tables(self):
        for backend in ("pandas", "polars"):
            for module, state, name in ((simple, SimpleState, "SimpleState"),
                                        (single, SingleState, "SingleState"),
                                        (multi, TimedState, "TimedState")):
                with self.subTest(backend=backend, module=module.__name__), contextlib.redirect_stdout(io.StringIO()):
                    experiment = module.Experiment(
                        "test", [state()], name, frames.make_frame({"time": [5]}, backend),
                        experiment_uuid="fixed",
                    )
                    restored = module.Experiment.from_json(experiment.to_json())
                    self.assertEqual(restored.dataframe_backend, backend)
                    self.assertEqual(frames.records(restored.shared_variable_history), [{"time": 5}])
                    self.assertEqual(restored.current_state_name, name)
                    self.assertEqual(restored.current_task if module == simple else restored.current_task_group,
                                     experiment.current_task if module == simple else experiment.current_task_group)
                    if module != simple:
                        default = module.TaskGroup.create_non_completed_tasks_df([])
                        self.assertIsInstance(default, pd.DataFrame)
                        table = module.TaskGroup.create_non_completed_tasks_df([], backend)
                        self.assertEqual(frames.backend_of(table), backend)

    def test_rejected_result_does_not_complete_task(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            experiment = single.Experiment(
                "test", [SingleState()], "SingleState", pd.DataFrame({"time": [5]}),
                experiment_uuid="fixed",
            )
            manager = single.Experiments([experiment], Path(directory))
            manager.execute_scheduling()
            group, task = manager.task_groups[0], manager.task_groups[0].tasks[0]
            with self.assertRaises(TypeError):
                manager.update_shared_variable_history(group.task_group_id, task.task_id,
                                                       pl.DataFrame({"time": ["bad"]}))
            self.assertFalse(task.completed)
            self.assertEqual(frames.records(experiment.shared_variable_history), [{"time": 5}])

    def test_mixed_experiments_save_and_csv_autoload(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            manager = multi.Experiments([
                multi.Experiment("slow", [TimedState()], "TimedState",
                                 experiment_uuid="slow", dataframe_backend="polars"),
                multi.Experiment("fast", [TimedState(1, 1, 1)], "TimedState",
                                 experiment_uuid="fast", dataframe_backend="pandas"),
            ], root, machine_list=multi.MachineList([
                multi.Machine(0, machine_id="slow_machine"), multi.Machine(1, machine_id="fast_machine"),
            ]))

            def opposite_observation(state, history, group, task, minute):
                return frames.make_frame(
                    {"time": [minute], "value": [1]},
                    "polars" if frames.backend_of(history) == "pandas" else "pandas",
                )

            result = manager.simulate_one(observation_provider=opposite_observation, save_each_step=True)
            self.assertEqual((result["experiment_name"], result["simulated_time"]), ("fast", 2))
            self.assertIsInstance(manager.experiments[0].shared_variable_history, pl.DataFrame)
            self.assertIsInstance(manager.experiments[1].shared_variable_history, pd.DataFrame)
            restored = multi.Experiments.from_json((manager.save_dir() / "experiments.json").read_text())
            restored.execute_scheduling()
            self.assertEqual(
                frames.records(restored.experiments[1].shared_variable_history),
                [{"time": 2, "value": 1}],
            )
            self.assertEqual(restored.task_groups, manager.task_groups)
            self.assertEqual(restored.machine_list, manager.machine_list)
            self.assertEqual(restored.to_json(), manager.to_json())
            # CSV completion targets the Polars experiment while the manager is mixed.
            group = next(g for g in manager.task_groups if g.experiment_uuid == "slow")
            task = group.tasks[0]
            csv_path = root / "result.csv"
            pd.DataFrame({"time": [20], "value": [1]}).to_csv(csv_path, index=False)
            (manager.save_dir() / "experiment_result.json").write_text(json.dumps({
                "task_response": "Completed",
                "task_group_id": group.task_group_id, "task_id": task.task_id,
                "schedule_reference_time": 20, "result_path": str(csv_path),
            }))
            manager.auto_load()
            self.assertEqual(frames.records(manager.experiments[0].shared_variable_history),
                             [{"time": 20, "value": 1}])
            self.assertIsInstance(manager.experiments[0].shared_variable_history, pl.DataFrame)
            self.assertEqual(frames.read_csv(manager.save_dir() / "schedule.csv", "pandas").columns[0],
                             "experiment_name")

    def test_all_penalties_schedule_after_json_restore(self):
        for backend in ("pandas", "polars"):
            for variant, module, state, name in (
                ("simple", simple, SimpleState, "SimpleState"),
                ("single_interval", single, SingleState, "SingleState"),
                ("multi_interval", multi, TimedState, "TimedState"),
            ):
                for kind in PENALTIES:
                    with self.subTest(backend=backend, variant=variant, penalty=kind), \
                            tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                        experiment = module.Experiment("test", [state()], name,
                                                       dataframe_backend=backend, experiment_uuid="fixed")
                        task = experiment.current_task if variant == "simple" else experiment.current_task_group
                        task.penalty_type = make_penalty(variant, kind)
                        restored = module.Experiment.from_json(experiment.to_json())
                        kwargs = {"machine_list": multi.MachineList([multi.Machine(0, machine_id="m0")])} \
                            if variant == "multi_interval" else {}
                        before = module.Experiments([experiment], Path(directory), **kwargs)
                        after = module.Experiments([restored], Path(directory), **kwargs)
                        before.execute_scheduling()
                        after.execute_scheduling()
                        self.assertEqual(evaluate(before, variant), evaluate(after, variant))
                        if variant != "simple":
                            tables = [
                                module.TaskGroup.create_non_completed_tasks_df(before.task_groups, b)
                                for b in ("pandas", "polars")
                            ]
                            self.assertEqual(frames.records(tables[0]), frames.records(tables[1]))

    def test_invalid_observation_keeps_history_and_task_uncompleted(self):
        for backend in ("pandas", "polars"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                experiment = single.Experiment("test", [SingleState()], "SingleState",
                                               dataframe_backend=backend)
                manager = single.Experiments([experiment], Path(directory))
                result = manager.simulate_one(observation_provider=lambda *args: {"time": [1]})
                self.assertEqual(result["status"], "error")
                self.assertIn("dummy_output_failed", result["error"])
                self.assertEqual(len(experiment.shared_variable_history), 0)
                self.assertTrue(all(not task.completed for group in manager.task_groups for task in group.tasks))


if __name__ == "__main__":
    unittest.main()
