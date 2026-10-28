"""Acceptance tests for complete snapshots, legacy reads, and execution resume."""

import contextlib
import copy
from dataclasses import fields
from datetime import datetime, timedelta
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import matplotlib.image as image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from gems_python.common import dataframes as frames
from gems_python.common.class_dumper import recursive_to_dict, recursive_from_dict
from gems_python.common.persistence import save_snapshot
from gems_python.common.serialization import SerializationError, equivalent
from gems_python.one_machine_problem_simple_task import transition_manager as simple
from gems_python.one_machine_problem_interval_task import transition_manager as single
from gems_python.multi_machine_problem_interval_task import transition_manager as multi
from gems_python.multi_machine_problem_interval_task.task_info import TaskStatus
from scripts.compare_schedulers import PENALTIES, evaluate, make_penalty
from tests.roundtrip_fixtures import Bundle, Choice, ConfiguredExperiment, ConfiguredState, Leaf
from tests.simulators.dataframe_comparison import _build, cases, SimpleState, SingleState, TimedState
from tests.experiment_samples_simple_task.ips.experimental_settings import IPSExperiment as SimpleIPS
from tests.experiment_samples_one_interval_task.ips.experimental_settings import IPSExperiment as SingleIPS


def advance(manager, observer, variant, case):
    if variant != "simple":
        result = manager.simulate_one(
            scheduling_method="o",
            observation_provider=None if case["name"] == "single_dummy" else observer,
        )
    else:
        task = min(manager.tasks, key=lambda t: (t.scheduled_time, t.task_id))
        experiment = next(e for e in manager.experiments if e.experiment_uuid == task.experiment_uuid)
        minute = task.scheduled_time + task.processing_time
        manager.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
            task.task_id, frames.make_frame(
                {"time": [minute], "value": [len(experiment.shared_variable_history) + 1]},
                experiment.dataframe_backend,
            ), scheduling_method="o", optimal_time_reference_time=minute,
        )
        result = {"status": "ok", "simulated_time": minute, "task_id": task.task_id}
    if result["status"] != "ok":
        raise AssertionError(result)
    schedule = evaluate(manager, variant)
    if any(schedule[key] for key in ("overlap_minutes", "assignment_errors", "interval_errors")):
        raise AssertionError(schedule)
    return result, schedule


class TestPersistenceRoundtrip(unittest.TestCase):
    def test_nested_types_and_no_input_mutation(self):
        source = Bundle({
            Choice.SECOND: [Leaf(Choice.FIRST, ((2, 5), (9, 12))), {"mixed": (None, uuid4())}],
            7: frozenset((1, 3)),
            "class": ConfiguredState,
            "numpy": np.int64(17),
            "bytes": b"\x00\xff",
            "time": datetime(2026, 10, 28, 7, tzinfo=ZoneInfo("Asia/Tokyo")),
            "duration": timedelta(seconds=71, microseconds=13),
            "timestamp": pd.Timestamp("2026-10-28 07:00:00.000000017"),
        }, Path("directory/保存.json"))
        source.progress = 19
        payload = source.to_dict()
        unchanged = copy.deepcopy(payload)
        for result in (Bundle.from_dict(payload), Bundle.from_json(source.to_json()),
                       Bundle.from_pickle(dumped=source.to_pickle()),
                       Bundle.from_dict(payload)):
            self.assertTrue(equivalent(source, result))
            self.assertEqual(result.progress, 19)
            self.assertIsInstance(result.values[Choice.SECOND][0], Leaf)
        self.assertEqual(payload, unchanged)
        self.assertIsNone(recursive_from_dict(type(None), recursive_to_dict(None)))

    def test_shared_references_survive_sorted_json_and_change_detection(self):
        shared = [Leaf(Choice.FIRST, ((2, 7),))]
        source = Bundle({"first": shared, "second": shared}, Path("."))
        result = Bundle.from_json(json.dumps(source.to_dict(), sort_keys=True))
        self.assertIs(result.values["first"], result.values["second"])
        self.assertTrue(equivalent(source, result))
        result.values["second"] = copy.deepcopy(result.values["second"])
        self.assertFalse(equivalent(source, result))
        with self.assertRaisesRegex(SerializationError, "unknown shared reference"):
            Bundle.from_dict({"_gems_serialization": 1, "kind": "reference", "target": 5})

    def test_concrete_subclasses_settings_and_no_constructor_side_effects(self):
        for backend in ("pandas", "polars"):
            with self.subTest(backend=backend), contextlib.redirect_stdout(io.StringIO()):
                source = ConfiguredExperiment(0.73, backend)
                source.states[0].counter = 11
                source.current_task_group.tasks[0].scheduled_time = 91
                source.current_task_group.tasks[0].task_id = 23
                source.current_task_group.tasks[0].task_status = TaskStatus.IN_PROGRESS
                calls = ConfiguredExperiment.constructor_calls, ConfiguredState.constructor_calls
                for result in (multi.Experiment.from_json(source.to_json()),
                               multi.Experiment.from_pickle(dumped=source.to_pickle())):
                    self.assertIs(type(result), ConfiguredExperiment)
                    self.assertIs(type(result.states[0]), ConfiguredState)
                    self.assertTrue(equivalent(source, result))
                self.assertEqual(calls, (ConfiguredExperiment.constructor_calls, ConfiguredState.constructor_calls))

    def test_uuid_generation_and_sample_subclass_restoration(self):
        for cls in (SimpleIPS, SingleIPS):
            with self.subTest(cls=cls.__name__), contextlib.redirect_stdout(io.StringIO()):
                first, second = cls("GetImage1State"), cls("GetImage1State")
                self.assertIsInstance(first.experiment_uuid, str)
                UUID(first.experiment_uuid)
                self.assertNotEqual(first.experiment_uuid, second.experiment_uuid)
                supplied = uuid4()
                explicit = cls("GetImage1State", experiment_uuid=supplied)
                self.assertEqual(explicit.experiment_uuid, str(supplied))
                self.assertEqual(cls("GetImage1State", experiment_uuid="fixed").experiment_uuid, "fixed")
                for result in (cls.from_json(first.to_json()), cls.from_pickle(dumped=first.to_pickle())):
                    self.assertIs(type(result), cls)
                    self.assertTrue(equivalent(first, result))

    def test_all_penalties_both_formats_and_scheduling(self):
        modules = (("simple", simple, SimpleState), ("single_interval", single, SingleState),
                   ("multi_interval", multi, TimedState))
        for variant, module, state in modules:
            for backend in ("pandas", "polars"):
                for kind in PENALTIES:
                    with self.subTest(variant=variant, backend=backend, penalty=kind), \
                            tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                        experiment = module.Experiment(
                            "penalty", [state()], state.__name__,
                            frames.make_frame({"time": [11]}, backend), experiment_uuid="penalty",
                        )
                        task = experiment.current_task if variant == "simple" else experiment.current_task_group
                        task.penalty_type = make_penalty(variant, kind)
                        kwargs = {"machine_list": multi.MachineList([multi.Machine(0, machine_id="m")])} \
                            if module is multi else {}
                        source = module.Experiments([experiment], Path(directory), **kwargs)
                        source.execute_scheduling()
                        for result in (module.Experiments.from_json(source.to_json()),
                                       module.Experiments.from_pickle(dumped=source.to_pickle())):
                            self.assertTrue(equivalent(source, result))
                            result.execute_scheduling()
                            self.assertEqual(evaluate(source, variant), evaluate(result, variant))
                            p = result.tasks[0].penalty_type if module is simple else result.task_groups[0].penalty_type
                            self.assertEqual(
                                [p.calculate_penalty(i, 11) for i in (0, 5, 17, 63)],
                                [task.penalty_type.calculate_penalty(i, 11) for i in (0, 5, 17, 63)],
                            )

    def test_resume_all_virtual_experiments_matches_uninterrupted(self):
        for case in cases():
            for backend in ("pandas", "polars"):
                with self.subTest(case=case["name"], backend=backend), \
                        tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                    source, observer, variant = _build(case, Path(directory), backend)
                    source.execute_scheduling()
                    pause = max(1, case["steps"] // 2)
                    for _ in range(pause):
                        advance(source, observer, variant, case)
                    resumed = [type(source).from_json(source.to_json()),
                               type(source).from_pickle(dumped=source.to_pickle())]
                    for manager in resumed:
                        self.assertTrue(equivalent(source, manager))
                    for _ in range(case["steps"] - pause):
                        expected = advance(source, observer, variant, case)
                        for manager in resumed:
                            self.assertEqual(expected, advance(manager, observer, variant, case))
                            self.assertTrue(equivalent(source, manager))

    def test_task_progress_ids_machine_allocations_and_history_schemas(self):
        for backend in ("pandas", "polars"):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                source, _, _ = _build({"name": "concurrent"}, Path(directory), backend)
                source.execute_scheduling()
                source.step, source.reference_time = 6, 123
                source.parent_dir_path = Path(directory) / "must-not-be-created"
                for i, group in enumerate(source.task_groups):
                    group.tasks[0].task_status = (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED)[i]
                    group.tasks[0].scheduled_time = 25 + i
                    group.tasks[0].task_id = 100 + i
                data = pd.DataFrame({
                    "time": pd.to_datetime(["2026-10-28T01:02:03Z", None]),
                    "count": pd.Series([1, None], dtype="Int64"),
                    "flag": pd.Series([True, None], dtype="boolean"),
                    "value": [0.12, float("nan")],
                    "event": ["passage", None],
                })
                source.experiments[0].shared_variable_history = frames.convert(data, backend)
                for result in (multi.Experiments.from_json(source.to_json()),
                               multi.Experiments.from_pickle(dumped=source.to_pickle())):
                    self.assertTrue(equivalent(source, result))
                    self.assertEqual(result.step, 6)
                    self.assertEqual(result.task_groups[0].tasks[0].scheduled_time, 25)
                    self.assertFalse(source.parent_dir_path.exists())

    def test_legacy_identifiable_penalties_and_task_times(self):
        for variant in ("simple", "single_interval", "multi_interval"):
            for kind in PENALTIES:
                penalty = make_penalty(variant, kind)
                old = {f.name: getattr(penalty, f.name) for f in fields(penalty)}
                old["penalty_type"] = type(penalty).__name__
                restored = type(penalty).from_json(json.dumps(old))
                self.assertTrue(equivalent(penalty, restored))
        task = simple.OneMachineTask(5, 2, make_penalty("simple", "LinearPenalty"), "work", "test", "id", 9, 17)
        old = {f.name: getattr(task, f.name) for f in fields(task)}
        old["penalty_type"] = json.dumps({"penalty_type": "LinearPenalty", "penalty_coefficient": 1})
        restored = simple.OneMachineTask.from_json(json.dumps(old))
        self.assertEqual(restored.scheduled_time, 17)
        self.assertEqual(restored.penalty_type.calculate_penalty(10, 5), 5)
        with self.assertRaisesRegex(SerializationError, "missing concrete penalty"):
            simple.OneMachineTask.from_dict({**old, "penalty_type": {"penalty_coefficient": 1}})

    def test_actual_main_snapshot_compatibility(self):
        directory = Path(__file__).parent / "fixtures" / "legacy_snapshots"
        for name, cls in (("simple", simple.Experiment), ("single", single.Experiments),
                          ("multi", multi.Experiments)):
            with self.subTest(snapshot=name):
                payload = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
                unchanged = copy.deepcopy(payload)
                restored = cls.from_dict(payload)
                self.assertEqual(payload, unchanged)
                if name == "simple":
                    self.assertEqual(restored.current_task.scheduled_time, 17)
                    self.assertEqual(restored.current_task.task_id, 9)
                else:
                    self.assertEqual((restored.step, restored.reference_time), (3, 47))
                    self.assertEqual(restored.task_groups[0].tasks[0].scheduled_time, 17)
                    self.assertIsInstance(restored.parent_dir_path, Path)
                    if name == "multi":
                        self.assertEqual(restored.experiments[0].states[0].duration, 20)
                        self.assertIs(restored.task_groups[0].tasks[0].task_status, TaskStatus.IN_PROGRESS)
                        self.assertEqual(restored.task_groups[0].tasks[0].allocated_machine_id, "slow_machine")
                    else:
                        self.assertTrue(restored.task_groups[0].tasks[0].completed)
                self.assertTrue(equivalent(restored, cls.from_json(restored.to_json())))

    def test_incomplete_legacy_states_and_invalid_records_fail_with_paths(self):
        path = f"{ConfiguredState.__module__}.{ConfiguredState.__name__}"
        source = {"_class": path}
        for _ in range(2):
            with self.assertRaisesRegex(SerializationError, r"\$\.duration: missing legacy State"):
                multi.State.from_dict(source)
        self.assertEqual(source, {"_class": path})
        for data, message in (
            ({"_gems_serialization": 999, "kind": "object"}, "version"),
            ({"_gems_serialization": 1, "kind": "mystery"}, "kind"),
            ({"_gems_serialization": 1, "kind": "object", "class": "missing.module:Missing"}, "unavailable"),
            ({"_gems_serialization": 1, "kind": "object", "class": "tests.roundtrip_fixtures:Bundle",
              "attributes": {}}, "missing saved fields"),
        ):
            with self.assertRaisesRegex(SerializationError, message):
                Bundle.from_dict(data)
        source = Bundle({}, Path("."))
        source.values["cycle"] = source
        with self.assertRaisesRegex(SerializationError, r"\$\.values.*circular"):
            source.to_json()
        with self.assertRaisesRegex(SerializationError, "circular"):
            source.to_pickle()
        source.values = {"resource": io.StringIO("data")}
        with self.assertRaisesRegex(SerializationError, r"\$\.values.*unsupported"):
            source.to_json()

    def test_restore_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            source = Bundle({1: [Leaf(Choice.SECOND, ((1, 3),)), np.int64(9)]}, Path("保存"))
            path.write_text(source.to_json(), encoding="utf-8")
            command = (
                "from pathlib import Path; import sys; "
                "from tests.roundtrip_fixtures import Bundle; "
                "p=Path(sys.argv[1]); b=Bundle.from_json(p.read_text(encoding='utf-8')); "
                "p.write_text(b.to_json(), encoding='utf-8')"
            )
            subprocess.run([sys.executable, "-c", command, str(path)], check=True, env=os.environ.copy())
            self.assertTrue(equivalent(source, Bundle.from_json(path.read_text(encoding="utf-8"))))

    def test_reload_conflicts_fallbacks_and_failure_atomicity_all_managers(self):
        for module, state in ((simple, SimpleState), (single, SingleState), (multi, TimedState)):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                experiment = module.Experiment("test", [state()], state.__name__, experiment_uuid="test")
                kwargs = {"machine_list": multi.MachineList([multi.Machine(0, machine_id="m")])} \
                    if module is multi else {}
                source = module.Experiments([experiment], Path(directory), **kwargs)
                source.execute_scheduling()
                directory = Path(directory) if module is simple else source.save_dir()
                directory.mkdir(exist_ok=True)
                before = source.to_json()
                for selection in ("auto", "json", "pickle"):
                    with self.assertRaisesRegex(RuntimeError, "Cannot load snapshot"):
                        source.reload(source=selection)
                    self.assertEqual(before, source.to_json())
                save_snapshot(source, directory)
                self.assertTrue(equivalent(source, source.reload()))
                self.assertEqual(before, source.to_json())
                other = copy.deepcopy(source)
                other.experiments[0].experiment_name = "different"
                other.to_pickle(directory / "experiments.pkl")
                with self.assertRaisesRegex(ValueError, "snapshots differ"):
                    source.reload()
                self.assertTrue(equivalent(source, source.reload(source="json")))
                self.assertTrue(equivalent(other, source.reload(source="pickle")))
                self.assertEqual(before, source.to_json())
                (directory / "experiments.json").write_text("broken")
                with self.assertWarns(RuntimeWarning):
                    self.assertTrue(equivalent(other, source.reload()))
                with self.assertRaises(RuntimeError):
                    source.reload(source="json")
                (directory / "experiments.json").write_text(source.to_json())
                (directory / "experiments.pkl").write_bytes(b"broken")
                with self.assertWarns(RuntimeWarning):
                    self.assertTrue(equivalent(source, source.reload()))
                with self.assertRaises(RuntimeError):
                    source.reload(source="pickle")
                (directory / "experiments.json").unlink()
                with self.assertRaisesRegex(RuntimeError, "json:.*pickle:"):
                    source.reload()
                self.assertEqual(before, source.to_json())
                with self.assertRaisesRegex(ValueError, "source must"):
                    source.reload(source="unknown")

    def test_save_all_adds_full_collection_and_plots_do_not_modify_tasks(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            experiment = simple.Experiment("test", [SimpleState()], "SimpleState", experiment_uuid="test")
            manager = simple.Experiments([experiment], root)
            manager.execute_scheduling()
            manager.save_all()
            self.assertTrue((root / "tasks.json").is_file())
            self.assertTrue(equivalent(manager, manager.reload()))
            source = copy.deepcopy(manager.tasks)
            for method in (simple.OneMachineTask.vis, simple.OneMachineTask.vis_with_diff):
                path = root / f"{method.__name__}.png"
                method(manager.tasks, path)
                pixels = image.imread(path)
                self.assertGreater(pixels.shape[0], 0)
                self.assertGreater(pixels.shape[1], 0)
                self.assertGreater(float(pixels.max() - pixels.min()), 0)
                self.assertTrue(equivalent(source, manager.tasks))
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
