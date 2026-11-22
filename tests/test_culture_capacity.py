"""Scientific invariants and complete-path checks for the ROUND3 study."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.calibrate_round3_growth import curve, crossing_hours, fit_curve
from scripts.culture_capacity import CultureObservations, build_manager, draw_cultures, predict
from scripts.extract_round3_culture_data import history_rows, git_blob_hash
from scripts.measure_culture_capacity import advance, independent_event_check


CONFIG = json.loads((Path(__file__).parents[1] /
                     "scripts/culture_capacity_config.json").read_text())


def empirical_fixture():
    pool = []
    for kind in ("HEK", "iPS"):
        for donor, rate in [("a", .03), ("b", .045)]:
            for generation in range(4 if kind == "HEK" else 3):
                pool.append({"cell_type": kind, "donor": donor, "generation": generation,
                             "model": "logistic", "initial_density": .1 if kind == "HEK" else .01,
                             "rate_per_hour": rate, "lag_hours": 0})
    return {"parameter_pool": pool}


class GrowthTests(unittest.TestCase):
    def test_legacy_json_history_and_ragged_failure(self):
        encoded = json.dumps({"columns": [
            {"name": "time", "values": [0, 1]},
            {"name": "density", "values": [None, .25]},
        ]})
        self.assertEqual(history_rows(encoded), [
            {"time": 0, "density": None}, {"time": 1, "density": .25}])
        self.assertEqual(history_rows('{"columns":[]}'), [])
        with self.assertRaises(ValueError):
            history_rows({"columns": [{"name": "a", "values": [1]},
                                      {"name": "b", "values": []}]})
        self.assertEqual(git_blob_hash(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")

    def test_fitted_model_predicts_held_out_logistic_values(self):
        truth = {"model": "logistic", "initial_density": .08,
                 "rate_per_hour": .04, "lag_hours": 0}
        times = np.array([12, 24, 36, 48, 60])
        fit = fit_curve(times, curve(times, truth), "logistic")
        np.testing.assert_allclose(curve([72, 84], fit), curve([72, 84], truth), atol=1e-5)
        self.assertAlmostEqual(crossing_hours(.8, fit), crossing_hours(.8, truth), places=3)

    def test_lag_recovery_and_elapsed_time(self):
        truth = {"model": "lag_logistic", "initial_density": .01,
                 "rate_per_hour": .04, "lag_hours": 48}
        times = [12, 24, 48, 72, 96, 120, 144]
        fit = fit_curve(times, curve(times, truth), "lag_logistic")
        self.assertAlmostEqual(fit["lag_hours"], 48, places=3)
        self.assertGreater(curve([144], truth)[0], curve([96], truth)[0])

    def test_unidentifiable_and_negative_growth_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_curve([24, 48], [.1, .2], "logistic")
        with self.assertRaises(ValueError):
            fit_curve([24, 48, 72], [.3, .2, .1], "logistic")

    def test_online_prediction_has_no_latent_parameter_input(self):
        truth = {"model": "logistic", "initial_density": .1,
                 "rate_per_hour": .035, "lag_hours": 0}
        times = np.array([24, 36, 48])
        history = pd.DataFrame({"time": [0, *list(times*60)],
                                "operation": ["HEKPassage"]+["HEKGetImage"]*3,
                                "density": [None, *curve(times, truth)]})
        actual = predict(history, "HEK", .8)
        self.assertAlmostEqual(actual, crossing_hours(.8, truth)*60, delta=.1)
        self.assertTrue(math.isinf(predict(history.iloc[:2], "HEK", .8)))

    def test_individual_variation_and_paired_condition_prefixes(self):
        small = draw_cultures(empirical_fixture(), CONFIG, 3, 3, 0)
        large = draw_cultures(empirical_fixture(), CONFIG, 10, 10, 0)
        for identity, culture in small.items():
            self.assertEqual(culture, large[identity])
        hashes = {json.dumps(draw_cultures(empirical_fixture(), CONFIG, 3, 3, r),
                            sort_keys=True) for r in range(30)}
        self.assertGreater(len(hashes), 20)


class ProtocolTests(unittest.TestCase):
    def test_provider_scores_before_reset_and_records_refill(self):
        cultures = draw_cultures(empirical_fixture(), CONFIG, 1, 1, 0)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            manager = build_manager(cultures, CONFIG, directory)
            provider = CultureObservations(cultures, CONFIG)
            manager.execute_scheduling()
            advance(manager, provider)
            # iPS coating is first. The simulator parameters are not in any state.
            for experiment in manager.experiments:
                for state in experiment.states:
                    self.assertNotIn("parameters", vars(state))
                    self.assertNotIn("donor", vars(state))
            self.assertEqual(provider.events[0]["operation"], "iPSPlateCoating")
            refill = next(e for e in manager.experiments if e.experiment_uuid == "refill")
            self.assertEqual(len(refill.current_task_group.tasks), 40)
            self.assertEqual(refill.current_task_group.tasks[1].processing_time, 120)

    def test_received_result_driver_matches_public_simulator(self):
        cultures = draw_cultures(empirical_fixture(), CONFIG, 1, 1, 1)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first = build_manager(cultures, CONFIG, Path(directory)/"one")
            second = build_manager(cultures, CONFIG, Path(directory)/"two")
            a, b = CultureObservations(cultures, CONFIG), CultureObservations(cultures, CONFIG)
            first.execute_scheduling()
            for _ in range(12):
                advance(first, a)
                result = second.simulate_one(observation_provider=b)
                self.assertEqual(result["status"], "ok")
            for x, y in zip(a.events, b.events):
                self.assertEqual((x["operation"], x["start"], x["end"], x["density"]),
                                 (y["operation"], y["start"], y["end"], y["density"]))

    def test_independent_check_detects_duplicate_occupancy(self):
        event = {"start": 0, "end": 10, "duration": 10, "machine": "ROUND3-shared-resource",
                 "group_uuid": "g", "task_id": 0, "group_start": 0, "interval": 0}
        duplicate = {**event, "group_uuid": "other"}
        self.assertTrue(independent_event_check([event, duplicate]))
        self.assertEqual(independent_event_check([event]), [])

    def test_complete_normal_protocol_finishes_at_sampling(self):
        cultures = draw_cultures(empirical_fixture(), CONFIG, 1, 1, 2)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            manager = build_manager(cultures, CONFIG, directory)
            provider = CultureObservations(cultures, CONFIG)
            manager.execute_scheduling()
            for _ in range(256):
                if all(e.experiment_uuid == "refill" for e in manager.experiments):
                    break
                advance(manager, provider)
            else:
                self.fail("Normal fixture did not finish")
            self.assertEqual(set(manager.ended_experiments), {"HEK-00", "iPS-00"})
            targets = [e for e in provider.events if e["target"] is not None]
            self.assertEqual(len(targets), 6)
            self.assertEqual(sum(e["operation"].endswith("Sampling") for e in provider.events), 2)
            self.assertGreater(sum(e["duration"] for e in provider.events
                                   if e["culture_id"] == "refill"), 0)
            self.assertFalse(any(e["operation"].endswith("Waiting") for e in provider.events))
            self.assertEqual(independent_event_check(provider.events), [])


if __name__ == "__main__":
    unittest.main()
