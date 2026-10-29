"""Verify statistics distinguish failed attempts, valid outputs and paired inputs."""

import unittest

from scripts.analyze_scheduler_capacity import describe, pair_quality, summarize_trials


def trial(limit, outcome, seconds, penalty, fingerprint="same"):
    return dict(
        phase="scan", pattern="contention", search_limit_seconds=str(limit),
        size="8", input_hash=fingerprint, outcome=outcome,
        search_seconds=str(seconds), build_seconds="0.01", apply_seconds="0.01",
        validation_seconds="0.01", other_seconds="0.01", total_seconds=str(seconds),
        memory_peak_mib="200", movable_penalty=str(penalty) if penalty is not None else "",
        delay_median_minutes="0", delay_p95_minutes="0", delay_max_minutes="0",
        machine_utilization="0.5",
    )


class TestCapacityStatistics(unittest.TestCase):
    def test_linear_quantiles_and_sample_deviation(self):
        result = describe([1, 2, 3, 4, 5])
        self.assertEqual(result["median"], 3)
        self.assertEqual(result["q1"], 2)
        self.assertEqual(result["q3"], 4)
        self.assertAlmostEqual(result["p95"], 4.8)
        self.assertAlmostEqual(result["sample_sd"], 1.5811388300841898)
        self.assertIsNone(describe([2])["sample_sd"])
        self.assertEqual(describe([]), {"n": 0})

    def test_failed_attempts_count_in_time_but_not_quality(self):
        rows = [trial(1, "VALID", 1, 3), trial(1, "NO_SOLUTION", 2, None)]
        result = summarize_trials(rows)[0]
        self.assertEqual(result["valid_fraction"], .5)
        self.assertEqual(result["total_seconds"]["n"], 2)
        self.assertEqual(result["total_seconds"]["median"], 1.5)
        self.assertEqual(result["movable_penalty"]["n"], 1)
        self.assertEqual(result["delay_max_minutes"]["n"], 1)

    def test_failed_pair_is_missing_and_not_zero_quality(self):
        pair = pair_quality([trial(1, "NO_SOLUTION", 1, None),
                             trial(60, "VALID", 60, 10)])[0]
        self.assertIsNone(pair["movable_penalty_delta_60_minus_1"])
        self.assertEqual(pair["valid_1"], 0)
        self.assertEqual(pair["valid_60"], 1)

    def test_pairs_require_matching_input(self):
        with self.assertRaisesRegex(ValueError, "Paired input mismatch"):
            pair_quality([trial(1, "VALID", 1, 10), trial(60, "VALID", 60, 4, "changed")])

    def test_unequal_repetition_counts_compare_condition_medians(self):
        rows = [trial(1, "VALID", 1, 8), trial(1, "VALID", 1, 10),
                trial(60, "VALID", 60, 3)]
        pair = pair_quality(rows)[0]
        self.assertEqual(pair["runs_1"], 2)
        self.assertEqual(pair["runs_60"], 1)
        self.assertEqual(pair["movable_penalty_delta_60_minus_1"], -6)
