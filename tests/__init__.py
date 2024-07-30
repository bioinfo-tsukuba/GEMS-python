import unittest
from gems_python.transition_manager import (
    Linear,
    LinearWithRange,
    CyclicalRestPenalty,
    CyclicalRestPenaltyWithLinear,
    get_penalty,
    PenaltyType,
    PENALTY_MAXIMUM,
)

class TestTransitionManager(unittest.TestCase):

    def test_linear_penalty(self):
        linear_penalty = Linear(coefficient=10)
        result = get_penalty(linear_penalty, 100, 90)
        self.assertEqual(result, 100)

    def test_linear_with_range_penalty(self):
        linear_with_range_penalty = LinearWithRange(lower=10, lower_coefficient=5, upper=20, upper_coefficient=10)
        result = get_penalty(linear_with_range_penalty, 5, 0)
        self.assertEqual(result, 25)
        result = get_penalty(linear_with_range_penalty, 15, 0)
        self.assertEqual(result, 0)
        result = get_penalty(linear_with_range_penalty, 25, 0)
        self.assertEqual(result, 50)

    def test_cyclical_rest_penalty(self):
        cyclical_penalty = CyclicalRestPenalty(start_minute=0, cycle_minute=60, ranges=[(15, 30)])
        result = get_penalty(cyclical_penalty, 45, 30)
        self.assertEqual(result, 0)
        result = get_penalty(cyclical_penalty, 15, 0)
        self.assertEqual(result, PENALTY_MAXIMUM)

    def test_cyclical_rest_penalty_with_linear(self):
        cyclical_with_linear_penalty = CyclicalRestPenaltyWithLinear(start_minute=0, cycle_minute=60, ranges=[(15, 30)], coefficient=10)
        result = get_penalty(cyclical_with_linear_penalty, 45, 30)
        self.assertEqual(result, 150)
        result = get_penalty(cyclical_with_linear_penalty, 15, 0)
        self.assertEqual(result, PENALTY_MAXIMUM)

if __name__ == '__main__':
    unittest.main()
