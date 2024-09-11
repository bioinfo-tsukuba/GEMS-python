import json
import unittest

from gems_python.one_machine_problem.penalty.penalty_class import LinearPenalty, PenaltyType



class TestPenalty(unittest.TestCase):
    def test_linear_penalty(self):
        linear_penalty = LinearPenalty(10)
        result = linear_penalty.calculate_penalty(100, 90)
        self.assertEqual(result, 100)


    def test_linear_penalty_json(self):
        print()
        linear_penalty = LinearPenalty(10)
        pena_json = linear_penalty.to_json()
        read_pena = PenaltyType.from_json(pena_json)
        self.assertEqual(linear_penalty, read_pena)
        print("pena\n", linear_penalty)
        print("read_pena\n", read_pena)

        must_panic_json = {
            "coefficient": 10, 
            "penalty_type": "ERROR"
            }
        must_panic_json = json.dumps(must_panic_json, ensure_ascii=False, indent=2)
        try:
            LinearPenalty.from_json(must_panic_json)
            self.assertTrue(False)
        except Exception as e:
            print(e)
            self.assertTrue(True)