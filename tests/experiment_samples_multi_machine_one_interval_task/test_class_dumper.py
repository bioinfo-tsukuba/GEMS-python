from enum import Enum
import unittest
from typing import List, Dict
import inspect
from gems_python.common.class_dumper import auto_dataclass as dataclass
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from .minimum import gen_minimum_experiments, gen_standard_experiments


SEPARATE_LINE_LENGTH = 100

def print_separate_line(info: str):
    print("\n", "*" * SEPARATE_LINE_LENGTH, sep="")
    print(info, "*" * (SEPARATE_LINE_LENGTH-len(info)), sep="")
    print("*" * SEPARATE_LINE_LENGTH)


class TestExperimentsDumperStandard(unittest.TestCase):
    def setUp(self):
        self.c = gen_standard_experiments("volatile")

    def test_recursive_to_dict(self):
        # Print Class, Func
        print_separate_line(f"{self.__class__.__name__}:{inspect.currentframe().f_code.co_name}")
        dic = self.c.to_dict()
        print(f"{dic=}")
        c = Experiments.from_dict(dic)
        print(self.c)
        print(c)
        assert c.to_json() == self.c.to_json()


    def test_recursive_to_json(self):
        path = self.c.parent_dir_path
        print_separate_line(f"{self.__class__.__name__}:{inspect.currentframe().f_code.co_name}")
        json_str = self.c.to_json()
        print(f"{json_str=}")
        with open(path / "experiments.json", "w") as f:
            f.write(json_str)

        c = Experiments.from_json(json_str)
        print(f"{self.c=}")
        print(f"{c=}")

        assert c.to_json() == self.c.to_json()

    

    def test_to_pickle(self):
        path = self.c.parent_dir_path
        print_separate_line(f"{self.__class__.__name__}:{inspect.currentframe().f_code.co_name}")
        dumped = self.c.to_pickle()
        print(f"{dumped=}")
        with open(path / "experiments.pkl", "wb") as f:
            f.write(dumped)
        c = Experiments.from_pickle(dumped=dumped)
        print(f"{self.c=}")
        print(f"{c=}")
        assert c.to_json() == self.c.to_json()
