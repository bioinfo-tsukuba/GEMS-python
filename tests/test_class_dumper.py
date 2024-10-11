from enum import Enum
import unittest
from typing import List, Dict
import inspect
from gems_python.common.class_dumper import auto_dataclass as dataclass
from gems_python.one_machine_problem_interval_task.transition_manager import Experiments
from tests.experiment_samples_simple_task.minimum import gen_minimum_experiments, gen_standard_experiments


SEPARATE_LINE_LENGTH = 100

def print_separate_line(info: str):
    print("\n", "*" * SEPARATE_LINE_LENGTH, sep="")
    print(info, "*" * (SEPARATE_LINE_LENGTH-len(info)), sep="")
    print("*" * SEPARATE_LINE_LENGTH)

# Enum例
class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

@dataclass
class A:
    A_a: int
    status: Status = Status.ACTIVE

@dataclass
class B:
    A: A
    values: List[int]
    metadata: Dict[str, str]

@dataclass
class C:
    B_list: List[B]  # Bクラスのリストを持つ
    name: str

class TestDumper(unittest.TestCase):
    def setUp(self):
        self.a = A(A_a=1)
        self.b = B(A=self.a, values=[1, 2, 3], metadata={"key": "value"})
        self.c = C(B_list=[self.b], name="C")

    def test_recursive_to_dict(self):
        print_separate_line(inspect.currentframe().f_code.co_name)
        dic = self.c.to_dict()
        print(f"{dic=}")
        c = C.from_dict(dic)
        print(self.c)
        print(c)
        self.assertEqual(c, self.c)

    def test_recursive_to_json(self):
        print_separate_line(inspect.currentframe().f_code.co_name)
        json_str = self.c.to_json()
        print(f"{json_str=}")
        c = C.from_json(json_str)
        print(self.c)
        print(c)
        self.assertEqual(c, self.c)



class TestExperimentsDumperMinimum(unittest.TestCase):
    def setUp(self):
        self.c = gen_minimum_experiments("volatile")

    def test_recursive_to_dict(self):
        print_separate_line(inspect.currentframe().f_code.co_name)
        dic = self.c.to_dict()
        print(f"{dic=}")
        c = Experiments.from_dict(dic)
        print(self.c)
        print(c)

    def test_recursive_to_json(self):
        print_separate_line(inspect.currentframe().f_code.co_name)
        json_str = self.c.to_json()
        print(f"{json_str=}")
        c = Experiments.from_json(json_str)
        print(f"{self.c=}")
        print(f"{c=}")

        


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