import unittest
from dataclasses import dataclass
from typing import List, Dict
import inspect
from gems_python.common.class_dumper import recursive_from_dict, recursive_from_json, recursive_to_dict, recursive_to_json

SEPARATE_LINE_LENGTH = 100

def print_separate_line(info: str):
    print("\n", "*" * SEPARATE_LINE_LENGTH, sep="")
    print(info, "*" * (SEPARATE_LINE_LENGTH-len(info)), sep="")
    print("*" * SEPARATE_LINE_LENGTH)


@dataclass
class A:
    A_a: int

    def to_dict(self):
        return recursive_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return recursive_from_dict(cls, data)

@dataclass
class B:
    A: A
    values: List[int]
    metadata: Dict[str, str]

    def to_dict(self):
        return recursive_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return recursive_from_dict(cls, data)

@dataclass
class C:
    B_list: List[B]  # Bクラスのリストを持つ
    name: str

    def to_dict(self):
        return recursive_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return recursive_from_dict(cls, data)

class TestDumper(unittest.TestCase):
    def setUp(self):
        self.a = A(A_a=1)
        self.b = B(A=self.a, values=[1, 2, 3], metadata={"key": "value"})
        self.c = C(B_list=[self.b], name="C")

    def test_recursive_to_dict(self):
        print_separate_line(inspect.currentframe().f_code.co_name)
        dic = self.c.to_dict()
        c = recursive_from_dict(C, dic)
        print(self.c)
        print(c)
        self.assertEqual(c, self.c)

    def test_recursive_to_json(self):
        print_separate_line(inspect.currentframe().f_code.co_name)
        json_str = recursive_to_json(self.c)
        c = recursive_from_json(C, json_str)
        print(self.c)
        print(c)
        self.assertEqual(c, self.c)