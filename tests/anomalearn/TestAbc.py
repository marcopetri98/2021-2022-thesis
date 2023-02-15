import unittest

from anomalearn.abc import EqualityABC
from tests.anomalearn.stubs.AbstractObjects import ObjectWithEquality, \
    ObjectWithoutEquality, ObjectNoMoreWithEquality, ObjectWithoutEquality2, \
    ObjectWithEqualityInherit


class TestAbc(unittest.TestCase):
    def test_equality_abc(self):
        self.assertTrue(issubclass(ObjectWithEquality, EqualityABC))
        self.assertFalse(issubclass(ObjectNoMoreWithEquality, EqualityABC))
        self.assertFalse(issubclass(ObjectWithoutEquality, EqualityABC))
        self.assertFalse(issubclass(ObjectWithoutEquality2, EqualityABC))
        self.assertTrue(issubclass(ObjectWithEqualityInherit, EqualityABC))
