import unittest
from dict_vec import DictVector

class DictVecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vec = DictVector()
        self.vec["a"] = 1
        self.vec["b"] = 2
        self.vec["c"] = 3

    def testAdditionWithEmpty(self):
        other = {}
        self.assertEqual(self.vec + other, self.vec)

    def testAdditionWithNonepmty(self):
        other = DictVector()
        other["a"] = 1
        other["b"] = 2
        result = self.vec + other
        self.assertEqual(result["a"], 2)
        self.assertEqual(result["b"], 4)
        self.assertEqual(result["c"], 3)

    def testWithDifferentKeys(self):
        other = DictVector()
        other["a"] = 1
        other["e"] = 2
        result = self.vec + other
        self.assertEqual(result["a"], 2)
        self.assertEqual(result["b"], 2)
        self.assertEqual(result["c"], 3)
        self.assertEqual(result["e"], 2)

    def testSubtraction(self):
        other = DictVector()
        other["a"] = 1
        other["e"] = 2
        result = self.vec - other
        self.assertEqual(result["a"], 0)
        self.assertEqual(result["e"], -2)

    def testMul(self):
        a = 2
        result = self.vec * a
        self.assertEqual(result["a"], 2)
        self.assertEqual(result["b"], 4)
        self.assertEqual(result["c"], 6)

    def testRMul(self):
        a = 2
        result = a * self.vec
        self.assertEqual(result["a"], 2)
        self.assertEqual(result["b"], 4)
        self.assertEqual(result["c"], 6)

    def testConstructor(self):
        dic = {"a": 1, "b": 2}
        vec = DictVector(dic)
        self.assertEqual(vec["a"], 1)
        self.assertEqual(vec["b"], 2)