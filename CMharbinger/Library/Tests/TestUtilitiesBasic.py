
import unittest
from UtilitiesBasic import flatten

class TestFlatten(unittest.TestCase):

    def test_flatten(self):
        list1Test = ['a', ['b', 'c']]
        list1Expect = ['a', 'b', 'c']
        self.assertEqual(list1Expect, flatten(list1Test))
        #
        list2Test =  ['a', 'b', 'c']
        list2Expect = ['a', 'b', 'c']
        self.assertEqual(list2Expect, flatten(list2Test))
