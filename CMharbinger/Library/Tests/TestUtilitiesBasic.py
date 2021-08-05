
import unittest
import pandas

# Functions to test
from UtilitiesBasic import flatten
from UtilitiesBasic import count_tokens_out_df

class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        import TestData
        self.assertEqual(TestData.list1Expect, flatten(TestData.list1Test))
        #
        self.assertEqual(TestData.list2Expect, flatten(TestData.list2Test))

class TestCount_tokens_out_df(unittest.TestCase):
    def test_direct_count_fast(self):
        import TestData
        pandas._testing.assert_frame_equal(TestData.doc2_Expect,
                                           count_tokens_out_df(TestData.doc2))