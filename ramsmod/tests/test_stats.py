import unittest
import pandas as pd
from ramsmod.stats import log_rank_test

class TestLogRank(unittest.TestCase):

    def test_dataset1(self):
        t1 = pd.Series([53, 28, 69, 58, 54, 25, 51, 61, 57, 57, 50])
        d1 = pd.Series([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        t2 = pd.Series([34, 32, 9, 19, 50, 48])
        d2 = pd.Series([1, 1, 1, 1, 0, 0])

        table, stat, p = log_rank_test(t1, d1, t2, d2)
        expected = 0.0138
        self.assertAlmostEqual(p, expected, places=4)

if __name__ == '__main__':
    unittest.main()



