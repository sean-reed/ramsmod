import unittest
import numpy as np
import numpy.testing as npt
from ramsmod.generate_data import *

class MockDistribution:
    def __init__(self, rvs_sequence):
        """
        A mock of a frozen scipy.stats distribution object where rvs method
        outputs a specified sequence of values.
        Parameters
        ----------
        rvs_sequence: sequence of outputs to give when rvs method called.
        Cycles back to start of sequence when end reached.
        """
        self.rvs_sequence = rvs_sequence
        self.i = 0

    def rvs(self, size):
        output = []
        while size > 0:
            output.append(self.rvs_sequence[self.i])
            self.i = self.i+1 if self.i < len(self.rvs_sequence) - 1 else 0
            size -= 1

        return np.array(output)


class TestGenerateRandomRightCensored(unittest.TestCase):
    def setUp(self):
        ttf1 = MockDistribution([10.1, 40.3, 19.8, 30, 45, 50, 70, 80, 90, 30])
        censor1 = MockDistribution([30, 35, 40, 22, 40, 89, 75, 60, 48, 40])
        self.censored_ttf1 = generate_right_censored(ttf1, censor1, 10, sort=False)
        self.censored_ttf1_sorted = generate_right_censored(ttf1, censor1, 10)

    def test_table_shape(self):
        self.assertEqual(self.censored_ttf1.shape, (10, 2))
        self.assertEqual(self.censored_ttf1_sorted.shape, (10, 2))

    def test_times_correct(self):
        expected_times1 = np.array([10, 35, 20, 22, 40, 50, 70, 60, 48, 30])
        expected_times1_sorted = np.array([10, 20, 22, 30, 35, 40, 48, 50, 60, 70])

        npt.assert_equal(self.censored_ttf1['t'], expected_times1)
        npt.assert_equal(self.censored_ttf1_sorted['t'], expected_times1_sorted)

    def test_right_censoring_correct(self):
        expected_d1 = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1])
        expected_d1_sorted = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

        npt.assert_equal(self.censored_ttf1['d'], expected_d1)
        npt.assert_equal(self.censored_ttf1_sorted['d'], expected_d1_sorted)


class TestGeneratePeriodicIntervalCensored(unittest.TestCase):
    def setUp(self):
        ttf = MockDistribution([10, 40, 20, 30, 45, 50, 70, 80, 90, 30])
        inspection_period = 20
        sojourn_rv   = MockDistribution([3,3,3,3,3,3,3,3,3,3])
        self.censored_failure_data = generate_interval_censored(ttf, inspection_period, 10,
                                                                sojourn_rv=sojourn_rv)

        ttf = MockDistribution([10, 50, 20, 30, 45, 55, 70, 90, 85])
        entry_rv = MockDistribution([0, 1, 2, 2, 3, 4, 2, 0, 1])
        sojourn_rv   = MockDistribution([3, 3, 1, 4, 8, 2, 4, 2, 3])
        self.censored_failure_data2 = generate_interval_censored(ttf, inspection_period, 9,
                                                                 entry_rv= entry_rv,
                                                                 sojourn_rv=sojourn_rv, )

    def test_table_shape(self):
        self.assertEqual(self.censored_failure_data.shape, (10, 2))
        self.assertEqual(self.censored_failure_data2.shape, (9, 2))

    def test_lower_bounds_correct(self):
        expected_t_min = np.array([0, 20, 0, 20, 40, 40, 60, 60, 60, 20])
        npt.assert_equal(self.censored_failure_data['tmin'], expected_t_min)

        expected_t_min2 = np.array([0, 40, 0, 0, 0, 0, 60, 40, 80])
        npt.assert_equal(self.censored_failure_data2['tmin'], expected_t_min2)

    def test_upper_bounds_correct(self):
        expected_t_max = np.array([20, 40, 20, 40, 60, 60, np.infty, np.infty, np.infty, 40])
        npt.assert_equal(self.censored_failure_data['tmax'], expected_t_max)

        expected_t_max2 = np.array([20, 60, 60, 60 ,80, 100, 80, np.infty, np.infty])
        npt.assert_equal(self.censored_failure_data2['tmax'], expected_t_max2)


if __name__ == '__main__':
    unittest.main()