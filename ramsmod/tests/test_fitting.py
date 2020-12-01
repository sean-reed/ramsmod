import unittest
import pandas as pd
import numpy as np
import numpy.testing as npt
from ramsmod.fitting import *

class TestKaplanMeierFitter(unittest.TestCase):

    def setUp(self):
        t = pd.Series([7, 10, 10, 11, 11, 12, 12, 18, 18, 22, 24, 25, 25, 25, 26, 26,
                       30, 35, 35, 35, 36, 37, 41, 49, 51, 53])
        d = pd.Series([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0])
        self.km_table = kaplan_meier_fit(t, d)

        self.km_table_with_gw = kaplan_meier_fit(t, d, ci="gw")
        self.km_table_with_egw = kaplan_meier_fit(t, d, ci="egw")

    def test_table_shape(self):
        self.assertEqual(self.km_table.shape, (8, 4))

    def test_table_index(self):
        npt.assert_equal(self.km_table.index.to_numpy(), np.arange(8))

    def test_ordered_failure_times(self):
        tf = self.km_table['t'].to_numpy()
        npt.assert_equal(tf, np.asarray([0,7,10,11,24,26,35,41]))

    def test_failed_at_ordered_failure_times(self):
        mf = self.km_table['m'].to_numpy()
        npt.assert_equal(mf, np.asarray([0,1,2,1,1,1,2,1]))

    def test_number_at_risk(self):
        nf = self.km_table['n'].to_numpy()
        npt.assert_equal(nf, np.asarray([26,26,25,23,16,12,9,4]))

    def test_reliability_values(self):
        r = self.km_table['R']
        npt.assert_allclose(r, np.asarray([1.00, 0.96, 0.88, 0.85, 0.79, 0.73, 0.57, 0.42]), atol=0.005)

    def test_greenwoods_ci(self):
        npt.assert_allclose(
            self.km_table_with_gw['CI Lower'], np.asarray([1.00, 0.89, 0.76, 0.71, 0.63, 0.53, 0.32, 0.12]), atol=0.005)
        npt.assert_allclose(
            self.km_table_with_gw['CI Upper'], np.asarray([1.00, 1.04, 1.01, 0.98, 0.96, 0.92, 0.81, 0.73]), atol=0.005)

    def test_exponential_greenwoods_ci(self):
        npt.assert_allclose(
            self.km_table_with_egw['CI Lower'], np.asarray([1.00,0.76,0.68,0.64,0.57,0.48,0.29,0.14]), atol=0.005)
        npt.assert_allclose(
            self.km_table_with_egw['CI Upper'], np.asarray([1.00,0.99,0.96,0.94,0.91,0.87,0.77,0.69]), atol=0.005)


class TestTurnbullFitter(unittest.TestCase):

    def setUp(self):

        # Data from Turnbull's original 1974 paper
        # "Nonparametric Estimation of a Survivorship Function with Doubly Censored Data".
        t1= 10
        t2= 20
        t3 = 30
        t4 = 40
        t_inf = np.infty
        tmin = []
        tmax = []
        # Deaths
        for i in range(12):
            tmin.append(0)
            tmax.append(t1)
        for i in range(6):
            tmin.append(t1)
            tmax.append(t2)
        for i in range(2):
            tmin.append(t2)
            tmax.append(t3)
        for i in range(3):
            tmin.append(t3)
            tmax.append(t4)
        # Losses (right-censored).
        for i in range(3):
            tmin.append(t1)
            tmax.append(t_inf)
        for i in range(2):
            tmin.append(t2)
            tmax.append(t_inf)
        for i in range(3):
            tmin.append(t4)
            tmax.append(t_inf)
        # Late entries (left-censored).
        for i in range(2):
            tmin.append(0)
            tmax.append(t1)
        for i in range(4):
            tmin.append(0)
            tmax.append(t2)
        for i in range(2):
            tmin.append(0)
            tmax.append(t3)
        for i in range(5):
            tmin.append(0)
            tmax.append(t4)

        tmin = pd.Series(tmin)
        tmax = pd.Series(tmax)

        self.tb_table = turnbull_fit(tmin, tmax, tol=0.001)

    def test_shapes(self):
        self.assertEqual(self.tb_table.shape, (6, 2))

    def test_t_values(self):
        expected_t = np.array([0, 10, 20, 30, 40, np.inf])
        npt.assert_equal(self.tb_table['t'].to_numpy(), expected_t)

    def test_reliability_estimates(self):
        expected_r = np.array([1, 0.538, 0.295, 0.210, 0.095, 0])
        npt.assert_allclose(self.tb_table['R'], expected_r, atol=0.001)



if __name__ == '__main__':
    unittest.main()