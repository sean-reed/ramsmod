import numpy as np
import pandas as pd
from math import pow
from scipy.stats import chi2, norm
from ramsmod.utils import convert_to_pd_series

__all__ = ['log_rank_test', 'mantel_test']


def log_rank_test(t1, d1, t2, d2):
    """
    Performs a log-rank test to evaluate the null hypothesis that
    two groups have the same reliability from right-censored failure data.
    :param t1: Survival times for the observations in the failure data
    for group 1.
    :param d1: Indicator variable values showing if observations
    were failures (value 1) or right-censored (value 0) for group 1.
    :param t2: Survival times of the observations in the failure data
    for group 2.
    :param d2: Indicator variable values showing if observations
    were failures (value 1) or right-censored (value 0) for group 2.
    :return:  A tuple containing a Pandas DataFrame with a table of results from
    the calculations used to perform the test, the log-rank test statistic, the
    estimated variance of the statistic distribution and the calculated P-value for the test.
    """

    # Convert inputs to pd.Series if not already.
    t1 = convert_to_pd_series(t1)
    d1 = convert_to_pd_series(d1)
    t2 = convert_to_pd_series(t2)
    d2 = convert_to_pd_series(d2)

    t = pd.concat([t1, t2])
    d = pd.concat([d1, d2])
    # Ordered failure times.
    tf = pd.Series(t[d == 1].unique()).sort_values(ignore_index=True)
    # Observed failures.
    m1 = tf.apply(lambda x: sum(t1[d1 == 1] == x))
    m2 = tf.apply(lambda  x: sum(t2[d2 == 1] == x))
    # Number at risk.
    n1 = tf.apply(lambda x: sum(t1 >= x))
    n2 = tf.apply(lambda x: sum(t2 >= x))
    # Expected failures under null hypothesis.
    e1 = n1 / (n1 + n2) * (m1 + m2)
    e2 = n2 / (n1 + n2) * (m1 + m2)

    table = pd.DataFrame({'tf': tf, 'm1f': m1, 'm2f': m2, 'n1f': n1, 'n2f': n2,
                          'e1f': e1, 'e2f': e2})

    # Calculate log-rank statistic.
    num = (n1 * n2 * (m1 + m2) * (n1 + n2 - m1 - m2))
    den = (n1 + n2).pow(2) * (n1 + n2 - 1)
    var = sum((num / den).replace([np.nan], 0))
    log_rank_stat = pow(sum(m1) - sum(e1), 2) / var
    p = chi2(1).sf(log_rank_stat)

    return table, log_rank_stat, var, p


def mantel_test(t_min_1, t_max_1, t_min_2, t_max_2):
    """
    Performs a Mantel test to evaluate the null hypothesis that
    two groups have the same reliability from interval-censored failure data.
    :param t_min_1: Exclusive lower bounds of the failure intervals
    for the observations from the group 1 failure data.
    :param t_max_1: Inclusive upper bounds of the failure intervals
    for the observations from the group 1 failure data.
   :param t_min_2: Exclusive lower bounds of the failure intervals
    for the observations from the group 2 failure data.
    :param t_max_2: Inclusive upper bounds of the failure intervals
    for the observations from the group 2 failure data.
    :return: A tuple containing a Pandas DataFrame with a table containing results from
    calculations used to perform the test, the Mantel test statistic, the estimated
    variance in the test statistic and the calculated P-value for the test.
    """
    # Convert inputs to pd.Series if not already.
    t_min_1 = convert_to_pd_series(t_min_1)
    t_max_1 = convert_to_pd_series(t_max_1)
    t_min_2 = convert_to_pd_series(t_min_2)
    t_max_2 = convert_to_pd_series(t_max_2)

    t_min = pd.concat([t_min_1, t_min_2], ignore_index=True)
    t_max = pd.concat([t_max_1, t_max_2], ignore_index=True)

    n_1 = t_min_1.size
    n_2 = t_min_2.size
    n = n_1 + n_2

    later = np.zeros(n)
    earlier = np.zeros(n)
    for i in range(n):
        later[i] = sum(t_min[i] >= t_max)
        earlier[i] = sum(t_max[i] <= t_min)

    v = later - earlier

    table = pd.DataFrame({'t_min': t_min, 't_max': t_max, 'later': later,
                          'earlier': earlier, 'v': v}, index=range(1,n+1))
    table.index.name = "Observation #"

    var = n_1 * n_2 * sum(np.power(v, 2)) / ((n_1 + n_2) * (n_1 + n_2 - 1))
    sd = np.sqrt(var)
    w = sum(v[:n_1])
    p = norm.sf(abs(w), scale=sd)*2

    return table, w, var, p







