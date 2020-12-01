import pandas as pd
import numpy as np
from scipy.stats import norm
from ramsmod.utils import convert_to_pd_series

__all__ = ['kaplan_meier_fit', 'turnbull_fit']


def kaplan_meier_fit(t, d, ci=None, alpha=0.05):
    """
    Produces a table from right-censored failure data containing data relating to
    the Kaplan-Meier estimate of the reliability function.
    :param t: Survival times of the observations in the failure data.
    :param d: Indicator variable values showing if observations were failures (value 1) or right-censored (value 0).
    :param ci: string with value 'gw' for Greenwood's and 'egw' for exponential
    Greenwood's confidence interval bounds.
    :param alpha: float giving level of significance for confidence interval (i.e.
    giving (1-alpha)*100% confidence level it contains true reliability). Default is 0.05.
    :return: Pandas DataFrame with columns 't', 'm', 'n' and 'R' containing
     the ordered failure times and corresponding number of observed failures,
     number at risk and Kaplan-Meier reliability estimates respectively.
   """
    t = convert_to_pd_series(t)
    d = convert_to_pd_series(d)

    if t.size != d.size:
        raise ValueError("times and observed must be equal size.")

    # Get the ordered failure times.
    only_failures = t[d == 1]
    failures_with_zero = only_failures.append(
        pd.Series([0]), ignore_index=True)
    unique_failures = pd.Series(failures_with_zero.unique())
    tf = unique_failures.sort_values(ignore_index=True)

    # Get the number of failures observed at the ordered failure times.
    mf = tf.apply(lambda x: (only_failures == x).sum())

    # Get the number right-censored prior in prior interval to each ordered failure time.
    only_censored = t[d == 0]
    total_q = tf.apply(lambda x: (only_censored < x).sum()).shift(periods=-1)
    qf = total_q.diff()
    qf.iloc[0] = total_q.iloc[0] # Set censored between t_(0) and t_(1).
    qf.iloc[-1] = (only_censored >= tf.iloc[-1]).sum() # Set censored after final ordered failure time.

    # Get the number at risk to fail at the ordered failure times.
    nf = tf.apply(lambda x: (t >= x).sum())

    # Calculate conditional survival probability
    # at the ordered failure times.
    surv_probs = 1 - (mf / nf)

    # Calculate Kaplan-Meier reliability estimates at the
    # ordered failure times.
    r = surv_probs.cumprod()

    # Create Pandas DataFrame with results.
    km_table = pd.DataFrame({'t': tf, 'm': mf, 'q':qf, 'n': nf,
                             'R': r})
    km_table.index.name = "f"

    # Add confidence intervals.
    if ci == 'gw':
        km_table['CI Lower'], km_table['CI Upper'] =  _greenwoods_ci(mf, nf, r, alpha)
    elif ci == 'egw':
        km_table['CI Lower'], km_table['CI Upper'] = _exp_greenwoods_ci(mf, nf, r, alpha)
    elif ci is not None:
        raise ValueError("Invalid ci value, must be 'gw' for Greenwood's or 'egw' for exponential Greenwood's"
                         " confidence interval.")

    return km_table


def _get_greenwoods_formula_cumsums(mf, nf):
    """
    Gets the value of the sum term from Greenwood's confidence
    interval at each ordered failure time.
    """
    sum_terms = (mf / (nf * (nf - mf)))
    sum_terms = sum_terms.replace([np.inf], 0) # No interval where R(t)=0.
    cumulative_sums = np.cumsum(sum_terms)

    return cumulative_sums

def _greenwoods_ci(mf, nf, r, alpha=0.05):
    """
    Gets the confidence interval given by Greenwood's formula.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1 exclusive.")

    cumulative_sums = _get_greenwoods_formula_cumsums(mf, nf)
    # Compute the variance values given by Greenwood's formula at
    # each ordered failure time.
    var = np.power(r, 2) * cumulative_sums
    # Compute the standard deviations at each ordered failure time.
    sd = np.sqrt(var)
    z = norm.ppf(1 - alpha / 2)  # Z-score corresponding to the alpha / 2.
    # Compute the lower and upper bounds of the confidence interval.
    ci_lb = r - z * sd
    ci_ub = r + z * sd

    return ci_lb, ci_ub


def _exp_greenwoods_ci(mf, nf, r, alpha=0.05):
    """
    Gets the exponential Greenwood's confidence interval.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1 exclusive.")

    # Don't calculate confidence interval for time 0 where R(t)=1.
    mf = mf[1:]
    nf = nf[1:]
    r = r[1:]

    with np.errstate(divide='ignore'):
        cumulative_sums = _get_greenwoods_formula_cumsums(mf, nf)
        # Compute the V value in exponential Greenwood's confidence interval.
        v = (1 / np.power(np.log(r), 2)) * cumulative_sums
        z = norm.ppf(1 - alpha / 2)  # Z-score corresponding to the alpha / 2.
        # Compute the c+- values in exponential Greenwood's confidence interval.
        c_plus = np.log(-np.log(r)) + z * np.sqrt(v)
        c_minus = np.log(-np.log(r)) - z * np.sqrt(v)
        # Compute the lower and upper bounds of the confidence interval.
        ci_lb = np.exp(-np.exp(c_plus))
        ci_ub = np.exp(-np.exp(c_minus))

    # Pre-pend confidence interval for time 0 of (1.0, 1.0).
    ci_lb = np.concatenate((pd.Series([1.0]), ci_lb))
    ci_ub = np.concatenate((pd.Series([1.0]), ci_ub))

    return ci_lb, ci_ub


def turnbull_fit(tmin, tmax, tol=0.001):
    """
    Computes Turnbull estimates of reliability from interval censored
    failure data.

    Parameters
    ----------
    tmin: Lower bounds of failure times for observations.
    tmax: Upper bounds of failure times for observations.
    tol: Tolerance for the iterative procedure, convergence terminates when
     maximum difference from previous reliability estimate at any time is less than
     tolerance.

    Returns
    -------
    Pandas DataFrame with column 't' containing the interval end point time values
     and 'R' containing the corresponding Turnbull reliability estimates.
    """
    tmin = convert_to_pd_series(tmin)
    tmax = convert_to_pd_series(tmax)

    if tmin.size != tmax.size:
        raise ValueError("t_min and t_max must be equal size.")

    t = np.sort(pd.Series([0]).append(tmin).append(tmax).unique())

    # Initial reliability function as equal reduction at each time grid point.
    reliabilities = np.linspace(1.0, 0, len(t))

    # Form alphas matrix describing if each observation (matrix rows)
    # could fail (value 1) or not (value 0) in each interval (matrix columns).
    n = len(tmin)
    m = len(t) - 1
    alphas = np.empty((n, m), dtype=np.bool)
    for i in range(n):
        t_min_i = tmin.iloc[i]
        t_max_i = tmax.iloc[i]
        for j in range(m):
            grid_t_min_j = t[j]
            grid_t_max_j = t[j + 1]
            if t_min_i != t_max_i:
                alphas[(i, j)] = t_min_i < grid_t_max_j and t_max_i > grid_t_min_j
            else:  # Exact failure observation.
                alphas[(i, j)] = t_min_i <= grid_t_max_j and t_max_i > grid_t_min_j

    while True:
        # Compute estimated failures in each interval.
        p = -np.diff(reliabilities)  # Interval failure probabilities.
        p_alphas = alphas * p
        d = ((p_alphas.T / p_alphas.sum(axis=1)).T.sum(axis=0))
        # Compute number at risk in each interval.
        y = np.cumsum(d[::-1])[::-1]
        updated_reliabilities = np.insert(np.cumprod(1 - (d / y)), 0, 1)

        difference = np.max(np.abs(updated_reliabilities - reliabilities))
        reliabilities = updated_reliabilities
        if difference <= tol:
            break

    turnbull_table = pd.DataFrame({'t': t, 'R': reliabilities}, index=range(1, m+2))

    return turnbull_table


