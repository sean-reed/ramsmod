import numpy as np
import pandas as pd


def generate_right_censored(ttf_rv, sojourn_rv, size, decimals=0, sort=True):
    """
    Generates a table of right-censored failure data with random censoring.
    :param ttf_rv: A continuous distribution from scipy.stats for the times to failure,
    e.g. weibull_min(1.4, scale=100).
    :param sojourn_rv: A continuous distribution from scipy.stats for the maximum
    observation time, e.g. uniform(0, 100). Right-censoring occurs for an
    observation if this time is less than the time to failure.
    :param size: Number of observations to generate.
    :param decimals: Number of decimals to round the times.
    :param sort: If True then observations are sorted in order of increasing observed
     survival time.
    :return: Pandas DataFrame where the index gives the observation number and
     column 't' gives the survival time (failure or censorship) and column 'd' gives the value of an indicator
     variable that has value 1 if failure was observed and value 0 if
     failure was right-censored.
    """
    ttf = ttf_rv.rvs(size=size)
    censor_times = sojourn_rv.rvs(size=size)
    t = np.minimum(ttf, censor_times).round(decimals)
    d = (ttf <= censor_times).astype(int)

    ds = pd.DataFrame({'t': t, 'd': d})
    if sort:
        ds = ds.sort_values(by=['t'], ignore_index=True)
    ds.index.name = "Observation #"
    ds.index += 1

    return ds


def generate_interval_censored(ttf_rv, inspection_interval, size, entry_rv=None,
                               sojourn_rv=None):
    """
    Generates a table of interval-censored failure data where failures are only revealed
    at inspections performed periodically at the end of each fixed duration interval
     from the as-new condition.
    :param ttf_rv: A distribution from scipy.stats from which the
     times to failure are sampled, e.g. weibull_min(1.4, scale=100).
    :param inspection_interval: Duration of time between inspections.
    :param size: Number of observations to generate.
    :param entry_rv: A discrete distribution from scipy.stats
    describing the number of periods that complete before an item enters observation
    (e.g. 0 if enters at time 0, 1 if enters after 1 period is completed and misses
    the first inspection etc.)
    :param sojourn_rv: A discrete distribution from scipy.stats describing how many complete periods
    an item remains under observation after entry (e.g. 2 if it will be inspected twice before
    failure is right-censored if not yet revealed).
    :return Pandas DataFrame where the index gives the observation number and
     columns 'tmin' and 'tmax' contain the corresponding lower
     and upper bounds of the interval in which failure occurred (upper bound
     is set to np.infty for right-censored observations).
    """

    # Sample failure time, entry period and sojourn periods.
    ttf = ttf_rv.rvs(size=size)
    if entry_rv :
        entry_at_start_period = entry_rv.rvs(size=size)
    else:
        entry_at_start_period = np.zeros(size)

    if sojourn_rv:
        sojourn_periods = sojourn_rv.rvs(size=size)
    else:
        sojourn_periods = np.ones(size) * np.inf

    # Compute the period at which an observation exits and fails.
    # Assume first period is period 0, second is period 1 etc.
    exit_at_start_period = entry_at_start_period + sojourn_periods
    failed_in_period = np.where(np.mod(ttf, inspection_interval) != 0, np.floor(ttf / inspection_interval),
                                (ttf / inspection_interval)-1)  # Period includes end point not start point.

    # Calculate failure interval bounds.
    tmin = np.where(failed_in_period <= entry_at_start_period, 0, np.minimum(failed_in_period,
                                                                 exit_at_start_period)) * inspection_interval

    tmax = np.where(failed_in_period < exit_at_start_period, np.maximum(failed_in_period, entry_at_start_period) + 1,
                     np.infty) * inspection_interval

    ds = pd.DataFrame({'tmin':tmin, 'tmax': tmax}, index=range(1,size+1))
    ds.index.name = "Observation #"

    return ds
