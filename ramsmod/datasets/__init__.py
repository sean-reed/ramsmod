import pandas as pd
from pathlib import Path
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def load_right_censored_data():
    """ Returns a set of right-censored failure data comprising of 26 observations as a
    Pandas Dataframe with index of the observation number, column 't' giving
    the survival time and column 'd' an indicator variable where 1 means
    the an exact failure time was observed and 0 indicating failure was
    right-censored.  """
    pathname = os.path.join(__location__, 'right-censored.csv')
    return pd.read_csv(pathname, index_col='Observation #')


def load_right_censored_data2():
    """ Returns a set of right-censored failure data comprising of 22 observations as a
    Pandas Dataframe with index of the observation number, column 't' giving
    the survival time and column 'd' an indicator variable where 1 means
    the an exact failure time was observed and 0 indicating failure was
    right-censored.  """
    pathname = os.path.join(__location__, 'right-censored-2.csv')
    return pd.read_csv(pathname, index_col='Observation #')


def load_interval_censored_data():
    """ Returns a set of interval-censored failure data comprising of 26 observations as a
    Pandas Dataframe with index of the observation number, column 'tmin' giving
    the lower bound of the failure time interval and column 'tmax' giving the upper bound of the
     failure time interval."""
    pathname = os.path.join(__location__, 'interval-censored.csv')
    return pd.read_csv(pathname, index_col='Observation #')


def load_interval_censored_data2():
    """ Returns a set of interval-censored failure data comprising of 24 observations as a
    Pandas Dataframe with index of the observation number, column 'tmin' giving
    the lower bound of the failure time interval and column 'tmax' giving the upper bound of the
     failure time interval."""
    pathname = os.path.join(__location__, 'interval-censored-2.csv')
    return pd.read_csv(pathname, index_col='Observation #')


def load_exponential_data():
    """ Returns a set of right-censored failure data comprising of 100 observations as a
    Pandas Dataframe with index of the observation number, column 't' giving
    the survival time and column 'd' an indicator variable where 1 means
    the an exact failure time was observed and 0 indicating failure was
    right-censored. This data was generated from an exponential distribution
    with a scale parameter value of 300 and uniform random right-censoring on the
    interval (0, 400). In total, 48 failure times were observed exactly and 52
    were right-censored."""
    pathname = os.path.join(__location__, 'exponential_data.csv')
    return pd.read_csv(pathname, index_col='Observation #')


def load_weibull_data():
    """ Returns a set of right-censored failure data comprising of 100 observations as a
    Pandas Dataframe with index of the observation number, column 't' giving
    the survival time and column 'd' an indicator variable where 1 means
    the an exact failure time was observed and 0 indicating failure was
    right-censored. This data was generated from a Weibull distribution
    with a shape parameter of 2 and a scale parameter value of 250, with
    uniform random right-censoring on the interval (0, 400). In total, 52 failure
    times were observed exactly and 48 were right-censored."""
    pathname = os.path.join(__location__, 'weibull_data.csv')
    return pd.read_csv(pathname, index_col='Observation #')


def load_lognormal_data():
    """ Returns a set of right-censored failure data comprising of 100 observations as a
    Pandas Dataframe with index of the observation number, column 't' giving
    the survival time and column 'd' an indicator variable where 1 means
    the an exact failure time was observed and 0 indicating failure was
    right-censored. This data was generated from an log-normal distribution
    with a shape parameter of 0.5 and a scale parameter value of 170, with
    uniform random right-censoring on the interval (0,400). In total, 51 failure
    times were observed exactly and 49 were right-censored."""
    pathname = os.path.join(__location__, 'lognormal_data.csv')
    return pd.read_csv(pathname, index_col='Observation #')