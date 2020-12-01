import numpy as np
import pandas as pd


def convert_to_pd_series(l):
    """Convert l to a Pandas series if it's a list or np.ndarray."""
    if isinstance(l, (list, np.ndarray)):
        l = pd.Series(l)
    return l