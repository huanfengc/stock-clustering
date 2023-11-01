import pandas as pd
import numpy as np



def load_original_data(path, year, dates, day):
    """Load the original data from a CSV file for a specific day."""
    file = path + "".join([year, dates[day], "minbar.csv"])
    return pd.read_csv(file)


def replace_zeros(df):
    """Replace zero values in a DataFrame with the nearest non-zero value."""
    df = df.replace(0, np.nan)
    df = df.ffill() # in case the last row contains zeros.
    df = df.bfill() # in case the first row contains zeros.
    df = df.replace(np.nan, 0)
    return df


def replace_nan(df):
    """Replace NaN values in a DataFrame with the nearest non-NaN value."""
    df = df.ffill() # in case the last row contains zeros.
    df = df.bfill() # in case the first row contains zeros.
    return df
