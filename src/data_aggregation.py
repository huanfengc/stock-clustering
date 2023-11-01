import pandas as pd
import numpy as np
from scipy.stats import zscore
from config import Z_SCORE



def aggregate_daily_data(df, days, features):
    """Aggregates daily data by computing the mean and standard deviation for specified features."""
    df_ave_std = [None for _ in range(days)]
    feature_dict = {feature: ["mean", "std"] for feature in features if feature != "symbol"}

    for i in range(days):
        df_ave_std[i] = df[i].groupby("symbol").agg(feature_dict).reset_index()
        df_ave_std[i].columns = ["_".join(col).strip() for col in df_ave_std[i].columns.values]
        df_ave_std[i] = df_ave_std[i].rename(columns={"symbol_": "symbol"})
    return df_ave_std


def select_input_features(df, features):
    """Selects specified features from input DataFrames."""
    df_input = [None for _ in range(len(df))]
    for i in range(len(df)):
        df_input[i] = df[i][features]
    return df_input


def combine_and_average(df1, df2):
    """Combines two DataFrames based on the "symbol" column and averages the corresponding numerical features."""
    combined = pd.merge(df1, df2, on="symbol", how="outer", suffixes=("", "_2"))
    for col in df1.columns:
        if col != "symbol":
            combined[col] = combined[col].add(combined[f"{col}_2"], fill_value=0)
            combined.drop(f"{col}_2", axis=1, inplace=True)
    return combined


def combine_two_days_as_one(dfs, take_average):
    """
    Combines dataFrames from two consecutive days into a single DataFrame.

    Args:
        dfs: A list of DataFrames, each representing a day's data.
        take_average: If True, average the overlapping features. If False, keep them separate.

    Returns:
        A list of DataFrames, each containing combined data for the latter day of two consecutive days.
    """
    df_combined = []
    for i in range(len(dfs) - 1):
        combined = pd.merge(dfs[i], dfs[i + 1], on="symbol", how="outer", suffixes=(f"_day{i+1}", f"_day{i+2}"))
        combined = replace_nan(combined)
        
        if take_average:
            for col in df[0].columns:
                if col != "symbol":
                    combined[col] = combined[[f"{col}_day{i+1}", f"{col}_day{i+2}"]].mean(axis=1)
                    combined.drop([f"{col}_day{i+1}", f"{col}_day{i+2}"], axis=1, inplace=True)
        df_combined.append(combined)
    return df_combined


def remove_outlier_stocks(df):
    """
    Removes rows in the DataFrame that have features with z-scores above a certain threshold.

    Args:
        df: The DataFrame from which outliers are to be removed.

    Returns:
        The DataFrame with outlier rows removed.
    """
    z_score_threshold = Z_SCORE
    features = [feature for feature in df.columns if feature != "symbol"]
    z_scores = zscore(df[features])
    # print(z_scores)
    abs_z_scores = abs(z_scores)
    # outliers = (abs_z_scores > z_score_threshold).all(axis=1)
    outliers = (abs_z_scores > z_score_threshold).any(axis=1)
    df_no_outliers = df[~outliers]
    return df_no_outliers
