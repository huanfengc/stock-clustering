import pandas as pd
import numpy as np



def compute_twap(df):
    """Compute the Time-Weighted Average Price (TWAP)."""
    df["twap"] = (df["open"] + df["close"]) / 2
    return df


def compute_vwap(df):
    """Compute the Volume-Weighted Average Price (VWAP)."""
    df["vwap"] = df["turnover"] / df["volume"]
    return df


def compute_volatility(df):
    """Compute the price volatility."""
    df["volatility"] = (df["high"] - df["low"]) / df["open"]
    return df


def compute_return(df, choice=0):
    """Compute the stock returns."""
    if choice == 0:
        df["return"] = np.log(df["close"] / df["pre_close"])
    elif choice == 1:
        df["return"] = np.divide((df["close"] - df["pre_close"]), df["pre_close"])
    else:
        return "Please select either log return (0) or normal return (1)!"
    return df


def compute_moving_average(df):
    """Compute the 5-minute simple moving average of close prices for each stock."""
    df['5_min_SMA'] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(window=5).mean())
    return df


def compute_bollinger_band(df):
    """Compute the Bollinger Bands and bandwidth percentage for each stock."""
    df["20_min_SMA"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(window=20).mean())
    df["moving_std"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(window=20).std())
    df["upper_band"] = df["20_min_SMA"] + (df["moving_std"] * 2)
    df["lower_band"] = df["20_min_SMA"] - (df["moving_std"] * 2)
    df["PBW"] = (df["upper_band"] - df["lower_band"]) / df["20_min_SMA"]

    df = df.drop(["moving_std"], axis=1)
    df = df.drop(["upper_band"], axis=1)
    df = df.drop(["lower_band"], axis=1)
    return df


def compute_rsi(df):
    """Compute the Relative Strength Index (RSI) for each stock."""
    def get_rsi(series, window):
        delta = series.diff().dropna()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df["RSI"] = df.groupby("symbol")["close"].transform(lambda x: get_rsi(x, 20))
    return df
    