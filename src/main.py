import numpy as np
import sys
from data_loading import load_original_data, replace_zeros, replace_nan
from feature_construction import compute_return, compute_vwap, compute_volatility, compute_bollinger_band, compute_rsi
from data_aggregation import aggregate_daily_data, select_input_features, combine_and_average, remove_outlier_stocks
from clustering import agglomerative_clustering, save_result
from functools import reduce
from config import DATA_PATH, YEAR, DATES, INPUT_FEATURES, WINDOW_SIZE, DISTANCE_THRESHOLD



if __name__ == "__main__":
    print("\nThe program started, set up the parameters ... \n")

    data_path, year, dates = DATA_PATH, YEAR, DATES
    days = len(dates)

    print(">>> \n")
    print(f"Step 1 started: Loading the stock data from {year}{dates[0]} to {year}{dates[-1]} ({days} trading days) ...")

    try:
        df = [None for _ in range(days)]
        for i in range(days):
            df[i] = load_original_data(data_path, year, dates, i)
        for i in range(days):
            df[i] = replace_zeros(df[i])
    except FileNotFoundError:
        print("\n\033[91mNo data is found, please put the data (in csv format) into '/data/minbar' directory !\033[0m\n")
        sys.exit()
    
    print(f"Step 1 finished: Successfully loaded stock data from {year}{dates[0]} to {year}{dates[-1]} ({days} trading days). \n")
    print(">>> \n")
    print("Step 2 started: Constructing all available features ...")
    
    for i in range(days):
        df[i] = compute_vwap(df[i])
        df[i] = compute_volatility(df[i])
        df[i] = compute_return(df[i], choice=0) # choice = 0 for log return, 1 for normal return.
        df[i] = compute_bollinger_band(df[i])
        # df[i] = compute_rsi(df[i])
        # df[i]["time"] = df[i]["date"] * 10000 + df[i]["time"]

    features = ["return", "vwap", "volatility", "20_min_SMA", "PBW", "symbol"]
    for i in range(days):
        df[i] = df[i][features]
        df[i] = replace_nan(df[i])
    
    print("Step 2 finished: All availble features have been computed and stored.\n")
    print(">>> \n")
    print("Step 3 started: Aggregate the data with selected features and window size for combining.\n")

    df_aggregate = aggregate_daily_data(df, days, features)

    input_features = INPUT_FEATURES
    df_to_cluster = select_input_features(df_aggregate, input_features)

    print("Removing outliers, i.e. the data points whose features have abs(Z-score) > 3.\n")
    print("Please check the reduction of the rows (number of data points), adjust the Z-score in config.py .\n")
    df_remove_outlier = []
    for i in range(len(df_to_cluster)):
        print("Before removing outliers:", df_to_cluster[i].shape)
        df_remove_outlier.append(remove_outlier_stocks(df_to_cluster[i]))
        print("After removing outliers: ", df_remove_outlier[i].shape)
        print("- - -\n")
    df_to_cluster = df_remove_outlier

    # to_combine_dfs = True
    window_size = WINDOW_SIZE
    if 1 < window_size < len(dates):
        df_combined = [reduce(combine_and_average, df_to_cluster[i:i+window_size]) for i in range(len(df_to_cluster) - window_size + 1)]
        for i in range(len(df_combined)):
            numeric_cols = df_combined[i].select_dtypes(include=[np.number]).columns
            df_combined[i][numeric_cols] = df_combined[i][numeric_cols].div(window_size)
    
    print("Step 3 finished: Ready for stock clustering.\n")
    print(">>> \n")
    print("Step 4 started: Clustering the stocks ..." )

    optimal_clusters = [5] * days
    metric = "pcc"
    distance_threshold = DISTANCE_THRESHOLD
    results = None

    for i in range(len(df_to_cluster)):
        df_to_cluster[i] = replace_nan(df_to_cluster[i])
    if 1 < window_size < len(dates):
        results = agglomerative_clustering(df_combined, optimal_clusters[:-1], metric, window_size - 1, distance_threshold)
    else:
        results = agglomerative_clustering(df_to_cluster, optimal_clusters, metric, window_size - 1, distance_threshold)
    save_result(results, year, dates, window_size - 1)
    
    print("Step 4 finished: The clustering results have been saved in the '/result' directory.\n")
    print("- - -\n")
    print("Please run result_analysis.py to check the results and refer to README for more instructions. \n")
    print("- - -\n")
    print("Thank you for using the program!")
