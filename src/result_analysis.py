import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
from config import YEAR, DATES, WINDOW_SIZE, INPUT_FEATURES



def get_clustering_results(start, dates):
    """Retrieve clustering results for given dates."""
    results = [None for _ in range(len(dates) - start + 1)]
    for i in range(start - 1, len(dates)):
        file_to_read = f"../results/{year}{dates[i]}_clustering.csv"
        results[i - start + 1] = pd.read_csv(file_to_read)
    return results

    
def match_clusters(cluster_stocks_day1, df_day2):
    """Find the best match of a cluster from day 1 in the clustering results of day 2."""
    max_overlap = 0
    best_match = None
    for cluster in df_day2["cluster"].unique():
        cluster_stocks_day2 = set(df_day2[df_day2["cluster"] == cluster]["symbol"])
        overlap = len(cluster_stocks_day1.intersection(cluster_stocks_day2))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = cluster
    if best_match is None:
        print(best_match, cluster, max_overlap)
        
    return best_match, max_overlap / len(cluster_stocks_day1)


def clustering_results_variations(results, dates, start):
    """Analyze variations in clustering results across consecutive days."""
    for i in range(1, len(results)):
        matched_pairs = []
        overlap_percentages = []
        for cluster in results[i - 1]["cluster"].unique():
            prev_day_cluster_stocks = set(results[i - 1][results[i - 1]["cluster"] == cluster]["symbol"])
            matched_cluster, overlap = match_clusters(prev_day_cluster_stocks, results[i])    
            matched_pairs.append((cluster, matched_cluster, overlap))

        matched_pairs.sort(key=lambda x: x[1])
        print("\n======================================================================================\n")
        print(f"\033[91mVariations of constituent stocks between matched clusters in day {dates[start + i - 2]} and day {dates[start + i - 1]}:\033[0m \n")
        print("Matched clusters | Number of constituent stocks | Overlap percentage \n")
        for j in range(len(matched_pairs)):
            c1, c2, percent = matched_pairs[j]
            n1 = len(results[i - 1][results[i - 1]["cluster"] == c1]["symbol"].unique())
            n2 = len(results[i][results[i]["cluster"] == c2]["symbol"].unique())
            l1 = "[{}, {}]".format(c1, c2)
            l2 = "[{}, {}]".format(n1, n2)
            print("{:<20}{:<30}{}".format(l1, l2, percent * 100))
        print(f"\nAverage overlap percentage: {np.mean([val[2] for val in matched_pairs]) * 100}% \n")
    return



if __name__ == "__main__":

    year, dates, window_size = YEAR, DATES, WINDOW_SIZE
    analysis_choice = {
        "0": clustering_results_variations,
    }
    try:
        results = get_clustering_results(window_size, dates)
    except FileNotFoundError:
        print("""\n\033[91mNo results is available!\n\nPlease set up config.py and then run 'python3 main.py' to genereate the stock clustering results!\033[0m\n""")
        sys.exit()

    which_analysis = input("""\nSelect the analysis that you want to check,\n
    0: consistency of the constituents of the the cluster between two consecutive days.\n
    1: statistics, i.e. mean and standard deviation of each feature in each cluster.\n
    2: pair plot of the clustering results.\n
    3: box plot of the clustering results.\n
    Please input 0, 1, 2 or 3 : """)

    which_analysis = int(which_analysis)
    input_features = INPUT_FEATURES
    
    if which_analysis == 0:
        clustering_results_variations(results, dates, window_size)
    elif which_analysis == 1:
        print("\033[91m= = = = = = = = = = = = = = = = = = = = = = = = =\n")
        print("Statistics of each feature in each cluster\033[0m\n")
        print(len(results))
        for i in range(len(results)):
            clustered_data = results[i].groupby("cluster")
            mean_values = clustered_data.mean()
            std_values = clustered_data.std()
            date = dates[window_size - 1 + i]
            print(" ")
            print(f"\033[91mDate {date} clustered result statistics:\033[0m\n")
            print("Mean values:\n", mean_values)
            print("- - -\n")
            print("Standard deviations:\n", std_values)
            print(" ")
            print("= = = = = = = = = = = = = = = = = = = = = = = = =\n")
    elif which_analysis == 2:
        which_day = input(f"""\nSelect the day that you want to check, there are {len(results)} days in total.\n
        Please input any number from 1 to {len(results)}: """)
        which_day = int(which_day)
        if which_day < 1 or which_day > len(results):
            print(f"\nPlease select a day within the range 1 ~ {len(results)} !\n")
            sys.exit()

        results[which_day - 1]["cluster"] = results[which_day - 1]["cluster"].astype("category")
        sns.pairplot(results[which_day - 1], hue="cluster")
        plt.suptitle(f"Date: {dates[window_size - 1 + which_day - 1]}")
        plt.show()
    elif which_analysis == 3:
        which_day = input(f"""\nSelect the day that you want to check, there are {len(results)} days in total.\n
        Please input any number from 1 to {len(results)}: """)
        which_day = int(which_day)
        if which_day < 1 or which_day > len(results):
            print(f"\nPlease select a day within the range 1 ~ {len(results)} !\n")
            sys.exit()

        input_features = [f for f in input_features if f != "symbol"]
        n_features = len(input_features)
        n_cols = 2
        n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        for i, feature in enumerate(input_features):
            row = i // n_cols
            col = i % n_cols
            sns.boxplot(data=results[which_day - 1], x="cluster", y=feature, ax=axes[row, col])
            axes[row, col].set_title(f"Distribution of {feature} across all clusters")

        if n_features % 2 != 0:
            fig.delaxes(axes[n_rows - 1, n_cols - 1])

        plt.tight_layout()
        plt.show()
    else:
        print("\n Your choice is not available, please make sure input 0, 1, 2 or 3 to select a valid analysis!")
        sys.exit()
