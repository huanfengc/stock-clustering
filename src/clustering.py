import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from config import SAVE_SCALED_DATA



def scale_data(df):
    """Scale the input DataFrame using the StandardScaler from scikit-learn."""
    scaler = StandardScaler()
    # scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data


def compute_distance_matrix(df):
   """Compute the Pearson correlation coefficient distance matrix of the input DataFrame."""
   scaled_data = scale_data(df)
   pcc_matrix = np.corrcoef(scaled_data, rowvar=True)
   distance_matrix = 1 - pcc_matrix
   np.fill_diagonal(distance_matrix, 0)
   return distance_matrix


def compute_cosine_similarity(df):
    """Compute the cosine similarity distance matrix of the input DataFrame."""
    scaled_data = scale_data(df)
    similarity_matrix = cosine_similarity(scaled_data)
    distance_matrix = 1 - similarity_matrix
    return distance_matrix


def delete_existing_results(path):
    """Deletes existing files in the specified directory."""
    files = glob.glob(path + "*")
    for f in files:
        os.remove(f)
    return

    
def agglomerative_clustering(dfs, optimal_clusters, metric, start, threshold):
    """
    Apply agglomerative clustering to the input DataFrame based on the specified parameters.

    Args:
        dfs: A list of DataFrames containing the muti-day data to be clustered.
        optimal_clusters: List of pre-selected number of clusters for each day.
        metric: The distance metric to use. "pcc" for Pearson correlation coefficient, "cosine" for cosine similarity.
        start: The start day, based on the WINDOW_SIZE and and DATES. It is handled automatically.
        threshold: The threshold to apply when forming clusters.

    Returns:
        A list of DataFrames for multiple days, each containing the clustering results for a day.
    """
    distance_matrix = None
    full_results = []
    save_scaled_data = SAVE_SCALED_DATA

    if type(save_scaled_data) != bool:
        save_scaled_data = True
        print("\n\033[91mYour input SAVE_SCALED_DATA is undefined! Scaled data is chosen.\033[0m\n")

    for i in range(len(dfs)):
        symbols = dfs[i]["symbol"]
        numerical_data = dfs[i].drop(columns=["symbol"])
        if metric == "pcc":
            distance_matrix = compute_distance_matrix(numerical_data)
        elif metric == "cosine":
            distance_matrix = compute_cosine_similarity(numerical_data)
        
        if threshold == 0:
            clustering = AgglomerativeClustering(n_clusters=optimal_clusters[i], affinity="precomputed", linkage="average")
        else:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, affinity="precomputed", linkage="average")
            
        labels = clustering.fit_predict(distance_matrix)
        clustering_result = pd.DataFrame({"symbol": symbols, "cluster": labels})

        if SAVE_SCALED_DATA:
            scaled_data = pd.DataFrame(scale_data(numerical_data), columns=numerical_data.columns)
            scaled_data["symbol"] = symbols
            full_result = pd.merge(clustering_result, scaled_data, on="symbol")
        else:
            original_data = pd.DataFrame(numerical_data, columns=numerical_data.columns)
            original_data["symbol"] = symbols
            full_result = pd.merge(clustering_result, original_data, on="symbol")

        full_results.append(full_result)
    return full_results


def save_result(dfs, year, dates, start):
    """Saves the clustering results to CSV files."""
    result_path = "../results/"
    # Delete existing results to avoid confusion, especially when window_size is applied.
    # Comment the following line to keep the existing results.
    delete_existing_results(result_path)
    for i in range(len(dfs)):
        file_to_save = f"{result_path}{year}{dates[start + i]}_clustering.csv"
        dfs[i].to_csv(file_to_save, index=False)
    return


def find_optimal_clusters(df):
    """
    Find the optimal number of clusters for agglomerative clustering using the elbow method and silhouette scores.

    Args:
        df: A DataFrame containing the data to be analyzed.

    Returns:
        None, but to generate Elbow and Silhouette scores plots.
    """
    distortions = []
    silhouette_scores = []
    max_clusters = 20
    symbols = df["symbol"]
    numerical_data = df.drop(columns=["symbol"])
    distance_matrix = compute_distance_matrix(numerical_data)

    for i in range(2, max_clusters+1):
        clustering = AgglomerativeClustering(n_clusters=i, affinity="precomputed", linkage="average")
        clustering.fit(distance_matrix)
        labels = clustering.labels_

        distortion = 0
        for j in range(i):
            cluster_points = distance_matrix[labels == j]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                distortion += np.sum(np.square(euclidean_distances(cluster_points, center.reshape(1, -1))))
        distortions.append(distortion)

        silhouette_scores.append(silhouette_score(distance_matrix, labels))
        
    plot_elbow_silhouette(max_clusters, distortions, silhouette_scores)
    return
