import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

"""
Overview:
This module is for matching chess player style vectors
and clustering them for my opening recommender.
You give it per-player style vectors (numeric summaries of how someone plays)
and it lets you:

  1. cluster_styles: group the reference (elite) players into style clusters
  using StandardScaler + KMeans.
  2. find_style_neighbors: get the closest elite peers to any user
  by euclidean distance in style space.

Pipeline Usage:
After I extract style features and summarize them per player, I use this to:
  - Cluster the reference set to see the global style landscape
  - Find the nearest elite neighbors for any user

Functions:
cluster_styles(
    style_vectors: pd.DataFrame, n_clusters: int = 4, random_state: int = 42)
    - Takes a dataframe with one row per player
        (columns: 'player', features...)
    - Drops the 'player' column, scales the features, clusters with kmeans
    - Returns: dataframe with 'cluster' column, the scaler,
        and the kmeans model

find_style_neighbors(
    user_vector: pd.Series,
    style_vectors: pd.DataFrame,
    scaler: StandardScaler,
    top_n: int = 5
) -> pd.DataFrame
    - Takes a user's style vector, the reference style_vectors, and the scaler
    - Drops 'player' and 'cluster', scales everything,
        computes euclidean distance
    - Returns: dataframe with ['player', 'distance']
        for the top_n closest peers

I put this in a module so I can use it in the api, cli,
    or anywhere else, not just notebooks.
"""


# cluster the style vectors and return clusters, scaler, and kmeans
def cluster_styles(
    style_vectors: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
):
    feats = style_vectors.drop(columns=["player"], errors="ignore")
    scaler = StandardScaler().fit(feats)
    X = scaler.transform(feats)
    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    df = style_vectors.copy()
    df["cluster"] = km.labels_
    return df, scaler, km


# find the closest style neighbors for a user
def find_style_neighbors(
    user_vector: pd.Series,
    style_vectors: pd.DataFrame,
    scaler: StandardScaler,
    top_n: int = 5,
) -> pd.DataFrame:
    feats = style_vectors.drop(columns=["player", "cluster"], errors="ignore")
    user_arr = scaler.transform([user_vector.values])
    peer_arr = scaler.transform(feats)
    dists = pairwise_distances(user_arr, peer_arr, metric="euclidean")[0]
    out = style_vectors[["player"]].copy()
    out["distance"] = dists
    return out.nsmallest(top_n, "distance").reset_index(drop=True)
