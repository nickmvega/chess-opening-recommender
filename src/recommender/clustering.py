import pandas as pd

"""
Overview:
This module is for matching chess player style vectors and clustering them for my opening recommender.
You give it per-player style vectors (numeric summaries of how someone plays) and it lets you:

  1. cluster_styles: group the reference (elite) players into style clusters using StandardScaler + KMeans.
  2. find_style_neighbors: get the closest elite peers to any user by euclidean distance in style space.

Pipeline Usage:
After I extract style features and summarize them per player, I use this to:
  - Cluster the reference set to see the global style landscape
  - Find the nearest elite neighbors for any user

Functions:
cluster_styles(style_vectors: pd.DataFrame, n_clusters, random_state: int = 42)
    - Takes a dataframe with one row per player (columns: 'player', features...)
    - Drops the 'player' column, scales the features, clusters with kmeans
    - Returns: dataframe with 'cluster' column, the scaler, and the kmeans model

find_style_neighbors(
    user_vector: pd.Series,
    style_vectors: pd.DataFrame,
    scaler: StandardScaler,
    top_n: int = 5
) -> pd.DataFrame
    - Takes a user's style vector, the reference style_vectors, and the scaler
    - Drops 'player' and 'cluster', scales everything, computes euclidean distance
    - Returns: dataframe with ['player', 'distance'] for the top_n closest peers

I put this in a module so I can use it in the api, cli, or anywhere else, not just notebooks.
"""
import numpy as np
import pandas as pd


def _scale_fit(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0) + 1e-9
    return mean, std, (X - mean) / std


def _kmeans_lloyd(X: np.ndarray, n_clusters: int, n_iter: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_clusters, replace=False)
    cent = X[idx]  # initial centroids

    for _ in range(n_iter):
        # assign points → nearest centroid
        labels = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1).argmin(1)
        # recompute centroids
        new_cent = np.vstack([X[labels == k].mean(0) for k in range(n_clusters)])
        if np.allclose(new_cent, cent):
            break
        cent = new_cent
    return labels, cent


def cluster_styles_np(style_vectors: pd.DataFrame, n_clusters: int, seed: int = 42):
    feats_df = style_vectors.drop(columns="player", errors="ignore")
    mean, std, Xs = _scale_fit(feats_df.to_numpy(dtype=np.float32))
    labels, cent = _kmeans_lloyd(Xs, n_clusters, seed=seed)

    df = style_vectors.copy()
    df["cluster"] = labels
    scaler = {"mean": mean, "std": std}
    return df, scaler, cent


def _scale_transform(arr: np.ndarray, scaler: dict):
    return (arr - scaler["mean"]) / scaler["std"]


def find_style_neighbors_np(
    user_vector: pd.Series, style_vectors: pd.DataFrame, scaler: dict, top_n: int = 5
) -> pd.DataFrame:
    feats = style_vectors.drop(columns=["player", "cluster"], errors="ignore")
    user = _scale_transform(user_vector.to_numpy(dtype=np.float32), scaler)
    peers = _scale_transform(feats.to_numpy(dtype=np.float32), scaler)

    dists = ((peers - user) ** 2).sum(1) ** 0.5
    out = style_vectors[["player"]].copy()
    out["distance"] = dists
    return out.nsmallest(top_n, "distance").reset_index(drop=True)
