"""
Overview:
This module handles all logic for recommending openings
    based on peer performance.
It filters games by peer list and color,
    computes performance statistics for each
opening, and selects the top openings by a weighted score.

Pipeline Usage:
1. After identifying a set of stylistic peers (`peer_list`) and
    loading a DataFrame
   of reference games (`elite_df`), call `compute_opening_stats` separately for
   White and Black.
2. Pass the resulting stats DataFrames to `recommend_openings` to extract the
   top‐N openings for each side.

Functions:
get_peer_games(elite_df, peer_list) -> pd.DataFrame
    Returns the subset of `elite_df` where either
      White or Black is in `peer_list`.

compute_opening_stats(
    games_df: pd.DataFrame,
    peer_list: list[str],
    color: str = 'white',
    min_games: int = 0
) -> pd.DataFrame
    Filters `games_df` to only peer games of the given `color`, then groups by
    opening (ECO code + name) to compute:
      • games_played  — total games played in that opening
      • wins          — count of wins by that color
      • draws         — count of drawn games
      • score_pct     — normalized performance: (wins + 0.5 × draws) /
        games_played
      • weight        —
      `score_pct` weighted by sample size: score_pct ×
        log10(games_played + 1)
    Returns a DataFrame sorted descending by `weight` and `games_played`.  If
    `min_games` > 0, openings with fewer games are filtered out.

recommend_openings(
    white_stats: pd.DataFrame,
    black_stats: pd.DataFrame,
    top_n: int = 5
) -> tuple[list, list]
    Extracts the top `top_n` openings from each stats DataFrame as lists of
    tuples `(eco, opening, games_played, score_pct)`.
"""

import numpy as np
import pandas as pd


def get_peer_games(elite_df: pd.DataFrame, peer_list: list[str]) -> pd.DataFrame:
    mask = elite_df["white"].isin(peer_list) | elite_df["black"].isin(peer_list)
    return elite_df[mask].copy()


def compute_opening_stats(
    games_df: pd.DataFrame,
    peer_list: list[str],
    color: str = "white",
    min_games: int = 0,
) -> pd.DataFrame:
    if color not in ("white", "black"):
        raise ValueError("color must be 'white' or 'black'")

    # Filter to peer games for the given color
    df = games_df[games_df[color].isin(peer_list)].copy()
    win_str = "1-0" if color == "white" else "0-1"

    stats = (
        df.groupby(["eco", "opening"])
        .agg(
            games_played=("result", "size"),
            wins=("result", lambda s: (s == win_str).sum()),
            draws=("result", lambda s: (s == "1/2-1/2").sum()),
        )
        .reset_index()
    )
    stats["score_pct"] = (stats["wins"] + 0.5 * stats["draws"]) / stats["games_played"]
    stats["weight"] = stats["score_pct"] * np.log10(stats["games_played"] + 1)

    if min_games > 0:
        stats = stats[stats["games_played"] >= min_games]

    return stats.sort_values(["weight", "games_played"], ascending=False).reset_index(
        drop=True
    )


def recommend_openings(
    white_stats: pd.DataFrame, black_stats: pd.DataFrame, top_n: int = 5
) -> tuple[list, list]:
    white_recs = white_stats.head(top_n)[
        ["eco", "opening", "games_played", "score_pct"]
    ].values.tolist()
    black_recs = black_stats.head(top_n)[
        ["eco", "opening", "games_played", "score_pct"]
    ].values.tolist()
    return white_recs, black_recs
