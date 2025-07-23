"""
Chess Opening Recommender — Feature Engineering Module

Overview:
This module provides functions to transform raw game DataFrames into
quantitative style features.

Objective:
Compute per-game and per-player metrics that capture playing style traits
(e.g., game length, trade frequency, queen deployment, castling behavior,
tactical aggression, and result-based performance).

Purpose:
- Encapsulate feature extraction logic for reuse in analysis scripts or an API.
- Ensure consistent, well-documented style metrics across users and reference
datasets.
"""

import chess
import pandas as pd
from tqdm import tqdm


def extract_style_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns:
      - 'moves' (list of UCI strings or iterable)
      - 'result' (string: '1-0', '0-1', '1/2-1/2')
    Compute per-game style features:
      - ply_count: total number of plies (half-moves)
      - avg_trades: number of captures per game
      - first_queen_ply: ply index of first queen move (or ply_count+1
      if never moved)
      - castled_early: bool, True if castled by ply 20
      - checks: number of checks delivered
      - result_score: numeric game result (1.0=win, 0.5=draw, 0.0=loss)
    Returns:
      A new DataFrame with these features appended to the original columns.
    """
    records = []
    score_map = {"1-0": 1.0, "1/2-1/2": 0.5, "½-½": 0.5, "0-1": 0.0}

    for _, row in tqdm(
        games_df.iterrows(), total=len(games_df), desc="Extracting features"
    ):
        # Coerce moves to list
        raw_moves = row["moves"]
        try:
            moves = list(raw_moves)
        except TypeError:
            moves = []
        result = row.get("result", "")
        board = chess.Board()

        trades = 0
        checks = 0
        first_queen = None
        castled_ply = None

        for ply, uci in enumerate(moves, start=1):
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                continue
            if board.is_capture(move):
                trades += 1
            board.push(move)
            # queen deployment
            if first_queen is None:
                piece = board.piece_at(move.to_square)
                if piece and piece.piece_type == chess.QUEEN:
                    first_queen = ply
            # detect castling by loss of castling rights
            if castled_ply is None:
                if (
                    not board.has_kingside_castling_rights(chess.WHITE)
                    and not board.has_queenside_castling_rights(chess.WHITE)
                    and not board.has_kingside_castling_rights(chess.BLACK)
                    and not board.has_queenside_castling_rights(chess.BLACK)
                ):
                    castled_ply = ply
            # checks
            if board.is_check():
                checks += 1

        ply_count = len(moves)
        first_q = first_queen or (ply_count + 1)
        castled_early = bool(castled_ply and castled_ply <= 20)
        result_score = score_map.get(result, 0.0)

        rec = row.to_dict()
        rec.update(
            {
                "ply_count": ply_count,
                "avg_trades": trades,
                "first_queen_ply": first_q,
                "castled_early": castled_early,
                "checks": checks,
                "result_score": result_score,
            }
        )
        records.append(rec)

    return pd.DataFrame(records)


def summarize_player_features(features_df: pd.DataFrame) -> pd.Series:
    """
    Aggregate per-game feature DataFrame into a single style vector
    (mean of each numeric feature).
    Returns a pandas Series keyed by feature name.
    """
    summary = {
        "avg_moves": features_df["ply_count"].mean(),
        "pct_long_games": (features_df["ply_count"] > 80).mean(),
        "avg_trades": features_df["avg_trades"].mean(),
        "avg_queen_move": features_df["first_queen_ply"].mean(),
        "pct_castled_early": features_df["castled_early"].mean(),
        "avg_checks": features_df["checks"].mean(),
        "win_rate": features_df["result_score"].mean(),
        "pct_wins": (features_df["result_score"] == 1.0).mean(),
        "pct_draws": (features_df["result_score"] == 0.5).mean(),
        "pct_losses": (features_df["result_score"] == 0.0).mean(),
    }
    return pd.Series(summary)
