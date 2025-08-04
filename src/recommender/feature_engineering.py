from collections import defaultdict
from typing import Dict, List

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _material_value(board: chess.Board) -> int:
    """Return simple material balance (white – black) using PIECE_VALUES."""
    total = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        total += sign * PIECE_VALUES[piece.piece_type]
    return total


def extract_style_features(games_df: pd.DataFrame) -> pd.DataFrame:
    score_map = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "½-½": 0.5}
    recs = []

    for _, row in tqdm(
        games_df.iterrows(), total=len(games_df), desc="Extracting style features"
    ):

        moves = row.get("moves", [])
        if isinstance(moves, np.ndarray):
            moves = moves.tolist()
        elif moves is None:
            moves = []

        board = chess.Board()
        trades = checks = sacrifice_count = 0
        first_q = castled_ply = queen_trade_ply = None
        dev_first_plys = []
        prev_material = _material_value(board)

        for ply, uci in enumerate(moves, start=1):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                # malformed UCI (rare) – skip
                continue

            # skip moves that aren’t legal in the current position
            if not board.is_legal(move):
                continue

            # capture count
            if board.is_capture(move):
                trades += 1

            piece = board.piece_at(move.from_square)
            board.push(move)

            # queen movement
            if first_q is None and piece and piece.piece_type == chess.QUEEN:
                first_q = ply

            # castling detection
            if castled_ply is None and board.castling_rights == 0:
                castled_ply = ply

            # checks
            if board.is_check():
                checks += 1

            # queen trade detection
            if queen_trade_ply is None and (
                board.pieces(chess.QUEEN, chess.WHITE) == 0
                and board.pieces(chess.QUEEN, chess.BLACK) == 0
            ):
                queen_trade_ply = ply

            # simple sacrifice heuristic
            curr_material = _material_value(board)
            if abs(curr_material - prev_material) <= -3:
                sacrifice_count += 1
            prev_material = curr_material

        ply_count = len(moves)
        endgame_reached = ply_count >= 80 or len(board.piece_map()) <= 10

        recs.append(
            {
                **row.to_dict(),
                "ply_count": ply_count,
                "avg_trades": trades,
                "first_queen_ply": first_q or ply_count + 1,
                "castled_early": bool(castled_ply and castled_ply <= 20),
                "checks": checks,
                "result_score": score_map.get(row.get("result", ""), 0.0),
                "sacrifice_count": sacrifice_count,
                "queen_traded_early": bool(queen_trade_ply and queen_trade_ply <= 40),
                "endgame_reached": endgame_reached,
            }
        )

    return pd.DataFrame(recs)


def summarize_player_features(features_df: pd.DataFrame) -> pd.Series:
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
        "queen_trade_freq": features_df["queen_traded_early"].mean(),
        "pct_games_endgame": features_df["endgame_reached"].mean(),
        "opening_variety": features_df["eco"].nunique() / len(features_df),
    }
    return pd.Series(summary, dtype="float32")


def build_elite_style_vectors(elite_games_df: pd.DataFrame) -> pd.DataFrame:

    required_cols = {"white", "black", "ply_count"}
    if not required_cols.issubset(elite_games_df.columns):
        raise ValueError("elite_games_df missing required columns")

    # treat each game twice: once from White perspective, once from Black
    white_df = elite_games_df.copy()
    white_df["player"] = white_df["white"]
    black_df = elite_games_df.copy()
    black_df["player"] = black_df["black"]

    all_games = pd.concat([white_df, black_df], ignore_index=True)

    style_vectors = all_games.groupby("player", sort=False).apply(
        summarize_player_features
    )
    style_vectors.index.name = None
    return style_vectors.reset_index().rename(columns={"index": "player"})
