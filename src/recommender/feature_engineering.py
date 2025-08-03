"""
Overview:
This module transforms raw chess game data into quantitative “style” features
and aggregates them into per‐player style vectors. It powers Phase 2 of the
pipeline—feature engineering—by exposing:

  • extract_style_features: computes per‐game metrics from a DataFrame of games  
  • summarize_player_features: reduces per‐game metrics to a single per‐player vector  
  • build_elite_style_vectors: applies the above to an entire reference set of
    elite games, producing one vector per elite player  

Pipeline Usage:
1. Ingest games via `data_fetcher.parse_user_pgn` or `parse_elite_pgn_fast`.  
2. Call `extract_style_features(games_df)` to compute per‐game features.  
3. Call `summarize_player_features(features_df)` to collapse a user’s games into
   a single style vector.  
4. (Reference only) Use `build_elite_style_vectors(elite_games_df)` to prepare
   the style database of elite players.

Functions:
extract_style_features(games_df: pd.DataFrame) -> pd.DataFrame  
    Given a DataFrame of games with columns:
      - 'moves' (list of UCI strings)  
      - 'result' (string: '1-0', '0-1', '1/2-1/2', '½-½')  
    Computes per‐game style metrics and returns a new DataFrame with these added:
      • ply_count          — total half‐moves  
      • avg_trades         — number of captures  
      • first_queen_ply    — ply index of first queen move  
      • castled_early      — True if castled by ply 20  
      • checks             — number of checks given  
      • result_score       — numeric result (1/0.5/0)  

summarize_player_features(features_df: pd.DataFrame) -> pd.Series  
    Aggregates a per‐game features DataFrame into a single style vector by
    taking means and proportions across games. Returns a pandas Series with:
      • avg_moves, pct_long_games, avg_trades, avg_queen_move,  
        pct_castled_early, avg_checks, win_rate, pct_wins, pct_draws, pct_losses  

build_elite_style_vectors(elite_games_df: pd.DataFrame) -> pd.DataFrame  
    Converts a DataFrame of elite games (with 'white' and 'black' columns plus
    per‐game features) into one style vector per elite player by:
      1. Tagging each row once with white as player and once with black as player  
      2. Concatenating and grouping by 'player'  
      3. Applying `summarize_player_features` to each group  

Features:
This pipeline captures a variety of intuitive style dimensions:

  • Game Length & Endgame Tendency
    – ply_count, pct_long_games  

  • Material Exchanges  
    – avg_trades  

  • Queen Deployment
    – first_queen_ply  

  • King Safety
    – castled_early  

  • Tactical Aggression
    – checks  

  • Overall Performance
    – result_score → win_rate, pct_wins, pct_draws, pct_losses  

By combining these features, we numerically profile how each player prefers to
navigate the opening, middlegame tactics, and endgame transitions, enabling
style‐aware peer matching and personalized opening recommendations.
"""
import numpy as np
import pandas as pd
import chess
from tqdm import tqdm

def extract_style_features(games_df: pd.DataFrame) -> pd.DataFrame:
    score_map = {"1-0":1.0,"0-1":0.0,"1/2-1/2":0.5,"½-½":0.5}
    recs = []
    for _, row in tqdm(games_df.iterrows(), total=len(games_df), desc="Extracting features"):
        moves = row['moves'] or []
        result = row.get('result',"")
        board = chess.Board()
        trades=checks=0
        first_q=None
        castled_ply=None
        for ply, uci in enumerate(moves,1):
            try:
                mv=chess.Move.from_uci(uci)
            except:
                continue
            if board.is_capture(mv): trades+=1
            board.push(mv)
            if first_q is None:
                p=board.piece_at(mv.to_square)
                if p and p.piece_type==chess.QUEEN:
                    first_q=ply
            if castled_ply is None and not any(board.has_castling_rights(side) for side in [chess.WHITE, chess.BLACK]):
                castled_ply=ply
            if board.is_check(): checks+=1
        ply_count=len(moves)
        recs.append({
            **row.to_dict(),
            "ply_count":ply_count,
            "avg_trades":trades,
            "first_queen_ply":first_q or ply_count+1,
            "castled_early":bool(castled_ply and castled_ply<=20),
            "checks":checks,
            "result_score":score_map.get(result,0.0)
        })
    return pd.DataFrame(recs)

def summarize_player_features(features_df: pd.DataFrame) -> pd.Series:
    summary = {
        'avg_moves':         features_df['ply_count'].mean(),
        'pct_long_games':    (features_df['ply_count'] > 80).mean(),
        'avg_trades':        features_df['avg_trades'].mean(),
        'avg_queen_move':    features_df['first_queen_ply'].mean(),
        'pct_castled_early': features_df['castled_early'].mean(),
        'avg_checks':        features_df['checks'].mean(),
        'win_rate':          features_df['result_score'].mean(),
        'pct_wins':          (features_df['result_score'] == 1.0).mean(),
        'pct_draws':         (features_df['result_score'] == 0.5).mean(),
        'pct_losses':        (features_df['result_score'] == 0.0).mean(),
    }
    return pd.Series(summary)

def build_elite_style_vectors(elite_games_df: pd.DataFrame) -> pd.DataFrame:
    white_df = elite_games_df.copy()
    white_df['player'] = white_df['white']
    black_df = elite_games_df.copy()
    black_df['player'] = black_df['black']

    feature_cols = [
        'player',
        'ply_count',
        'avg_trades',
        'first_queen_ply',
        'castled_early',
        'checks',
        'result_score'
    ]
    all_games = pd.concat([
        white_df[feature_cols],
        black_df[feature_cols]
    ], ignore_index=True)

    style_vectors = all_games.groupby('player').apply(summarize_player_features)
    style_vectors = style_vectors.reset_index().rename(columns={'index':'player'})
    return style_vectors
