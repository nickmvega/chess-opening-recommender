import io
import os
from pathlib import Path
from typing import Optional

import chess.pgn
import pandas as pd
import requests
from tqdm import tqdm

"""
Chess Opening Recommender — Data Fetcher Module

Overview:
This module centralizes all data collection and parsing functions needed
for the Chess Opening Recommender system. It handles fetching user games,
parsing PGN data into structured Pandas DataFrames, and loading reference (
elite) game datasets.

Objective:
Provide reusable, well-documented functions that can be imported into other
scripts or an API backend to retrieve and prepare chess game data for
feature extraction and recommendation phases.

Purpose:
- Decouple data-fetching logic from analysis notebooks.
- Enable easy integration into a web service or CLI tool.
- Ensure consistency and reliability by isolating API calls and parsing
routines.
"""

# Configuration
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN", "")
if not LICHESS_TOKEN:
    raise EnvironmentError("Please set the LICHESS_TOKEN")
HEADERS = {"Authorization": f"Bearer {LICHESS_TOKEN}"}


def fetch_user_games(
    username: str, max_games: int = 300, save_to: Optional[Path] = None
) -> str:
    """
    Fetch up to `max_games` recent games for a Lichess user in PGN format
    (with moves, evals, ECO). Optionally save raw PGN to `save_to`.
    Returns the raw PGN text.
    """
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": max_games,
        "moves": "true",
        "evals": "true",
        "opening": "true",
        "clocks": "false",
        "format": "pgn",
    }
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    pgn_text = response.text
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(pgn_text, encoding="utf-8")
    return pgn_text


def pgn_to_games_df(pgn_text: str) -> pd.DataFrame:
    """
    Parse a multi-game PGN string into a DataFrame.
    Columns: white, black, result, eco, opening,
             utc_date, utc_time, time_control, moves (list), evals (list)
    """
    records = []
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        hdr = game.headers
        moves, evals = [], []
        node = game
        while node.variations:
            nxt = node.variation(0)
            moves.append(nxt.move.uci())
            if hasattr(nxt, "eval") and nxt.eval is not None:
                evals.append(nxt.eval)
            node = nxt
        records.append(
            {
                "white": hdr.get("White"),
                "black": hdr.get("Black"),
                "result": hdr.get("Result"),
                "eco": hdr.get("ECO"),
                "opening": hdr.get("Opening", ""),
                "utc_date": hdr.get("UTCDate"),
                "utc_time": hdr.get("UTCTime"),
                "time_control": hdr.get("TimeControl"),
                "moves": moves,
                "evals": evals,
            }
        )
    return pd.DataFrame(records)


def parse_user_pgn(pgn_text: str) -> pd.DataFrame:
    """
    Given raw PGN text of a user's games, return a DataFrame of parsed games.
    """
    return pgn_to_games_df(pgn_text)


def parse_elite_pgn_fast(pgn_path: Path, n_games: int = 500) -> pd.DataFrame:
    """
    Stream the first `n_games` from the elite PGN file at `pgn_path`,
    parse each game directly into dicts (headers + moves + evals),
    and return a DataFrame. Faster than re-serializing to PGN.
    """
    records = []
    with pgn_path.open(encoding="utf-8", errors="ignore") as fh:
        for _ in tqdm(range(n_games), desc="Parsing elite PGN"):
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            hdr = game.headers
            moves, evals = [], []
            node = game
            while node.variations:
                nxt = node.variation(0)
                moves.append(nxt.move.uci())
                if hasattr(nxt, "eval") and nxt.eval is not None:
                    evals.append(nxt.eval)
                node = nxt
            records.append(
                {
                    "white": hdr.get("White"),
                    "black": hdr.get("Black"),
                    "result": hdr.get("Result"),
                    "eco": hdr.get("ECO"),
                    "opening": hdr.get("Opening", ""),
                    "utc_date": hdr.get("UTCDate"),
                    "utc_time": hdr.get("UTCTime"),
                    "time_control": hdr.get("TimeControl"),
                    "moves": moves,
                    "evals": evals,
                }
            )
    return pd.DataFrame(records)


def get_user_profile(username: str) -> dict:
    """
    Fetch the Lichess user profile JSON for `username`.
    Returns the raw dict so you can extract ratings, play counts, etc.
    """
    url = f"https://lichess.org/api/user/{username}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()
