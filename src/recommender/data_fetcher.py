import io
from pathlib import Path
from typing import Optional

import chess.pgn
import pandas as pd
import requests
from tqdm import tqdm
import os

"""
Overview:
Centralizes data collection and parsing routines
    for the chess opening recommender.
Handles fetching user games, parsing pgn, and loading reference datasets.

Pipeline Usage:
- Fetch a user's games from lichess api
- Parse pgn data into structured dataframes
- Load reference (elite) game datasets

Functions:
fetch_user_games(username: str, save_to: Optional[Path] = None) -> str
    - Fetches all available recent games for a lichess user in pgn format
    - Optionally saves raw pgn to `save_to`
    - Returns the raw pgn text

pgn_to_games_df(pgn_text: str) -> pd.DataFrame
    - Parses a multi-game pgn string into a dataframe with columns:
      white, black, result, eco, opening, utc_date, utc_time,
        time_control, moves (list), evals (list)

parse_user_pgn(pgn_text: str) -> pd.DataFrame
    - Parses raw user pgn into a dataframe

parse_elite_pgn_fast(pgn_path: Path, n_games: int = 500) -> pd.DataFrame
    - Streams the first `n_games` from the elite pgn file,
        parses headers+moves+evals, returns as dataframe

get_user_profile(username: str) -> dict
    - Fetches lichess user profile json
"""

# config
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")
HEADERS = {"Authorization": f"Bearer {LICHESS_TOKEN}"}


# fetch all available recent games for a lichess user in pgn format
def fetch_user_games(username: str, save_to: Optional[Path] = None) -> str:
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "moves": "true",
        "evals": "true",
        "opening": "true",
        "clocks": "false",
        "format": "pgn",
    }
    resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
    resp.raise_for_status()
    text = resp.text
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(text, encoding="utf-8")
    return text


# fetch lichess user profile as json
def get_user_profile(username: str) -> dict:
    url = f"https://lichess.org/api/user/{username}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


# parse a multi-game pgn string into a dataframe
def pgn_to_games_df(pgn_text: str) -> pd.DataFrame:
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
            if getattr(nxt, "eval", None) is not None:
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


# parse raw user pgn into a dataframe
def parse_user_pgn(pgn_text: str) -> pd.DataFrame:
    return pgn_to_games_df(pgn_text)


# stream the first n_games from the elite pgn file,
# parse headers+moves+evals, return as dataframe
def parse_elite_pgn(pgn_path: Path) -> pd.DataFrame:
    records = []
    with pgn_path.open(encoding="utf-8", errors="ignore") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            hdr = game.headers
            moves, evals = [], []
            node = game
            while node.variations:
                nxt = node.variation(0)
                moves.append(nxt.move.uci())
                if getattr(nxt, "eval", None) is not None:
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
                }
            )
    return pd.DataFrame(records)
