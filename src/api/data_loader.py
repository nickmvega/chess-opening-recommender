# src/api/data_loader.py
import gzip
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "frontend" / "data"

CSV_GZ = DATA_DIR / "lichess_elite_2025-05.csv.gz"
STYLE_CSV = DATA_DIR / "elite_style_vectors.csv"


@lru_cache()
def get_elite_games_df() -> pd.DataFrame:
    with gzip.open(CSV_GZ, "rt") as f:
        return pd.read_csv(f)


@lru_cache()
def get_style_vectors_df() -> pd.DataFrame:
    return pd.read_csv(STYLE_CSV)
