import gzip, pandas as pd
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "frontend/data"
CSV_GZ   = DATA_DIR / "lichess_elite_2025-05.csv.gz"
STYLE_CSV = DATA_DIR / "elite_style_vectors.csv"

@lru_cache(1)
def get_elite_games_df() -> pd.DataFrame:
    with gzip.open(CSV_GZ, "rt") as f:
        return pd.read_csv(f)

@lru_cache(1)
def get_style_vectors_df() -> pd.DataFrame:
    return pd.read_csv(STYLE_CSV)
