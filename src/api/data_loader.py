import gzip
from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE_URL = "/data/" 
CSV_GZ = "lichess_elite_2025-05.csv.gz"
STYLE_CSV = "elite_style_vectors.csv"


@lru_cache(1)
def get_elite_games_df():
    import io

    import httpx

    r = httpx.get(f"{BASE_URL}{CSV_GZ}", timeout=60)
    r.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
        return pd.read_csv(f)


@lru_cache(1)
def get_style_vectors_df():
    import io

    import httpx

    r = httpx.get(f"{BASE_URL}{STYLE_CSV}", timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))
