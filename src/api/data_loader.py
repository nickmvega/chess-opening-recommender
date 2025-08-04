import gzip
import io
import os
from functools import lru_cache

import httpx
import pandas as pd

BASE_URL = "/data/"
PARQUET = "lichess_elite_2025-05.parquet"
CSV = "elite_style_vectors.csv"


def _cdn_url(name: str) -> str:
    base = os.getenv("VERCEL_URL") or "http://localhost:3000"
    return f"https://{base}{BASE_URL}{name}"


@lru_cache(1)
def get_elite_games_df() -> pd.DataFrame:
    r = httpx.get(_cdn_url(PARQUET), timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


@lru_cache(1)
def get_style_vectors_df() -> pd.DataFrame:
    r = httpx.get(_cdn_url(CSV), timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))
