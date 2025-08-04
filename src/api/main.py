import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recommender.clustering import cluster_styles, find_style_neighbors
from recommender.data_fetcher import fetch_user_games, parse_user_pgn
from recommender.feature_engineering import (extract_style_features,
                                             summarize_player_features)
from recommender.opening_recommender import (compute_opening_stats,
                                             recommend_openings)

# directory resolution
src_dir = Path(__file__).resolve().parent.parent
project_root = src_dir.parent
frontend_dir = project_root / "frontend"
storage_dir = project_root / "storage"
user_cache_dir = Path(
    os.getenv("USER_CACHE_DIR", str(project_root / "cache" / "user_cache"))
)

# load reference data once
elite_games_df = pd.read_parquet(storage_dir / "lichess_elite_2025-05.parquet")
elite_style_v = pd.read_csv(storage_dir / "elite_style_vectors.csv")
clustered_elite, scaler, kmeans = cluster_styles(elite_style_v, n_clusters=50, random_state=42)

# app setup
app = FastAPI(title="Chess Opening Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# serve index.html at `/` and static assets under `/static`
@app.get("/", include_in_schema=False)
def serve_index():
    idx = frontend_dir / "index.html"
    if not idx.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(idx)


app.mount("/static", StaticFiles(directory=str(frontend_dir), html=True), name="static")


# schemas
class RecommendResp(BaseModel):
    top_peers: list[str]
    white_recommendations: list[dict]
    black_recommendations: list[dict]


# recommendation endpoint
@app.post("/recommend/{username}", response_model=RecommendResp)
def recommend_all(
    username: str,
    time_control: Optional[str] = Query(
        None, description="bullet, blitz, rapid, classical"
    ),
):
    user = username.strip()
    if not user:
        raise HTTPException(400, "Invalid username")

    # fetch & cache raw pgn
    user_cache_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = user_cache_dir / f"{user}.pgn"
    try:
        fetch_user_games(user, save_to=pgn_path)
    except Exception as e:
        raise HTTPException(502, f"Fetch failed: {e}")

    # parse into dataframe
    raw_pgn = pgn_path.read_text("utf-8")
    games_df = parse_user_pgn(raw_pgn)
    if games_df.empty:
        raise HTTPException(404, "No games found for user")

    # filter by time_control if requested
    if time_control:
        games_df = games_df[
            games_df["time_control"].str.contains(time_control, na=False)
        ]
        if games_df.empty:
            raise HTTPException(404, f"No {time_control} games for user")

    # style features + summary vector
    feats_df = extract_style_features(games_df)
    user_vec = summarize_player_features(feats_df)

    # find top-5 stylistic peers
    neighbors_df = find_style_neighbors(user_vec, clustered_elite, scaler, top_n=5)
    peer_list = neighbors_df["player"].tolist()

    # gather only peer games (and filter by time_control)
    peer_games = elite_games_df[
        elite_games_df["white"].isin(peer_list)
        | elite_games_df["black"].isin(peer_list)
    ]
    if time_control:
        peer_games = peer_games[
            peer_games["time_control"].str.contains(time_control, na=False)
        ]

    # compute opening stats on peer_games (min_games=5 for robustness)
    white_stats = compute_opening_stats(elite_games_df, peer_list, color="white")
    black_stats = compute_opening_stats(elite_games_df, peer_list, color="black")

    # top-3 recommendations each
    white_recs, black_recs = recommend_openings(white_stats, black_stats, top_n=3)

    return RecommendResp(
        top_peers=peer_list,
        white_recommendations=[
            {"eco": e, "opening": o, "games_played": g, "score_pct": s}
            for e, o, g, s in white_recs
        ],
        black_recommendations=[
            {"eco": e, "opening": o, "games_played": g, "score_pct": s}
            for e, o, g, s in black_recs
        ],
    )
