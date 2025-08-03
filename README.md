# Chess Opening Recommender

A full‐stack pipeline that fetches Lichess games, profiles player “style” via feature engineering, clusters elite players, finds stylistic peers, and recommends personalized openings for White and Black.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture & Directory Layout](#architecture--directory-layout)
4. [Getting Started](#getting-started)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
   * [Environment Variables](#environment-variables)
5. [Running Locally](#running-locally)

   * [API Server](#api-server)
   * [Frontend](#frontend)
6. [API Reference](#api-reference)
7. [Caching & Storage](#caching--storage)
8. [Examples](#examples)
9. [Next Steps](#next-steps)
10. [Credits](#credits)

---

## Project Overview

This system allows you to input a Lichess username (and optionally a time control) and receive:

* **Your style profile** (20+ numeric features: game length, material trades, queen deployment, checks, win-rate, etc.).
* **Top 5 stylistic peers** drawn from a reference set of 500 “elite” players.
* **Recommended openings** (2–3 for White and Black) based on peer performance, weighted by both score% and sample size.

Built with:

* **Python** (data fetching & feature engineering)
* **FastAPI** (backend REST endpoints)
* **Pandas & NumPy** (data manipulation)
* **scikit-learn** (StandardScaler, K-Means, nearest-neighbors)
* **Vanilla HTML** 

---

## Features

* **Data Fetching**: Streaming PGN from Lichess API with moves, engine evals, and opening metadata.
* **Feature Engineering**: Per-game metrics (`ply_count`, `avg_trades`, `first_queen_ply`, `castled_early`, `checks`, `result_score`).
* **Style Vectorization**: Aggregation into per-player style vectors (means & proportions across games).
* **Clustering & Matching**: StandardScaler + K-Means clustering for style archetypes, Euclidean nearest-neighbors for peer selection.
* **Opening Stats**: Frequency & performance (`score_pct`, log-weighted) of ECO codes among peers.
* **Recommendations**: Top openings for White and Black, balancing success and sample size.

---

## Architecture & Directory Layout

```
CHESS-OPENING-RECOMMENDER/      ← project root
├── README.md
├── cache/                      ← per-user PGN + parsed caches
│   └── user_cache/
├── frontend/                   ← static assets
│   └── index.html
├── src/
│   ├── api/
│   │   └── main.py             ← FastAPI application
│   └── recommender/
│       ├── data_fetcher.py
│       ├── feature_engineering.py
│       ├── clustering.py
│       └── opening_recommender.py
├── storage/                    ← reference datasets
│   ├── lichess_elite_2025-05.parquet
│   └── elite_style_vectors.csv
├── pyproject.toml              ← dependencies, scripts
└── poetry.lock                 ← locked versions
```

---

## Getting Started

### Prerequisites

* Python 3.10+
* [`poetry`](https://python-poetry.org/) or `pipenv` for venv & deps
* A [Lichess API token](https://lichess.org/account/oauth/token)

### Installation

```bash
git clone git@github.com:nickmvega/chess-opening-recommender.git
cd chess-opening-recommender
poetry install
```

### Environment Variables

Create a `.env` in project root:

```
LICHESS_TOKEN=your_lichess_api_token_here
USER_CACHE_DIR=/absolute/path/to/cache/user_cache
```

---

## Running Locally

### API Server

```bash
uvicorn src.api.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

### Frontend

Open `http://127.0.0.1:8000/` in your browser.

---

## API Reference

### `POST /fetch`

Fetch raw PGN.

* **Body**: `{ "username": "Chessanonymous1" }`
* **Response**: `{ "pgn_path": "/.../cache/user_cache/Chessanonymous1.pgn" }`

### `POST /parse`

Parse cached PGN.

* **Body**: same as `/fetch`
* **Response**:

  ```
  {
    "parsed_path": "/.../cache/user_cache/parsed/Chessanonymous1.parquet",
    "games_count": 123
  }
  ```

### `POST /features`

Compute style features.

* **Body**: same
* **Response**:

  ```
  {
    "user_style": {
      "avg_moves": 55.3,
      "pct_long_games": 0.12,
      … 10 metrics …
    }
  }
  ```

### `POST /recommend/{username}`

Full pipeline: fetch→parse→features→match→recommend.

* **Query**: `?time_control=blitz` (optional)
* **Response**:

  ```
  {
    "top_peers": ["Attack2GM","rtahmass",…],
    "white_recommendations":[
      {"eco":"A00","opening":"Kádas Opening","games_played":12,"score_pct":0.75},…
    ],
    "black_recommendations":[…]
  }
  ```

---

## Caching & Storage

* **Raw PGN**: `USER_CACHE_DIR/{username}.pgn`
* **Parsed Parquet**: `USER_CACHE_DIR/parsed/{username}.parquet`
* Consider TTL eviction (e.g. delete >24 hr old files).

---

## Examples

```bash
curl -X POST "http://127.0.0.1:8000/recommend/Chessanonymous1?time_control=rapid"
```

