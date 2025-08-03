# Chess Opening Recommender

A full-stack pipeline that fetches Lichess games, profiles player *style* via feature-engineering, clusters elite players, finds stylistic peers, and recommends personalized openings for White and Black.

---

## Table of Contents

1. [Project Flow](#project-flow)  
2. [Project Overview](#project-overview)  
3. [Features & Sources](#features--sources)  
4. [Architecture & Directory Layout](#architecture--directory-layout)  
5. [Getting Started](#getting-started)  
   * [Prerequisites](#prerequisites)  
   * [Installation](#installation)  
   * [Environment Variables](#environment-variables)  
6. [Running Locally](#running-locally)  
   * [API Server](#api-server)  
   * [Frontend](#frontend)  
7. [API Reference](#api-reference)  
8. [Caching & Storage](#caching--storage)  
9. [Examples](#examples)  
10. [References](#references)  

---

## Project Flow

1. **Overview**  
   Load a user’s PGN plus an elite reference set, then parse, feature-engineer, cluster, and recommend.

2. **Goal**  
   Produce a 10-dimensional numerical *style vector* for any user and leverage it to surface statistically strong, style-matching openings.

3. **Purpose**  
   Validate the entire pipeline on a **single user** in each notebook before scaling it behind the API.

4. **Step-by-Step (notebook cells)**  
   * **Load & Parse PGN** – convert raw PGN text into a `DataFrame` of moves & metadata.  
   * **Extract Features** – compute per-game metrics (`ply_count`, `avg_trades`, …).  
   * **Summarize Style** – aggregate per-game rows into one per-player vector.  
   * **Cluster & Neighbor Search** – place the user inside the elite style space, find nearest peers.  
   * **Opening Stats & Recommendation** – rank ECO codes by peer performance and output top picks.  

---

## Project Overview

Input a Lichess username (and optional time control) → receive:

* **Style profile** – 20 + numeric features (game length, trades, queen deployment, checks, win-rate, …)  
* **Top 5 stylistic peers** – nearest elite players from a 500-player dataset  
* **Opening recommendations** – 2–3 ECO codes for White *and* Black, balanced by peer score × sample size  

Stack:

* **Python** (Pandas, NumPy, scikit-learn)  
* **FastAPI** (backend)  
* **Vanilla HTML + JS** (frontend)  

---

## Features & Sources

Below are the extracted style dimensions, why they matter, and where to read more.

- **Game Length** (`avg_moves`, `pct_long_games`)  
  Longer games lean positional; short games often indicate sharp tactics or early mistakes.  
  *Sources:*  
  • [Lichess Insights – Game Length Analysis](https://lichess.org/blog/Insights/chess-game-length)  
  • Kruijswijk et al., “Statistical Patterns in Human Chess Play” (2020)

- **Material Trades** (`avg_trades`)  
  High trade counts signal comfort simplifying to clear endgames; low counts reflect piece-rich complexity.  
  *Sources:*  
  • Harikrishna et al., “Material Imbalances in Grandmaster Games” (2018)  
  • [Chess.com Blog – To Trade or Not to Trade](https://www.chess.com/blog)

- **Queen Deployment** (`avg_queen_move`)  
  Early queen moves reveal aggressive intentions; late deployment stresses development & safety.  
  *Sources:*  
  • John Nunn, *Understanding Chess Move by Move* (2001)  
  • [StackExchange – When Should You Move Your Queen?](https://chess.stackexchange.com/questions/when-should-you-move-your-queen)

- **Castling Behavior** (`pct_castled_early`)  
  Early castling highlights king safety; delaying castling keeps options open but risks exposure.  
  *Sources:*  
  • de Groot & Gobet, *Perception and Memory in Chess* (1996)  
  • [Lichess Blog – Castling Trends](https://lichess.org/blog)

- **Tactical Checks** (`avg_checks`)  
  Frequent checks denote tactical pressure and attacking propensity.  
  *Sources:*  
  • Yuri Averbakh, *Comprehensive Chess Endings* (1975)  
  • [Chessable – Using Checks as Tactical Weapons](https://www.chessable.com)

- **Performance Metrics** (`win_rate`, `pct_wins`, `pct_draws`, `pct_losses`)  
  Captures effectiveness against openings across a large sample.  
  *Sources:*  
  • Arpad Elo, *The Rating of Chessplayers* (1986)  
  • [FIDE Handbook – Performance Rating](https://handbook.fide.com)

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

