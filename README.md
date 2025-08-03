# Chess Opening Recommender

With over 1000+ named chess openings, it can be hard to choose a chess opening that matches your particular style.
In this project, I have focused on fixing that issue. I have created a full pipeline that fetches a users Lichess games,
profiles a users playing style through feature engineering, then takes a dataset from elite players and finds their style. I then cluster the elite players with the users playing style to find match users openings with similar styles to elite players who win more games. I then recommend oepnings for both the WHite and Black pieces. 

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Note Flow](#project-flow)  
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

## Project Overview

Input a Lichess username (and optional time control) → receive:

* **Style profile** – 20 + numeric features (game length, trades, queen deployment, checks, win-rate, …)  
* **Top 5 stylistic peers** – nearest elite players from a 500-player dataset  
* **Opening recommendations** – 2–3 ECO codes for White *and* Black, balanced by peer score × sample size  

Stack:

* **Python** (Pandas, NumPy, scikit-learn)  
* **FastAPI** (backend)  
* **Vanilla HTML + JS** (frontend)  

## Notebook Flow

The notebooks/ folder was used for initial testing on fetching data, feature engineering, clustering users, and recommending openings before the API itself and website was created. 

**Overview**  
   Load a user’s PGN plus an elite reference set, then parse, feature-engineer, cluster, and recommend.

**Goal**  
   Produce a 10-dimensional numerical *style vector* for any user and leverage it to surface statistically strong, style-matching openings.

**Purpose**  
   Validate the entire pipeline on a **single user** in each notebook before scaling it behind the API.

**Step-by-Step (notebook cells)**  
   * **Load & Parse PGN** – convert raw PGN text into a `DataFrame` of moves & metadata.  
   * **Extract Features** – compute per-game metrics (`ply_count`, `avg_trades`, …).  
   * **Summarize Style** – aggregate per-game rows into one per-player vector.  
   * **Cluster & Neighbor Search** – place the user inside the elite style space, find nearest peers.  
   * **Opening Stats & Recommendation** – rank ECO codes by peer performance and output top picks.  


---

## Features & Sources

### Game Length (`avg_moves`, `pct_long_games`)
Longer games usually favour positional, technical play, while very short ones point to sharp tactics or early mistakes.  
**Sources:**  
- [Regan & Biswas – *Human and Computer Preferences at Chess*, AAAI-14 Workshop](https://cdn.aaai.org/ocs/ws/ws1274/8859-38007-1-PB.pdf)  
- [Mehdiyev – *Data-Driven Chess Typology: A Computational Approach*, Uppsala Univ. MSc Thesis 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Material Trades (`avg_trades`)
High trade counts signal comfort simplifying; low counts keep tension and complexity.  
**Sources:**  
- [Regan, Haworth & Biswas – *Psychometric Modeling of Decision-Making via Game Play*, CIG 2013](https://cse.buffalo.edu/~regan/papers/pdf/BHR2015ACG.pdf)  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Queen Deployment (`avg_queen_move`)
Early queen moves index attacking intentions; late deployment stresses development & safety.  
**Sources:**  
- [Lappo et al. – *Cultural Transmission of Move Choice in Chess*, arXiv 2302.10375](https://arxiv.org/abs/2302.10375)  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Castling Behavior (`pct_castled_early`)
Early castling secures the king; delaying can keep options open but risks exposure.  
**Sources:**  
- [Ringrose – Castling-timing statistics on 1.7 M games (Chess SE discussion, 2022)](https://chess.stackexchange.com/questions/39183/has-anyone-done-a-correlation-of-castling-with-winning-and-losing)  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Tactical Checks (`avg_checks`)
Frequent checks denote constant tactical pressure; low frequency suggests manoeuvring play.  
**Sources:**  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)  
- [Li – *Analysis of Trends in Chess Games* (UCSD report 2015)](https://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/058.pdf)

### Performance Metrics (`win_rate`, `pct_wins`, `pct_draws`, `pct_losses`)
Outcome statistics capture effectiveness and risk-taking tendencies.  
**Sources:**  
- [Regan & Haworth – *Intrinsic Chess Ratings*, AAAI 2011](https://cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)  
- [Chowdhary et al. – *Quantifying Human Performance in Chess*, *Sci. Rep.* 2023](https://www.nature.com/articles/s41598-023-27735-9)

### Material Sacrifices (`sacrifice_rate`)
Higher sacrifice rates reflect gambit-loving, initiative-driven styles.  
**Sources:**  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Opening Diversity (`opening_variety`)
Breadth of distinct openings played; experts tend to specialise, beginners diversify.  
**Sources:**  
- [Chowdhary et al. 2023](https://www.nature.com/articles/s41598-023-27735-9)

### Center Control & Pawn Structure (`center_control`, `pawn_structure`)
Healthy pawn-structure and central dominance typify positional styles.  
**Sources:**  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Piece Development Speed (`avg_dev_moves`)
Rapid minor-piece development indicates initiative-seeking classical play.  
**Sources:**  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Queen Exchange Tendency (`queen_trade_freq`)
Early queen trades reduce volatility; avoiding them keeps attacking chances.  
**Sources:**  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Endgame Involvement (`pct_games_endgame`)
High endgame frequency implies patient, technical play; low frequency points to tactical games decided early.  
**Sources:**  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Draw Rate (`draw_rate`)
Very high draw rates mark ultra-solid, risk-averse styles; low draw rates indicate “fight-to-win” attitudes.  
**Sources:**  
- [Sonas – *Draw Statistics at Top Level Chess*, ChessBase 2011](https://en.chessbase.com/post/sonas-what-exactly-is-the-problem-)  
- [Ringrose 2022](https://chess.stackexchange.com/questions/39183/has-anyone-done-a-correlation-of-castling-with-winning-and-losing)

### Prophylactic Moves (`pct_prophylactic`)
Frequency of anticipatory defensive moves that pre-empt opponent plans.  
**Sources:**  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Engine-Move Agreement (`engine_alignment`)
Closeness of a player’s moves to top-engine choices; high alignment = objective precision.  
**Sources:**  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)

### Accuracy & Mistakes (`error_rate`, `accuracy`)
Aggregate computer-evaluated error rate; lower error = solid style, higher = volatile/risky.  
**Sources:**  
- [Regan & Haworth 2011](https://cse.buffalo.edu/~regan/papers/pdf/ReHa11c.pdf)  
- [Mehdiyev 2025](https://uu.diva-portal.org/smash/get/diva2:1977201/FULLTEXT01.pdf)


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

