# Chess Opening Recommender

A personalized opening recommendation tool for Lichess users. It analyzes a player’s game history (strengths, weaknesses, style) and suggests suitable chess openings (with study resources), then displays the chosen opening’s main line on an interactive board.

---

## Project Objectives

1. **Understand play style:** Extract quantitative features from a user’s public Lichess games.
2. **Approximate style similarity:** Embed every active Lichess user (and all titled users) in the same vector space; measure closeness.
3. **Opening recommendation:** Suggest openings that stylistically similar users both *play* and *succeed* with.
4. **Celeb GM/IM match:** Assign the titled player whose style vector is closest to the user’s.

---

## Model & Data‑Prep Overview

- **Data ingestion:** Parse monthly **Lichess PGN.zst** dumps into a columnar store (Parquet/DuckDB).
- **Feature engineering:** Move-sequence embeddings (Doc2Vec/Word2Vec on SAN tokens), aggregated opening metrics (win-rate, engine eval after move 10), positional/tactical counts (piece activity, captures, checks, pawn-structure density).
- **Style embedding:** Concatenate engineered stats with sequence embedding → reduce via PCA → ℝ<sub>d</sub> style vector.
- **Similarity search:** Build FAISS ANN index over all style vectors; cosine distance retrieves top-K similar users.
- **Recommendation:** For each opening, compute weighted success among top-K neighbors → rank & return N best.
- **Celeb match:** Compute nearest titled player (GM/IM/FM) in the same vector space.

## Step‑by‑Step Workflow

### 1. Ingest Raw Data


```bash
python src/process_data.py  # parses PGN → processed parquet
```
- Reads `<dataset_file>` set in **config.json**.
- Outputs `<processed_file>` containing: `game_id, white_id, black_id, white_elo, black_elo, eco, result, pgn`.

### 2. Further Processing & Wrangling

```bash
python src/format_data.py  # loads parquet → pandas/duckdb
```
- Produces per-game and per-user tables ready for feature generation.

### 3. Feature Engineering

```bash
python src/feature_engineering.py  # builds style vectors
```
- Generates move embeddings + engineered counts.
- Saves `user_embedding.parquet` and `titled_embedding.parquet`.

### 4. Similarity Index

```bash
python src/build_ann.py  # FAISS index
```
- Indexes all user vectors for millisecond retrieval.

### 5. Opening Recommendation


### 5. Opening Recommendation

```bash
python src/recommend.py --user <lichess_username> --topk 10
```
- Returns JSON with openings (`eco`, `name`, `pgn`, `score`) and closest titled player.

### 6. API Service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- **GET `/recommend/{username}`** → opening list.
- **GET `/celeb/{username}`** → `{ "celeb": "GM MagnusCarlsen" }`.


### 7. Static Demo
- `/chess` page (GitHub Pages) embeds **chessboard.js** & **chess.js**.
- Fetches API, animates the top opening line for interactive viewing.

---

## Data Sources

| Dataset                                       | Purpose                           | License         |
| --------------------------------------------- | --------------------------------- | --------------- |
| **Lichess Open Database** – monthly PGN dumps | Core game history                 | CC-0            |
| **Lichess API** `/api/games/user/{username}`  | On-demand latest games            | AGPL-3          |
| **Chess Openings (Kaggle)**                   | ECO code ↔︎ name ↔︎ PGN main line | CC BY-SA 4.0    |
| (Optional) **Chess.com Games (Kaggle)**       | Cross-platform extension          | CC BY-NC-SA 4.0 |

---

## Tech Stack *(TBD)*

---


## Planned Repo Structure

```text
chess-openings-recs/
├─ src/
│  ├─ data/                # ingestion & processing scripts
│  ├─ features/            # feature engineering modules
│  ├─ models/              # similarity & recommendation logic
│  ├─ api/                 # FastAPI service
│  └─ utils/               # helpers
├─ notebooks/              # exploratory analysis
├─ tests/                  # pytest
├─ config.json
├─ Dockerfile
└─ README.md
```

---

## License

MIT

---

## Acknowledgements

- Lichess for openly publishing game data.
- Kaggle community for curated opening datasets.
