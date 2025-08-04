# api/index.py  – Vercel's entry point
import sys
from pathlib import Path

# make the repo root importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api.main import app
