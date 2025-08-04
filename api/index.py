# api/index.py  – Vercel's entry point
from pathlib import Path
import sys

# make the repo root importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api.main import app  
