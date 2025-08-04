import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent    
sys.path.append(str(repo_root / "src"))             

from src.api.main import app               
