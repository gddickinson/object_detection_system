"""Put the repo root on sys.path so tests import the project packages (models, postprocess, web)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
