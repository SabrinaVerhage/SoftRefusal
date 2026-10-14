"""Research measurements, not semantic embeddings or a universal avoidance axis."""

__version__ = "0.1.0"

# Keep model downloads with this project; credentials remain in the user's HF profile.
import os
from pathlib import Path
_cache = Path(__file__).resolve().parents[2] / "models" / "huggingface"
if _cache.exists():
    os.environ.setdefault("HF_HUB_CACHE", str(_cache))
