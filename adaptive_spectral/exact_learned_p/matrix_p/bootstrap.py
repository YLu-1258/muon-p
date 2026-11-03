"""Allow direct script execution while sharing the established numerical helpers."""
from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
