"""Forward to the unified four-method suite in the parent directory."""
import runpy
from pathlib import Path
import sys

if __name__ == "__main__":
    parent = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(parent))
    runpy.run_path(str(parent / "plot_comparison.py"), run_name="__main__")
