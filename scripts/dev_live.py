"""Run the real app locally from any working directory: keys from .env, live Qloo and model calls.

    .venv/bin/python scripts/dev_live.py   ->  http://127.0.0.1:7862
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PORT", "7862")

from taste_mcp.app import main  # noqa: E402

if __name__ == "__main__":
    main()
