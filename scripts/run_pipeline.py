"""Run the analysis pipeline.

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --max-participants 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_mental_rotation.pipeline import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the EEG mental-rotation analysis.")
    parser.add_argument("--max-participants", type=int, default=None)
    args = parser.parse_args()
    status = run(max_participants=args.max_participants)
    print(json.dumps(status, indent=2, default=str))
    return 0 if status.get("status") in {"ok", "aligned_insufficient", "no_data", "no_usable_participants"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
