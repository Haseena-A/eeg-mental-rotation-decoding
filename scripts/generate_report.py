"""Rebuild the research report from results already on disk."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_mental_rotation.config import load_config, resolve  # noqa: E402
from eeg_mental_rotation.reporting import write_report  # noqa: E402


def main() -> int:
    config = load_config()
    destination = write_report(resolve(config, "paths.tables_dir"), resolve(config, "paths.reports_dir") / "research_report.md")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
