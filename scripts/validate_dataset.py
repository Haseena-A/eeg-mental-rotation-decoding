"""Inspect the local copy of the dataset and write the verified inventory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_mental_rotation.config import load_config, resolve  # noqa: E402
from eeg_mental_rotation.data_loader import find_dataset_root, inspect_dataset, locate_zip, inventory_zip  # noqa: E402
from eeg_mental_rotation.reporting import write_report  # noqa: E402


def main() -> int:
    config = load_config()
    raw_dir = resolve(config, "paths.raw_dir")
    tables = resolve(config, "paths.tables_dir")
    reports = resolve(config, "paths.reports_dir")
    tables.mkdir(parents=True, exist_ok=True)
    zip_path = locate_zip(raw_dir, config["dataset"]["zip_name"])
    root = find_dataset_root(raw_dir)
    if root is None and zip_path is not None:
        inventory = inventory_zip(zip_path)
    elif root is not None:
        inventory = inspect_dataset(root)
    else:
        print("No archive and no extracted STData directory in", raw_dir)
        print("Run: python scripts/download_data.py")
        return 1
    destination = tables / "dataset_inventory.json"
    destination.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    write_report(tables, reports / "research_report.md")
    print(f"wrote {destination}")
    print("participant folders:", inventory.get("n_participant_folders"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
