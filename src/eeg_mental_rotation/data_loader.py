"""Locate and load the published EEG recordings and PsychoPy task logs.

The Figshare deposit is one zip (``STData.zip``). The paper states that it
contains a ``STData`` folder with one subfolder per participant, and that each
subfolder contains an EEG file and a PsychoPy file (PSY). File names inside the
archive are discovered, not assumed. A participant is skipped, with a reason,
when a required file is absent.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import BEHAVIOUR_ALIASES, EEG_ALIASES, canonical_columns


def find_dataset_root(raw_dir: Path) -> Path | None:
    """Return an extracted STData directory, if one is already on disk."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None
    candidates = [raw_dir / "STData", raw_dir]
    for candidate in candidates:
        if candidate.is_dir() and _looks_like_participant_tree(candidate):
            return candidate
    for child in raw_dir.iterdir():
        if child.is_dir() and _looks_like_participant_tree(child):
            return child
    return None


def _looks_like_participant_tree(path: Path) -> bool:
    folders = [item for item in path.iterdir() if item.is_dir()]
    if len(folders) < 1:
        return False
    sample = folders[0]
    names = {item.name.lower() for item in sample.iterdir() if item.is_file()}
    return any(name.endswith("eeg.csv") or name == "eeg.csv" for name in names) and any(
        name.endswith("psy.csv") or name == "psy.csv" for name in names
    )


def locate_zip(raw_dir: Path, zip_name: str) -> Path | None:
    raw_dir = Path(raw_dir)
    direct = raw_dir / zip_name
    if direct.exists():
        return direct
    matches = list(raw_dir.glob("*.zip"))
    return matches[0] if matches else None


def inventory_zip(zip_path: Path) -> dict[str, Any]:
    """List archive members without extracting the recordings."""
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    participants: dict[str, list[str]] = {}
    for name in names:
        parts = Path(name).parts
        if len(parts) < 2 or name.endswith("/"):
            continue
        participant = parts[-2]
        participants.setdefault(participant, []).append(parts[-1])
    return {
        "zip_path": str(zip_path),
        "n_members": len(names),
        "n_participant_folders": len(participants),
        "participants": {key: sorted(value) for key, value in sorted(participants.items())},
    }


def extract_recordings(zip_path: Path, destination: Path, stems: tuple[str, ...] = ("eeg", "psy")) -> Path:
    """Extract only the EEG and PsychoPy tables. Video and other sensors stay in the zip."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        selected = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx"}
            and any(_matches_stem(Path(name), stem) for stem in stems)
        ]
        for name in selected:
            parts = Path(name).parts
            relative = Path(*parts[1:]) if parts and parts[0].lower() == "stdata" else Path(*parts[-2:])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(archive.read(name))
    return destination


def list_participants(dataset_root: Path) -> list[Path]:
    return sorted(path for path in Path(dataset_root).iterdir() if path.is_dir())


def _matches_stem(path: Path, token: str) -> bool:
    """True for ``EEG.csv``, ``1_EEG.csv`` and ``eeg_data.csv``."""
    stem = path.stem.lower().replace("-", "_")
    token = token.lower()
    return stem == token or stem.startswith(token + "_") or stem.endswith("_" + token)


def _find_file(participant_dir: Path, stem: str) -> Path | None:
    matches = [path for path in participant_dir.iterdir() if path.is_file() and _matches_stem(path, stem)]
    if not matches:
        return None
    # Prefer a csv when several exports share the stem.
    matches.sort(key=lambda path: (path.suffix.lower() != ".csv", path.name.lower()))
    return matches[0]


def _count_rows(path: Path) -> int:
    """Count records without parsing a multi-hundred-megabyte EEG table twice."""
    with path.open("rb") as handle:
        newlines = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))
    return max(newlines - 1, 0)


def read_table(path: Path) -> pd.DataFrame:
    """Read a comma- or tab-separated table. The paper states sensor files are CSV."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    sample = path.read_text(encoding="utf-8", errors="replace")[:2000]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    return pd.read_csv(path, sep=delimiter, low_memory=False)


def load_eeg(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = read_table(path)
    mapping = canonical_columns(frame.columns, EEG_ALIASES)
    renamed = frame.rename(columns={original: canonical for canonical, original in mapping.items()})
    return renamed, mapping


def load_behaviour(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = read_table(path)
    mapping = canonical_columns(frame.columns, BEHAVIOUR_ALIASES)
    renamed = frame.rename(columns={original: canonical for canonical, original in mapping.items()})
    return renamed, mapping


def load_participant(participant_dir: Path) -> dict[str, Any]:
    """Load one participant. Missing files are recorded rather than raising."""
    participant_id = participant_dir.name
    eeg_path = _find_file(participant_dir, "eeg")
    behaviour_path = _find_file(participant_dir, "psy")
    record: dict[str, Any] = {
        "participant_id": participant_id,
        "eeg_path": str(eeg_path) if eeg_path else None,
        "behaviour_path": str(behaviour_path) if behaviour_path else None,
        "problems": [],
    }
    if eeg_path is None:
        record["problems"].append("no EEG file")
    else:
        eeg, eeg_map = load_eeg(eeg_path)
        eeg.insert(0, "participant_id", participant_id)
        record["eeg"] = eeg
        record["eeg_columns"] = eeg_map
    if behaviour_path is None:
        record["problems"].append("no PsychoPy file")
    else:
        behaviour, behaviour_map = load_behaviour(behaviour_path)
        behaviour.insert(0, "participant_id", participant_id)
        record["behaviour"] = behaviour
        record["behaviour_columns"] = behaviour_map
    return record


def inspect_dataset(dataset_root: Path) -> dict[str, Any]:
    """Build a verified inventory: files, headers, row counts. No invented fields."""
    participants = []
    for folder in list_participants(dataset_root):
        eeg_path = _find_file(folder, "eeg")
        behaviour_path = _find_file(folder, "psy")
        entry: dict[str, Any] = {
            "participant_id": folder.name,
            "files": sorted(path.name for path in folder.iterdir() if path.is_file()),
            "eeg_file": eeg_path.name if eeg_path else None,
            "behaviour_file": behaviour_path.name if behaviour_path else None,
        }
        if eeg_path is not None:
            header = [str(column) for column in pd.read_csv(eeg_path, nrows=0).columns]
            mapping = canonical_columns(header, EEG_ALIASES)
            entry["eeg_rows"] = _count_rows(eeg_path)
            entry["eeg_headers"] = header
            entry["eeg_canonical"] = sorted(mapping)
            entry["eeg_unmapped"] = [column for column in header if column not in mapping.values()]
        if behaviour_path is not None:
            header = [str(column) for column in pd.read_csv(behaviour_path, nrows=0).columns]
            mapping = canonical_columns(header, BEHAVIOUR_ALIASES)
            entry["behaviour_rows"] = _count_rows(behaviour_path)
            entry["behaviour_headers"] = header
            entry["behaviour_canonical"] = sorted(mapping)
            entry["behaviour_unmapped"] = [column for column in header if column not in mapping.values()]
        participants.append(entry)
    return {
        "dataset_root": str(dataset_root),
        "n_participant_folders": len(participants),
        "participants": participants,
    }


def iter_zip_participants(zip_path: Path, limit: int | None = None):
    """Yield one participant at a time from the archive, without writing the tables to disk.

    The EEG tables are several gigabytes uncompressed. Extracting them is optional
    and is not required to run the analysis.
    """
    groups: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.suffix.lower() not in {".csv", ".tsv"}:
                continue
            if _matches_stem(path, "eeg"):
                groups.setdefault(path.parent.name, {})["eeg"] = name
            elif _matches_stem(path, "psy"):
                groups.setdefault(path.parent.name, {})["psy"] = name
        participant_ids = sorted(groups)
        if limit is not None:
            participant_ids = participant_ids[:limit]
        for participant_id in participant_ids:
            entry = groups[participant_id]
            record: dict[str, Any] = {
                "participant_id": participant_id,
                "eeg_path": entry.get("eeg"),
                "behaviour_path": entry.get("psy"),
                "problems": [],
            }
            if "eeg" not in entry:
                record["problems"].append("no EEG file")
            else:
                eeg, eeg_map = _read_archived_table(archive, entry["eeg"], EEG_ALIASES)
                eeg.insert(0, "participant_id", participant_id)
                record["eeg"] = eeg
                record["eeg_columns"] = eeg_map
            if "psy" not in entry:
                record["problems"].append("no PsychoPy file")
            else:
                behaviour, behaviour_map = _read_archived_table(archive, entry["psy"], BEHAVIOUR_ALIASES)
                behaviour.insert(0, "participant_id", participant_id)
                record["behaviour"] = behaviour
                record["behaviour_columns"] = behaviour_map
            yield record


def _read_archived_table(
    archive: zipfile.ZipFile, name: str, aliases: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    with archive.open(name) as handle:
        frame = pd.read_csv(handle, low_memory=False)
    mapping = canonical_columns(frame.columns, aliases)
    renamed = frame.rename(columns={original: canonical for canonical, original in mapping.items()})
    return renamed, mapping


def write_inventory(inventory: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
