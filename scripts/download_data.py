"""Download the Figshare archive. The file is about 1.1 GB and is not committed."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_mental_rotation.config import load_config, resolve  # noqa: E402


def main() -> int:
    config = load_config()
    raw_dir = resolve(config, "paths.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / config["dataset"]["zip_name"]
    url = config["dataset"]["figshare_file_url"]
    expected = config["dataset"]["figshare_md5"]
    if destination.exists():
        digest = hashlib.md5(destination.read_bytes()).hexdigest()
        if digest == expected:
            print(f"already present and checksum matches: {destination}")
            return 0
        print(f"existing file checksum is {digest}, expected {expected}; downloading again")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, destination)
    digest = hashlib.md5(destination.read_bytes()).hexdigest()
    print(f"saved {destination} ({destination.stat().st_size} bytes), md5 {digest}")
    if digest != expected:
        print("checksum does not match the Figshare record")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
