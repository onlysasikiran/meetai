"""Fetch fingerprinted research artifacts. Model execution never uses the network."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", "models", "raw"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = {}
    for manifest in ("assets.json", "meetai-assets.json", "phase3-assets.json"):
        assets.update(json.loads((root / "configs" / manifest).read_text()))
    for name, info in assets.items():
        if args.group != "all" and not name.startswith(args.group + "/"):
            continue
        dest = root / ".assets" / name
        if dest.exists() and sha(dest) == info["sha256"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
            temp = Path(out.name)
        try:
            print("Downloading", name, flush=True)
            with urllib.request.urlopen(info["url"], timeout=180) as inp, temp.open("wb") as out:
                shutil.copyfileobj(inp, out, 1024 * 1024)
            if sha(temp) != info["sha256"]:
                raise ValueError(f"Checksum mismatch for {name}")
            os.replace(temp, dest)
        finally:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
