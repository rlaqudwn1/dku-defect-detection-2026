#!/usr/bin/env python3
"""Verify a downloaded competition dataset against the preserved manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Extracted open directory")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "data" / "manifests" / "data-manifest.json").read_bytes())
    failures = []
    for entry in manifest["files"]:
        path = args.data_root / entry["path"]
        if not path.is_file():
            failures.append(entry["path"] + ": missing")
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(entry["path"] + ": size mismatch")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            failures.append(entry["path"] + ": SHA-256 mismatch")
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {p.relative_to(args.data_root).as_posix()
              for p in args.data_root.rglob("*") if p.is_file()}
    failures.extend(path + ": unexpected file" for path in sorted(actual - expected))
    for failure in failures:
        print(failure)
    print(f"{'FAIL' if failures else 'PASS'}: {len(expected)} expected files")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
