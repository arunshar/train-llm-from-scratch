"""Record the bounded Pile subset used by the 400M stage-1 envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py

EXPECTED_FILES = ("data/pile_dev.h5", "data/pile_train.h5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle["tokens"].shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    files = []
    for relative in EXPECTED_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "tokens": token_count(path),
            }
        )

    raw_dir = root / "data" / "pile_raw"
    raw_files = []
    for name in ("val.jsonl.zst", "00.jsonl.zst"):
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing raw shard {path}")
        raw_files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})

    manifest = {
        "classification": "pile-uncopyrighted-bounded",
        "license": "pile-uncopyrighted",
        "num_train_shards": 1,
        "source": "https://huggingface.co/datasets/monology/pile-uncopyrighted",
        "files": files,
        "raw_files": raw_files,
    }
    out = root / "data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
