"""Create deterministic synthetic fixtures for the MSI end-to-end smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

CONTEXT_LENGTH = 256
VOCAB_SIZE = 50_304
SEED = 1_337


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_pretrain_data(path: Path, rng: np.random.Generator) -> None:
    token_count = CONTEXT_LENGTH * 64 + 1
    tokens = rng.integers(0, VOCAB_SIZE, size=token_count, dtype=np.int32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tokens", data=tokens)


def write_sft_data(path: Path, rng: np.random.Generator) -> None:
    tokens = rng.integers(
        0,
        VOCAB_SIZE,
        size=(32, CONTEXT_LENGTH),
        dtype=np.int32,
    )
    loss_mask = np.zeros_like(tokens, dtype=np.uint8)
    loss_mask[:, -64:] = 1
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tokens", data=tokens)
        handle.create_dataset("loss_mask", data=loss_mask)


def preference_rows(count: int) -> list[dict]:
    return [
        {
            "prompt": f"What is {index} + {index}?",
            "chosen": f"<think>{index} + {index} = {2 * index}</think><answer>{2 * index}</answer>",
            "rejected": f"<think>I guessed.</think><answer>{2 * index + 1}</answer>",
        }
        for index in range(1, count + 1)
    ]


def prompt_rows(count: int) -> list[dict]:
    return [
        {
            "prompt": f"What is {index} + {index}?",
            "gold": float(2 * index),
        }
        for index in range(1, count + 1)
    ]


def build_fixtures(root: Path) -> None:
    data_dir = root / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty smoke data directory: {data_dir}")

    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ckpts", "logs", "hf_cache"):
        (root / name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    write_pretrain_data(data_dir / "pile_train.h5", rng)
    write_pretrain_data(data_dir / "pile_dev.h5", rng)
    write_sft_data(data_dir / "sft_packed.h5", rng)
    write_sft_data(data_dir / "sft_dev_packed.h5", rng)
    write_jsonl(data_dir / "preferences.jsonl", preference_rows(32))
    write_jsonl(data_dir / "preferences_test.jsonl", preference_rows(8))
    write_jsonl(data_dir / "rl_prompts_train.jsonl", prompt_rows(32))
    write_jsonl(data_dir / "rl_prompts_test.jsonl", prompt_rows(8))
    write_jsonl(data_dir / "arithmetic_prompts.jsonl", prompt_rows(32))

    files = sorted(path for path in data_dir.iterdir() if path.is_file())
    manifest = {
        "classification": "synthetic",
        "license": "synthetic-no-restriction",
        "seed": SEED,
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (root / "fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    build_fixtures(args.root.resolve())
    print(f"created deterministic smoke fixtures under {args.root.resolve()}")


if __name__ == "__main__":
    main()
