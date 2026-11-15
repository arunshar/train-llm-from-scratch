"""Verify a 400M pretrain checkpoint without requiring train_steps to finish."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.eval_post_training import model_from_ckpt

EXPECTED_MODEL = {
    "vocab_size": 50_304,
    "context_length": 1024,
    "n_embed": 1024,
    "n_head": 16,
    "n_blocks": 24,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt.resolve()
    out = args.out.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "cfg", "stage", "step", "pytorch_version"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"{ckpt_path.name} is missing keys: {missing}")
    if checkpoint["stage"] != "pretrain":
        raise ValueError(f"stage is {checkpoint['stage']!r}, expected 'pretrain'")

    cfg = checkpoint["cfg"]
    actual = {key: cfg.get(key) for key in EXPECTED_MODEL}
    if actual != EXPECTED_MODEL:
        raise ValueError(f"model config is {actual}, expected {EXPECTED_MODEL}")

    model = model_from_ckpt(str(ckpt_path), args.device)
    tokens = torch.arange(8, device=args.device).reshape(1, 8)
    with torch.no_grad():
        logits, _ = model(tokens)
    if logits.shape != (1, 8, EXPECTED_MODEL["vocab_size"]):
        raise ValueError(f"unexpected logits shape {logits.shape}")
    if not torch.isfinite(logits).all():
        raise ValueError("produced non-finite logits")

    step = int(checkpoint["step"])
    result = {
        "status": "PRETRAIN_CKPT_VERIFIED",
        "complete_train_steps": step >= 50_000,
        "step": step,
        "bytes": ckpt_path.stat().st_size,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "metrics": checkpoint.get("metrics") or {},
    }
    (out / "pretrain_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "RESULT_STATUS").write_text("PRETRAIN_CKPT_VERIFIED\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
