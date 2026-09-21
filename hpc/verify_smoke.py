"""Verify every checkpoint produced by the MSI reduced-pipeline smoke run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from scripts.eval_post_training import model_from_ckpt

EXPECTED_CHECKPOINTS = {
    "base_pretrained.pt": "pretrain",
    "sft.pt": "sft",
    "reward.pt": "reward",
    "dpo.pt": "dpo_dpo",
    "ppo_upstream.pt": "ppo",
    "grpo_upstream.pt": "grpo",
    "ppo_sequential.pt": "ppo",
    "grpo_sequential.pt": "grpo",
}


def validate_checkpoint(path: Path, expected_stage: str, device: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "cfg", "stage", "step", "pytorch_version"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"{path.name} is missing keys: {missing}")
    if checkpoint["stage"] != expected_stage:
        raise ValueError(
            f"{path.name} stage is {checkpoint['stage']!r}, expected {expected_stage!r}"
        )

    cfg = checkpoint["cfg"]
    expected_model = {
        "vocab_size": 50_304,
        "context_length": 256,
        "n_embed": 128,
        "n_head": 4,
        "n_blocks": 2,
    }
    actual_model = {key: cfg.get(key) for key in expected_model}
    if actual_model != expected_model:
        raise ValueError(
            f"{path.name} model config is {actual_model}, expected {expected_model}"
        )

    state = checkpoint["model_state_dict"]
    first_tensor = next(iter(state.values()))
    if first_tensor.numel() == 0 or not torch.isfinite(first_tensor).all():
        raise ValueError(f"{path.name} begins with an empty or non-finite tensor")

    model = model_from_ckpt(str(path), device)
    tokens = torch.arange(8, device=device).reshape(1, 8)
    with torch.no_grad():
        logits, _ = model(tokens)
    if logits.shape != (1, 8, expected_model["vocab_size"]):
        raise ValueError(f"{path.name} produced unexpected logits shape {logits.shape}")
    if not torch.isfinite(logits).all():
        raise ValueError(f"{path.name} produced non-finite logits")
    del model, logits
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "bytes": path.stat().st_size,
        "stage": checkpoint["stage"],
        "step": checkpoint["step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = root / "fixture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing fixture manifest: {manifest_path}")

    checks = {}
    for filename, stage in EXPECTED_CHECKPOINTS.items():
        path = root / "ckpts" / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        checks[filename] = validate_checkpoint(path, stage, args.device)

    result = {
        "status": "SMOKE_RESULT_VERIFIED",
        "commit": os.environ.get("EXPECTED_COMMIT", "unknown"),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "checkpoints": checks,
    }
    (root / "smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "RESULT_STATUS").write_text(
        "SMOKE_RESULT_VERIFIED\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
