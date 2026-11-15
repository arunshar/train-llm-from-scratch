"""Checks that the 400M stage-1 envelope stays bounded and fail-closed.

    PYTHONPATH=. python tests/test_hpc_400m_envelope.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_yaml_is_stage1_only():
    text = (ROOT / "hpc" / "project_400m.yaml").read_text(encoding="utf-8")
    assert "num_train_shards: 1" in text
    assert "GPU_count: 1" in text
    assert "maximum_concurrent_jobs: 1" in text
    assert 'wall_time: "24:00:00"' in text
    assert "01 through 29" in text
    assert "SFT reward DPO PPO GRPO" in text
    print("ok  project_400m.yaml is one shard, one H100, 24h, no post-training")


def test_scripts_pin_one_shard_and_resume():
    data = (ROOT / "hpc" / "run_400m_data.sh").read_text(encoding="utf-8")
    assert "--num_shards 1" in data
    assert "02.jsonl" not in data
    pretrain = (ROOT / "hpc" / "run_400m_pretrain.sh").read_text(encoding="utf-8")
    assert "--resume" in pretrain
    assert "configs/pretrain.json" in pretrain
    assert "train_sft.py" not in pretrain
    for relative in ("hpc/submit_400m_data.sh", "hpc/submit_400m_pretrain.sh"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "dirty remote checkout" in text, relative
        assert "concurrency ceiling is one active job" in text, relative
        assert "verify_image_pin.sh" in text, relative
    print("ok  data uses shard 00 only and pretrain can resume")


def test_base_config_is_400m():
    base = json.loads((ROOT / "configs" / "base.json").read_text(encoding="utf-8"))
    assert base["n_embed"] == 1024
    assert base["n_head"] == 16
    assert base["n_blocks"] == 24
    assert base["context_length"] == 1024
    print("ok  default configs/base.json is the 400M shape")


def main() -> None:
    test_project_yaml_is_stage1_only()
    test_scripts_pin_one_shard_and_resume()
    test_base_config_is_400m()
    print("ALL 400M ENVELOPE TESTS PASSED")


if __name__ == "__main__":
    main()
