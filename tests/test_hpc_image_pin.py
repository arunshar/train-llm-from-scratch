"""Check that the MSI image pin is consistent and that the verifier fails closed.

    PYTHONPATH=. python tests/test_hpc_image_pin.py
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_FILE = ROOT / "hpc" / "expected_image.sha256"
PROJECT = ROOT / "hpc" / "project.yaml"
VERIFIER = ROOT / "hpc" / "verify_image_pin.sh"
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def load_pin() -> str:
    digest = PIN_FILE.read_text(encoding="utf-8").strip()
    if not digest:
        raise AssertionError("hpc/expected_image.sha256 is empty")
    return digest


def run_verifier(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(VERIFIER), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_pin_is_canonical():
    digest = load_pin()
    assert DIGEST_RE.fullmatch(digest), digest
    project_text = PROJECT.read_text(encoding="utf-8")
    assert f"container_digest: {digest}" in project_text
    for relative in (
        "hpc/submit_smoke.sh",
        "hpc/smoke.sbatch",
        "hpc/bootstrap.sbatch",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "verify_image_pin.sh" in text, relative
    print("ok  pin is 64 hex chars and is referenced by project.yaml plus submit paths")


def test_sidecar_match_and_mismatch():
    digest = load_pin()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        good = tmp_path / "container.sha256"
        bad = tmp_path / "wrong.sha256"
        missing = tmp_path / "absent.sha256"
        good.write_text(f"{digest}  /msi/images/fake.sif\n", encoding="utf-8")
        bad.write_text(("0" * 64) + "  /msi/images/fake.sif\n", encoding="utf-8")

        matched = run_verifier("--sidecar", str(good))
        assert matched.returncode == 0, matched.stderr
        assert matched.stdout.strip() == digest

        mismatched = run_verifier("--sidecar", str(bad))
        assert mismatched.returncode != 0
        assert "image digest mismatch" in mismatched.stderr

        absent = run_verifier("--sidecar", str(missing))
        assert absent.returncode != 0
        assert "missing digest source" in absent.stderr
    print("ok  sidecar verifier accepts the pin and rejects mismatch or a missing file")


def main() -> None:
    test_pin_is_canonical()
    test_sidecar_match_and_mismatch()
    print("ALL HPC IMAGE PIN TESTS PASSED")


if __name__ == "__main__":
    main()
