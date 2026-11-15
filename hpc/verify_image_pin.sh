#!/usr/bin/env bash
# Compare a SIF digest to the pinned value in hpc/expected_image.sha256.
# The pin is the digest recorded by bootstrap job 1503077 and smoke job 1503299
# on 2026-09-20. Do not bump it because a docker tag moved. Re-pin only after a
# new authorized bootstrap records a digest and Arun accepts it.
# --sidecar FILE is login-node safe. It reads a recorded sha256sum line.
# --image FILE hashes the SIF. Use it only on a compute node.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIN_FILE="${SCRIPT_DIR}/expected_image.sha256"

usage() {
  echo "usage: $0 --sidecar FILE | --image FILE" >&2
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
fi

mode=$1
path=$2

case "$mode" in
  --sidecar|--image)
    ;;
  *)
    usage
    ;;
esac

if [[ ! -f "$PIN_FILE" ]]; then
  echo "missing image pin file: $PIN_FILE" >&2
  exit 1
fi

EXPECTED_IMAGE_SHA256=$(tr -d ' \t\r\n' < "$PIN_FILE")
if [[ ! "$EXPECTED_IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "expected_image.sha256 is not a 64-character lowercase hex digest" >&2
  exit 1
fi
if [[ ! -f "$path" ]]; then
  echo "missing digest source: $path" >&2
  exit 1
fi

case "$mode" in
  --sidecar)
    actual=$(awk '{print $1}' "$path")
    ;;
  --image)
    actual=$(sha256sum "$path" | awk '{print $1}')
    ;;
  *)
    usage
    ;;
esac

if [[ "$actual" != "$EXPECTED_IMAGE_SHA256" ]]; then
  echo "image digest mismatch: got ${actual} expected ${EXPECTED_IMAGE_SHA256}" >&2
  exit 1
fi

printf '%s\n' "$actual"
