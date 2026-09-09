#!/usr/bin/env bash
# Fetches the vendored dependency (Rostlab/bindPredict) at the exact commit
# used to produce the results in this repo. Not vendored in git — third_party/
# is gitignored — so this must be run once before any training or inference
# script that imports `architectures` or the pretrained checkpoints.
#
# Usage:
#   bash scripts/setup_bindpredict.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/third_party/bindPredict"
COMMIT="c9b12e7"

if [ -d "${TARGET_DIR}" ]; then
    echo "third_party/bindPredict already exists at ${TARGET_DIR} — skipping clone."
    echo "Delete it first if you want a clean re-fetch."
    exit 0
fi

mkdir -p "${REPO_ROOT}/third_party"
git clone https://github.com/Rostlab/bindPredict.git "${TARGET_DIR}"
git -C "${TARGET_DIR}" checkout "${COMMIT}"

echo "bindPredict @ ${COMMIT} fetched to ${TARGET_DIR}"
