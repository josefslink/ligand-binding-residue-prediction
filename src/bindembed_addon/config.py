"""
Central path configuration for the pipeline. Every path defaults to a location
relative to the repo root but can be overridden by an environment variable,
so the pipeline runs unmodified on a machine with a different data layout.

Env vars:
    BINDEMBED_REPO_ROOT     — repo root (default: parent of src/)
    BINDEMBED_DATA_DIR      — challenge data root (default: REPO_ROOT/data)
    BINDEMBED_PROCESSED_DIR — cached FASTA/residue-maps/HDF5 (default: DATA_DIR/processed)
    BINDEMBED_THIRD_PARTY_DIR — vendored bindPredict clone (default: REPO_ROOT/third_party)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(
    os.environ.get("BINDEMBED_REPO_ROOT", Path(__file__).resolve().parents[2])
)

DATA_DIR = Path(
    os.environ.get("BINDEMBED_DATA_DIR", REPO_ROOT / "data")
)

PROCESSED_DIR = Path(
    os.environ.get("BINDEMBED_PROCESSED_DIR", DATA_DIR / "processed")
)

THIRD_PARTY_DIR = Path(
    os.environ.get("BINDEMBED_THIRD_PARTY_DIR", REPO_ROOT / "third_party")
)

BINDPREDICT_DIR = THIRD_PARTY_DIR / "bindPredict"

TRAIN_DIR = DATA_DIR / "train_data"
TEST_DIR = DATA_DIR / "test_data"
TRAIN_CSV = DATA_DIR / "train.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

TRAIN_H5 = PROCESSED_DIR / "train_prott5_embeddings.h5"
TEST_H5 = PROCESSED_DIR / "test_prott5_embeddings.h5"
TRAIN_MAPS = PROCESSED_DIR / "train_residue_maps.json"
TEST_MAPS = PROCESSED_DIR / "test_residue_maps.json"
TRAIN_FASTA = PROCESSED_DIR / "train_sequences.fasta"
TEST_FASTA = PROCESSED_DIR / "test_sequences.fasta"
SPLIT_JSON = PROCESSED_DIR / "splits" / "split.json"

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
TRAINED_DIR = REPO_ROOT / "trained"

PRETRAINED_CHECKPOINT_PREFIX = str(BINDPREDICT_DIR / "trained_models" / "checkpoint")
