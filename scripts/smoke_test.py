"""
End-to-end smoke test for third_party/bindPredict: creates fake embeddings, runs
the 5-checkpoint ensemble, and validates output shapes and column layout.
bindPredict does not embed sequences itself — embeddings must be pre-computed
(h5 key = protein ID, value = float32 (L, 1024)). The small-molecule output is
channel 2 of the (L, 3) prediction array.

This reimplements bindEmbed21DL.prediction_pipeline's loop rather than calling
it directly, because that function loads checkpoints via bindPredict's own
(vendored, unmodified) config.FileManager.load_classifier_torch(), which omits
weights_only=False and fails under torch>=2.6's stricter torch.load default.
scripts/predict.py, validate_baseline.py, and finetune_cnn.py all load
checkpoints directly with weights_only=False and are unaffected — this
incompatibility is confined to bindPredict's own high-level entry point, which
only this smoke test exercises. See THIRD_PARTY.md.
"""

import sys
import os
import tempfile
import textwrap
from pathlib import Path

import h5py
import numpy as np
import torch

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from bindembed_addon import config

BINDPREDICT_DIR = config.BINDPREDICT_DIR
MODEL_PREFIX = config.PRETRAINED_CHECKPOINT_PREFIX

TMP_DIR = Path(tempfile.mkdtemp(prefix="bindpredict_smoke_"))
EMBEDDINGS_H5 = TMP_DIR / "smoke_embeddings.h5"
FASTA_FILE = TMP_DIR / "smoke_test.fasta"
OUTPUT_DIR = TMP_DIR / "predictions"
OUTPUT_DIR.mkdir()

print(f"Smoke-test temp dir: {TMP_DIR}")

# 1. Build fake proteins (3 short sequences, random (L, 1024) embeddings)
FAKE_PROTEINS = {
    "prot_short":  "MKLVILNACDEFGHQRSTWY" * 5,    # 100 residues
    "prot_medium": "ACDEFGHQRSTWYMKLVILN" * 10,   # 200 residues
    "prot_long":   "QRSTWYMKLVILNACDEFGH" * 20,   # 400 residues
}

print("\ncreating fake embeddings")
with h5py.File(EMBEDDINGS_H5, "w") as f:
    for pid, seq in FAKE_PROTEINS.items():
        arr = np.random.rand(len(seq), 1024).astype(np.float32)
        f.create_dataset(pid, data=arr)
        print(f"  {pid}: shape {arr.shape}")

print("\nwriting FASTA")
with open(FASTA_FILE, "w") as f:
    for pid, seq in FAKE_PROTEINS.items():
        f.write(f">{pid}\n{seq}\n")
        print(f"  >{pid}  ({len(seq)} aa)")

# 2. Monkey-patch config.py before importing bindPredict
#    (bindPredict's `from config import FileSetter` runs at import time)
sys.path.insert(0, str(BINDPREDICT_DIR))

import config as bp_config

bp_config.FileSetter.embeddings_input  = staticmethod(lambda: str(EMBEDDINGS_H5))
bp_config.FileSetter.predictions_folder = staticmethod(lambda: str(OUTPUT_DIR))
bp_config.FileSetter.query_set         = staticmethod(lambda: str(FASTA_FILE))

# 3. Run the 5-checkpoint ensemble (loads checkpoints ourselves, averages, writes output)
#    Equivalent to BindEmbed21DL.prediction_pipeline(), except checkpoints are loaded
#    with weights_only=False directly instead of via FileManager.load_classifier_torch()
#    — see module docstring.
print("\nrunning prediction pipeline")
from config import FileManager
from data_preparation import ProteinInformation
from ml_predictor import MLPredictor

device = "cuda:0" if torch.cuda.is_available() else "cpu"

query_sequences = FileManager.read_fasta(str(FASTA_FILE))
query_ids = list(query_sequences.keys())

sequences, max_length, labels = ProteinInformation.get_data_predictions(query_ids, str(FASTA_FILE))
embeddings = FileManager.read_embeddings(str(EMBEDDINGS_H5))

proteins = dict()
for i in range(5):
    print(f"Load model {i + 1}/5")
    model_path = f"{MODEL_PREFIX}{i + 1}.pt"
    model = torch.load(model_path, map_location=device, weights_only=False)

    print("Calculate predictions")
    ml_predictor = MLPredictor(model)
    curr_proteins = ml_predictor.predict_per_protein(
        query_ids, sequences, embeddings, labels, max_length,
    )

    for k, curr_prot in curr_proteins.items():
        if k in proteins:
            proteins[k].add_predictions(curr_prot.predictions)
        else:
            proteins[k] = curr_prot

for prot in proteins.values():
    prot.normalize_predictions(5)

FileManager.write_predictions(proteins, str(OUTPUT_DIR), cutoff=0.5, ri=False)

# 4. Validate output
print("\nvalidation")
all_ok = True
for pid, seq in FAKE_PROTEINS.items():
    expected_len = len(seq)
    out_file = OUTPUT_DIR / f"{pid}.bindPredict_out"

    if not out_file.exists():
        print(f"  FAIL  {pid}: output file not found at {out_file}")
        all_ok = False
        continue

    with open(out_file) as fh:
        lines = [l for l in fh if not l.startswith("Position")]  # skip header

    actual_len = len(lines)
    status = "OK  " if actual_len == expected_len else "FAIL"
    print(f"  {status}  {pid}: expected {expected_len} rows, got {actual_len}")
    if actual_len != expected_len:
        all_ok = False

    # Spot-check: small-molecule probability is in column index 5 (0-based, after header)
    # Header: Position Metal.Proba Metal.Class Nuclear.Proba Nuclear.Class Small.Proba Small.Class Any.Class
    first_row = lines[0].strip().split("\t")
    try:
        small_proba = float(first_row[5])
        assert 0.0 <= small_proba <= 1.0, f"small_proba={small_proba} out of [0,1]"
    except (IndexError, ValueError, AssertionError) as e:
        print(f"  FAIL  {pid}: malformed first row — {e}")
        all_ok = False

print("\nprot_short output (first 3 rows):")
short_out = OUTPUT_DIR / "prot_short.bindPredict_out"
if short_out.exists():
    with open(short_out) as fh:
        for i, line in enumerate(fh):
            if i < 4:   # header + 3 data rows
                print(" ", line.rstrip())

print("\nin-memory predictions (prot_short):")
prot = proteins.get("prot_short")
if prot is not None:
    arr = prot.predictions   # shape (L, 3)
    print(f"  predictions.shape: {arr.shape}  (expected: ({len(FAKE_PROTEINS['prot_short'])}, 3))")
    print(f"  small-molecule channel (col 2) — first 5 values: {arr[:5, 2]}")
    shape_ok = arr.shape == (len(FAKE_PROTEINS["prot_short"]), 3)
    if not shape_ok:
        all_ok = False
        print("  FAIL: shape mismatch")
    else:
        print("  OK: shape matches")

# 5. Final verdict
print()
if all_ok:
    print("=" * 60)
    print("  ALL CHECKS PASSED — bindEmbed21DL runs cleanly on this machine.")
    print("  bindPredict smoke test: SUCCESS")
    print("=" * 60)
else:
    print("=" * 60)
    print("  ONE OR MORE CHECKS FAILED — see messages above.")
    print("=" * 60)
    sys.exit(1)
