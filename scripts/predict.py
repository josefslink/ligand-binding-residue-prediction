#!/usr/bin/env python3
"""
Run the full test pipeline: PDB extraction → ProtT5 embeddings → CNN ensemble →
submission CSV. Intermediate files go to data/processed/; the submission is written
to submissions/day1_pretrained_baseline.csv.

Usage:
    python scripts/predict.py
    python scripts/predict.py --threshold 0.5 --max-len 1000
    python scripts/predict.py --skip-embed   # reuse existing h5
    python scripts/predict.py --skip-embed --threshold 0.643 \\
        --model-prefix trained/ft_k9_f256_checkpoint   # final submitted model
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "bindPredict"))

from bindembed_addon import config
from bindembed_addon.structure.extractor import (
    extract_from_pdb_dir, load_fasta, load_residue_maps, resid_token,
)

TEST_DIR        = config.TEST_DIR
PROCESSED_DIR   = config.PROCESSED_DIR
BINDPREDICT_DIR = config.BINDPREDICT_DIR
DEFAULT_MODEL_PREFIX = config.PRETRAINED_CHECKPOINT_PREFIX

FASTA_OUT       = config.TEST_FASTA
MAPS_OUT        = config.TEST_MAPS
H5_OUT          = config.TEST_H5
PREDICTIONS_DIR = PROCESSED_DIR / "test_predictions"
SUBMISSION_OUT  = config.SUBMISSIONS_DIR / "day1_pretrained_baseline.csv"

PROTT5_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"


# Stage 2 — ProtT5 embedding

def _embed_sequence(model, tokenizer, seq: str, device, max_len: int) -> np.ndarray:
    """
    Embed one amino-acid sequence → np.float32 (L, 1024).
    Chunks sequences longer than max_len to fit GPU memory.
    """
    # Replace ambiguous / rare AAs
    seq_clean = re.sub(r"[UZOB]", "X", seq)
    L = len(seq_clean)

    if L <= max_len:
        chunks = [seq_clean]
    else:
        chunks = [seq_clean[i: i + max_len] for i in range(0, L, max_len)]

    parts = []
    for chunk in chunks:
        spaced = " ".join(list(chunk))
        encoding = tokenizer(spaced, return_tensors="pt",
                             add_special_tokens=True).to(device)
        with torch.no_grad():
            out = model(**encoding)
        # last_hidden_state: (1, chunk_L + 1, 1024)  — +1 for EOS token
        emb = out.last_hidden_state[0, : len(chunk), :].float().cpu().numpy()
        parts.append(emb)

    return np.concatenate(parts, axis=0)   # (L, 1024)


def embed_prott5(fasta_dict: dict, h5_path: Path, max_len: int, device):
    """
    Generate ProtT5 embeddings for all chains in fasta_dict.
    Saves float32 (L, 1024) arrays to h5_path keyed by chain key.
    Resumable: chains already in h5 are skipped.
    """
    print(f"\nstage 2: ProtT5 embeddings → {h5_path}")
    from transformers import T5EncoderModel, T5Tokenizer

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            done = set(f.keys())
        print(f"  Resuming: {len(done)} chains already embedded, "
              f"{len(fasta_dict) - len(done)} remaining.")

    remaining = {k: v for k, v in fasta_dict.items() if k not in done}
    if not remaining:
        print("  all chains already embedded")
        return

    tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL, torch_dtype=torch.float16)
    model = model.to(device).eval()
    print(f"  model on {device}, {len(remaining)} chains")

    with h5py.File(h5_path, "a") as f:
        for idx, (key, seq) in enumerate(remaining.items()):
            if idx % 100 == 0:
                print(f"    {idx}/{len(remaining)}  (current: {key}, len={len(seq)})")
            emb = _embed_sequence(model, tokenizer, seq, device, max_len)
            assert emb.shape == (len(seq), 1024), (
                f"{key}: expected ({len(seq)}, 1024), got {emb.shape}"
            )
            f.create_dataset(key, data=emb.astype(np.float32))

    print(f"  {len(fasta_dict)} chains in {h5_path} "
          f"({h5_path.stat().st_size / 1e9:.2f} GB)")

    del model   # free GPU memory before CNN inference
    torch.cuda.empty_cache()


# Stage 3 — CNN inference (reuses bindPredict checkpoints)

def _load_cnn_models(model_prefix: str, device) -> list:
    """
    Load 5 CNN checkpoints from model_prefix{1..5}.pt.

    Handles two formats:
      - Full model object (pretrained: torch.save(model, …))
      - State dict     (fine-tuned:  torch.save(model.state_dict(), …))
    """
    sys.path.insert(0, str(BINDPREDICT_DIR))
    from architectures import CNN2Layers   # noqa: F401

    models = []
    for i in range(1, 6):
        path = f"{model_prefix}{i}.pt"
        obj  = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, dict):
            # Infer architecture from state_dict so any kernel/features work
            w        = obj["conv1.0.weight"]        # (features, 1024, kernel)
            features = w.shape[0]
            kernel   = w.shape[2]
            padding  = (kernel - 1) // 2
            m = CNN2Layers(1024, features, kernel, 1, padding, 0.5)
            m.load_state_dict(obj)
        else:
            m = obj
        m.to(device).eval()
        models.append(m)
    return models


def _predict_chain(models: list, embedding: np.ndarray, device) -> np.ndarray:
    """
    Run one chain through all 5 CNNs and return the averaged probabilities.
    embedding : np.float32 (L, 1024)
    returns   : np.float32 (L, 3)  — [metal, nucleic, small-mol]
    """
    L = embedding.shape[0]

    # Build feature tensor (1025, L): 1024 embedding + 1 padding-mask channel
    feat = np.zeros((1025, L), dtype=np.float32)
    feat[:1024, :] = embedding.T   # (1024, L)
    feat[1024, :]  = 1.0           # mark as real positions (not padded)

    x = torch.from_numpy(feat).unsqueeze(0).to(device)   # (1, 1025, L)
    x_1024 = x[:, :-1, :]                                # (1, 1024, L)

    sigm = torch.nn.Sigmoid()
    preds = []
    for model in models:
        with torch.no_grad():
            out = model(x_1024)     # (3, L) — CNN2Layers squeezes batch dim
            out = sigm(out)
            preds.append(out.cpu().float().numpy())      # (3, L)

    avg = np.mean(preds, axis=0)   # (3, L)
    return avg.T                   # (L, 3)


def run_cnn_inference(fasta_dict: dict, h5_path: Path, model_prefix: str) -> dict:
    """
    Run the 5-checkpoint CNN ensemble on all chains.
    Returns  predictions: { chain_key: np.float32 (L, 3) }
    Processes one chain at a time — no need to hold all embeddings in RAM.
    """
    print("\nstage 3: CNN inference (5-checkpoint ensemble)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    models = _load_cnn_models(model_prefix, device)
    print(f"  Loaded 5 checkpoints.")

    predictions = {}
    chain_keys = list(fasta_dict.keys())

    with h5py.File(h5_path, "r") as f:
        for idx, key in enumerate(chain_keys):
            if idx % 200 == 0:
                print(f"  {idx}/{len(chain_keys)}  (current: {key})")
            emb = np.array(f[key], dtype=np.float32)   # (L, 1024)
            predictions[key] = _predict_chain(models, emb, device)

    print(f"  {len(predictions)} chains predicted")
    return predictions


# Stage 4 — Submission formatting

def format_submission(
    predictions: dict,
    residue_maps: dict,
    fasta_dict: dict,
    threshold: float,
    out_csv: Path,
):
    """
    For each protein (= integer id) collect all its chains, apply threshold
    on small-molecule channel (index 2), map to chain_resid tokens, write CSV.
    """
    print(f"\nstage 4: submission (threshold={threshold})")

    protein_ids = sorted(
        {key.rsplit("_", 1)[0] for key in predictions},
        key=lambda x: int(x) if x.isdigit() else x,
    )

    rows = []
    for pid in protein_ids:
        chain_keys = [k for k in predictions if k.rsplit("_", 1)[0] == pid]

        tokens = []
        for ck in sorted(chain_keys):
            probs = predictions[ck][:, 2]   # small-molecule channel
            rmap  = residue_maps[ck]
            for pos_idx, prob in enumerate(probs):
                if prob >= threshold:
                    e = rmap[pos_idx]
                    tokens.append(resid_token(e["chain"], e["resnum"], e["icode"]))

        rows.append({"id": int(pid) if pid.isdigit() else pid,
                     "prediction": " ".join(tokens)})

    df = pd.DataFrame(rows, columns=["id", "prediction"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_nonempty = (df["prediction"] != "").sum()
    print(f"  {len(df)} proteins written; {n_nonempty} with ≥1 predicted residue.")
    print(f"  Submission → {out_csv}")
    return df


def main():
    """Run the full test pipeline and write a submission CSV."""
    parser = argparse.ArgumentParser(description="Run pretrained bindEmbed21DL on test data.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for small-mol binding (default: 0.5)")
    parser.add_argument("--max-len", type=int, default=1000,
                        help="Max residues per ProtT5 chunk (default: 1000)")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip Stage 2 (use existing h5)")
    parser.add_argument(
        "--model-prefix", type=str, default=DEFAULT_MODEL_PREFIX,
        help=(
            "Path prefix for CNN checkpoint files (appended with 1.pt…5.pt). "
            "Default: pretrained bindPredict checkpoints. "
            "Use trained/finetune_checkpoint for fine-tuned models."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM")

    # Stage 1 — extract sequences
    if FASTA_OUT.exists() and MAPS_OUT.exists():
        print(f"\nstage 1: cached sequences from {FASTA_OUT}")
        fasta_dict = load_fasta(FASTA_OUT)
        residue_maps = load_residue_maps(MAPS_OUT)
        print(f"  Loaded {len(fasta_dict)} chains.")
    else:
        print(f"\nstage 1: extracting sequences from {TEST_DIR}")
        fasta_dict, residue_maps = extract_from_pdb_dir(TEST_DIR, FASTA_OUT, MAPS_OUT)

    # Stage 2 — embed
    if not args.skip_embed:
        embed_prott5(fasta_dict, H5_OUT, args.max_len, device)
    else:
        print("\nstage 2: skipped (--skip-embed)")
        if not H5_OUT.exists():
            sys.exit(f"ERROR: --skip-embed set but {H5_OUT} does not exist.")

    # Stage 3 — CNN inference
    predictions = run_cnn_inference(fasta_dict, H5_OUT, args.model_prefix)

    # Stage 4 — format submission
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df = format_submission(predictions, residue_maps, fasta_dict,
                           args.threshold, SUBMISSION_OUT)

    # Quick sanity check against sample_submission.csv schema
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV)
    assert list(df.columns) == ["id", "prediction"], "Column mismatch"
    assert set(df["id"]) == set(sample["id"]), \
        f"Protein ID mismatch: {len(set(df['id']) - set(sample['id']))} extra ids"
    print("\nSanity check: column names and protein IDs match sample_submission.csv")


if __name__ == "__main__":
    main()
