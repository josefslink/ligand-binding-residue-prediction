#!/usr/bin/env python3
"""
Run CNN inference on the val split, sweep 200 thresholds, and report the best
local IoU. Pass --write-submission to also write a tuned prediction CSV.

Usage:
    python scripts/validate_baseline.py
    python scripts/validate_baseline.py --write-submission
"""

import argparse
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
from bindembed_addon.structure.extractor import load_residue_maps
from bindembed_addon.labels.aligner      import load_all_labels
from bindembed_addon.splits.by_protein   import load_or_make_split
from bindembed_addon.metrics.iou         import sweep_threshold, compute_iou_fast

TRAIN_H5    = config.TRAIN_H5
TRAIN_MAPS  = config.TRAIN_MAPS
TRAIN_CSV   = config.TRAIN_CSV
SPLIT_OUT   = config.SPLIT_JSON
REPORTS_DIR = config.REPORTS_DIR

DEFAULT_MODEL_PREFIX = config.PRETRAINED_CHECKPOINT_PREFIX


# CNN inference (identical logic to run_pretrained_baseline.py Stage 3)

def _load_cnn_models(model_prefix: str, device) -> list:
    """
    Load 5 CNN checkpoints from model_prefix{1..5}.pt.

    Handles two formats:
      - Full model object (pretrained checkpoints saved with torch.save(model, …))
      - State dict (fine-tuned checkpoints saved with torch.save(model.state_dict(), …))
    """
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
    """embedding: (L,1024) → returns (L,3) averaged over 5 checkpoints."""
    L = embedding.shape[0]
    feat = np.zeros((1025, L), dtype=np.float32)
    feat[:1024, :] = embedding.T
    feat[1024, :]  = 1.0
    x = torch.from_numpy(feat).unsqueeze(0).to(device)
    x_1024 = x[:, :-1, :]

    sigm = torch.nn.Sigmoid()
    preds = []
    for model in models:
        with torch.no_grad():
            out = model(x_1024)
            preds.append(sigm(out).cpu().float().numpy())  # (3, L)

    avg = np.mean(preds, axis=0)   # (3, L)
    return avg.T                   # (L, 3)


def run_inference_on_split(
    val_ids: list[str], model_prefix: str, device
) -> dict[str, np.ndarray]:
    """Run CNN on all chains belonging to val_ids. Returns {chain_key: (L,3)}."""
    print(f"\nCNN inference on {len(val_ids)} val proteins")

    val_pid_set = set(str(p) for p in val_ids)
    with h5py.File(TRAIN_H5, "r") as f:
        val_chain_keys = [k for k in f.keys()
                          if k.rsplit("_", 1)[0] in val_pid_set]

    print(f"  {len(val_chain_keys)} chains to predict.")
    models = _load_cnn_models(model_prefix, device)
    print(f"  Loaded 5 checkpoints.")

    predictions: dict[str, np.ndarray] = {}
    with h5py.File(TRAIN_H5, "r") as f:
        for idx, key in enumerate(val_chain_keys):
            if idx % 200 == 0:
                print(f"  {idx}/{len(val_chain_keys)}  ({key})")
            emb = np.array(f[key], dtype=np.float32)
            predictions[key] = _predict_chain(models, emb, device)

    print(f"  {len(predictions)} chains predicted")
    return predictions


# Format submission CSV at a given threshold (for writing the tuned CSV)

def format_submission(
    predictions: dict[str, np.ndarray],
    residue_maps: dict[str, list[dict]],
    threshold: float,
    protein_ids: list[str],
    out_path: Path,
):
    """Threshold predictions, map to chain_resid tokens, write a submission CSV."""
    from bindembed_addon.structure.extractor import resid_token

    pid_set = set(str(p) for p in protein_ids)
    protein_id_list = sorted(
        pid_set, key=lambda x: int(x) if x.isdigit() else x
    )

    rows = []
    for pid in protein_id_list:
        chain_keys = [k for k in predictions if k.rsplit("_", 1)[0] == pid]
        tokens = []
        for ck in sorted(chain_keys):
            probs = predictions[ck][:, 2]
            rmap  = residue_maps[ck]
            for pos_idx, prob in enumerate(probs):
                if prob >= threshold:
                    e = rmap[pos_idx]
                    tokens.append(resid_token(e["chain"], e["resnum"], e["icode"]))
        rows.append({
            "id":         int(pid) if pid.isdigit() else pid,
            "prediction": " ".join(tokens),
        })

    df = pd.DataFrame(rows, columns=["id", "prediction"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    n_nonempty = (df["prediction"] != "").sum()
    print(f"  {len(df)} proteins, {n_nonempty} non-empty → {out_path}")


def main():
    """Run inference + threshold sweep on the val split and report the best local IoU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-submission", action="store_true",
                        help="Write a resubmission CSV at the optimal threshold")
    parser.add_argument(
        "--model-prefix", type=str, default=DEFAULT_MODEL_PREFIX,
        help=(
            "Path prefix for CNN checkpoint files (appended with 1.pt…5.pt). "
            "Default: pretrained bindPredict checkpoints. "
            "Use trained/finetune_checkpoint for fine-tuned models."
        ),
    )
    args = parser.parse_args()

    for path in [TRAIN_H5, TRAIN_MAPS, TRAIN_CSV]:
        if not path.exists():
            sys.exit(f"ERROR: required file not found: {path}\n"
                     "Run  python scripts/embed_sequences.py --split train  first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Split
    print("\ntrain/val split")
    train_ids, val_ids = load_or_make_split(
        TRAIN_CSV, split_out=SPLIT_OUT, val_fraction=0.12, seed=42
    )

    # 2. Load residue maps + labels
    print("\nloading residue maps + labels")
    residue_maps = load_residue_maps(TRAIN_MAPS)
    all_labels   = load_all_labels(TRAIN_CSV, residue_maps)
    val_labels   = {k: v for k, v in all_labels.items()
                    if k.rsplit("_", 1)[0] in set(str(p) for p in val_ids)}
    n_pos = sum(arr.sum() for arr in val_labels.values())
    n_tot = sum(len(arr) for arr in val_labels.values())
    print(f"  Val: {len(val_ids)} proteins, {n_tot:,} residues, "
          f"{n_pos:,} binding ({100*n_pos/n_tot:.1f}%)")

    # 3. CNN inference
    predictions = run_inference_on_split(val_ids, args.model_prefix, device)

    # 4. Threshold sweep
    print("\nthreshold sweep (200 values, 0.01 → 0.95)")
    best_t, best_iou, thresholds, ious = sweep_threshold(
        predictions, residue_maps, val_labels,
        protein_ids=val_ids,
    )

    print(f"\n  IoU at threshold=0.50 : {compute_iou_fast(predictions, residue_maps, val_labels, 0.50, val_ids):.4f}")
    print(f"  Best threshold        : {best_t:.3f}")
    print(f"  Best local val IoU    : {best_iou:.4f}")
    print(f"\n  (Public LB at t=0.5 was 0.162 — compare to tuned val IoU above)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df = pd.DataFrame({"threshold": thresholds, "iou": ious})
    sweep_path = REPORTS_DIR / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n  Sweep curve saved → {sweep_path}")

    top10 = sweep_df.nlargest(10, "iou")
    print("\n  Top 10 thresholds by val IoU:")
    print(top10.to_string(index=False))

    # 5. Optionally write tuned submission
    if args.write_submission:
        out = config.SUBMISSIONS_DIR / f"baseline_tuned_t{best_t:.3f}.csv"
        print(f"\nwriting tuned submission → {out}")
        print("  NOTE: this writes predictions for val proteins only.")
        format_submission(predictions, residue_maps, best_t, val_ids, out)


if __name__ == "__main__":
    main()
