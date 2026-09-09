"""
Wraps metric.py's IoU logic so local results stay in sync with the grader.
Provides a CSV-based interface (mirrors metric.py exactly) and a fast in-memory
interface for threshold sweeping. The score is a single pooled Jaccard over all
residues — do not average per-protein when tuning the threshold.
"""

from __future__ import annotations

import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

from bindembed_addon import config
from bindembed_addon.structure.extractor import resid_token


# Load metric.py's parse_submission function once at import time

def _load_parse_submission():
    """Import parse_submission directly from the course-provided metric.py at the repo root."""
    metric_path = config.REPO_ROOT / "metric.py"
    spec = importlib.util.spec_from_file_location("metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_submission


_parse_submission = _load_parse_submission()


# Interface 1 — CSV-based (mirrors metric.py exactly)

def compute_iou_from_csvs(submission_path: Path, target_path: Path) -> float:
    """
    Compute pooled Jaccard score using metric.py's exact logic.

    submission_path: id,prediction CSV (space-separated chain_resid tokens)
    target_path    : id,residue_id,true CSV (long format, SM3a output)
    """
    df_submission = pd.read_csv(submission_path)
    df_submission = _parse_submission(df_submission)

    df_target = pd.read_csv(target_path)

    df_merged = pd.merge(df_target, df_submission, on=["id", "residue_id"], how="left")
    df_merged["prediction"] = df_merged["prediction"].fillna(0)

    return float(jaccard_score(df_merged["true"], df_merged["prediction"]))


# Interface 2 — Fast in-memory sweep (no CSV I/O per threshold)

def compute_iou_fast(
    predictions: dict[str, np.ndarray],
    residue_maps: dict[str, list[dict]],
    labels: dict[str, np.ndarray],
    threshold: float,
    protein_ids: list[str] | None = None,
) -> float:
    """
    Compute pooled IoU for a given threshold entirely in memory.

    predictions: { chain_key: float32 (L, 3) } — CNN output, small-mol = col 2
    labels     : { chain_key: int8 (L,) } — SM2 output
    protein_ids: if given, restrict to these protein IDs only
    """
    pid_set = None if protein_ids is None else set(str(p) for p in protein_ids)

    y_true_all = []
    y_pred_all = []

    for chain_key, label_arr in labels.items():
        pid = chain_key.rsplit("_", 1)[0]
        if pid_set is not None and pid not in pid_set:
            continue
        if chain_key not in predictions:
            continue

        probs = predictions[chain_key][:, 2]   # small-mol channel
        pred_binary = (probs >= threshold).astype(np.int8)

        y_true_all.append(label_arr)
        y_pred_all.append(pred_binary)

    if not y_true_all:
        return 0.0

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    return float(jaccard_score(y_true, y_pred, zero_division=0))


def sweep_threshold(
    predictions: dict[str, np.ndarray],
    residue_maps: dict[str, list[dict]],
    labels: dict[str, np.ndarray],
    protein_ids: list[str] | None = None,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Sweep thresholds and return the best one for pooled IoU.

    thresholds defaults to 200 evenly-spaced values between 0.01 and 0.95.
    Returns (best_threshold, best_iou, thresholds, ious).
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.95, 200)

    ious = np.array([
        compute_iou_fast(predictions, residue_maps, labels, t, protein_ids)
        for t in thresholds
    ])

    best_idx = int(np.argmax(ious))
    return float(thresholds[best_idx]), float(ious[best_idx]), thresholds, ious
