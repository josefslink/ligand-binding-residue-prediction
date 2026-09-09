"""
Builds the long-format target CSV (id, residue_id, true) required by metric.py.
Every residue — binding and non-binding — must appear; metric.py left-joins from
the target onto the submission, so absent residues silently inflate local IoU.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bindembed_addon.structure.extractor import resid_token
from bindembed_addon.labels.aligner import load_all_labels


def build_target_csv(
    protein_ids: list[str],
    residue_maps: dict[str, list[dict]],
    train_csv: Path,
    output_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the long-format target CSV required by metric.py.

    protein_ids restricts output to the val (or test) split.
    Returns a DataFrame with columns [id, residue_id, true].
    """
    all_labels = load_all_labels(train_csv, residue_maps)

    rows = []
    pid_set = set(str(p) for p in protein_ids)

    for chain_key, label_array in all_labels.items():
        pid = chain_key.rsplit("_", 1)[0]
        if pid not in pid_set:
            continue

        rmap = residue_maps[chain_key]
        for pos_idx, label in enumerate(label_array):
            entry = rmap[pos_idx]
            token = resid_token(entry["chain"], entry["resnum"], entry["icode"])
            rows.append({
                "id":         int(pid) if pid.isdigit() else pid,
                "residue_id": token,
                "true":       int(label),
            })

    df = pd.DataFrame(rows, columns=["id", "residue_id", "true"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_pos = int(df["true"].sum())
    n_total = len(df)
    if verbose:
        print(f"  Target CSV: {n_total:,} residues, {n_pos:,} binding "
              f"({100*n_pos/n_total:.1f}%), {df['id'].nunique()} proteins → {output_path}")

    return df
