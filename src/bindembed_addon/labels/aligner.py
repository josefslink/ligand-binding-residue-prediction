"""
Turns train.csv's space-separated binding-residue tokens into per-chain binary
label arrays. Array index i lines up with residue_maps[chain_key][i].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def parse_token(token: str) -> tuple[str, int, str]:
    """Split 'A_136' or 'A_136_A' into (chain, resnum, icode)."""
    parts = token.split("_")
    if len(parts) == 2:
        return parts[0], int(parts[1]), ""
    elif len(parts) == 3:
        return parts[0], int(parts[1]), parts[2]
    else:
        raise ValueError(f"Unexpected token format: {token!r}")


def _token_to_map_key(chain: str, resnum: int, icode: str) -> tuple[str, int, str]:
    return chain, resnum, icode


def align_protein(
    protein_id: str,
    residue_maps: dict[str, list[dict]],
    resid_str: str,
) -> dict[str, np.ndarray]:
    """
    Build per-chain binary label arrays for one protein.

    protein_id filters residue_maps to the relevant chains (keys like "0_A", "0_B").
    resid_str is the raw space-separated token string from train.csv.
    Returns { chain_key: int8 (L,) }.
    """
    binding_set: set[tuple[str, int, str]] = set()
    if isinstance(resid_str, str) and resid_str.strip():
        for tok in resid_str.strip().split():
            tok = tok.strip()
            if tok:
                binding_set.add(parse_token(tok))

    labels: dict[str, np.ndarray] = {}
    chain_keys = [k for k in residue_maps if k.rsplit("_", 1)[0] == protein_id]
    for ck in chain_keys:
        rmap = residue_maps[ck]
        arr = np.zeros(len(rmap), dtype=np.int8)
        for i, entry in enumerate(rmap):
            key = (entry["chain"], entry["resnum"], entry["icode"])
            if key in binding_set:
                arr[i] = 1
        labels[ck] = arr

    return labels


def load_all_labels(
    train_csv: Path,
    residue_maps: dict[str, list[dict]],
) -> dict[str, np.ndarray]:
    """Build aligned label arrays for every protein in train.csv. Returns {chain_key: int8 (L,)}."""
    df = pd.read_csv(train_csv, dtype={"id": str})
    df["resid"] = df["resid"].fillna("")

    all_labels: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        pid = str(row["id"])
        chain_labels = align_protein(pid, residue_maps, row["resid"])
        all_labels.update(chain_labels)

    return all_labels
