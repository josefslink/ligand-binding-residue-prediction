"""
Deterministic by-protein train/val split. Splitting by protein (not by residue)
prevents sequence context from leaking between splits. Default: 12% val, seed=42.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


def make_split(
    all_ids: list[str],
    val_fraction: float = 0.12,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Randomly split protein IDs into train and val. Returns (train_ids, val_ids), both sorted."""
    ids = list(all_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = max(1, round(len(ids) * val_fraction))
    val_ids   = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


def load_or_make_split(
    train_csv: Path,
    split_out: Path | None = None,
    val_fraction: float = 0.12,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Load split from disk if it exists, otherwise compute and optionally save it.

    split_out: path to load/save split JSON; if None, not persisted.
    val_fraction and seed are ignored when loading from disk.
    """
    if split_out is not None and split_out.exists():
        with open(split_out) as fh:
            data = json.load(fh)
        train_ids = data["train"]
        val_ids   = data["val"]
        print(f"  Loaded split from {split_out}: "
              f"{len(train_ids)} train / {len(val_ids)} val")
        return train_ids, val_ids

    df = pd.read_csv(train_csv, dtype={"id": str})
    all_ids = df["id"].astype(str).tolist()

    train_ids, val_ids = make_split(all_ids, val_fraction=val_fraction, seed=seed)

    if split_out is not None:
        split_out.parent.mkdir(parents=True, exist_ok=True)
        with open(split_out, "w") as fh:
            json.dump({"train": train_ids, "val": val_ids}, fh, indent=2)
        print(f"  Split saved to {split_out}")

    print(f"  Split: {len(train_ids)} train / {len(val_ids)} val "
          f"(val_fraction={val_fraction}, seed={seed})")
    return train_ids, val_ids
