"""
Central seeding helper. The values actually used when the recorded results
were produced are unchanged by this refactor:
  - train/val split: seed=42 (bindembed_addon/splits/by_protein.py, its own
    random.Random instance — already isolated from global state)
  - from-scratch CNN training: seed = run_idx * 17, i.e. 17/34/51/68/85 for
    runs 1-5 (scripts/finetune_cnn.py)

Fine-tuning runs (loading a pretrained checkpoint) were not seeded beyond
the deterministic pretrained weights; this function is not called for them,
so that record is also unchanged.

Note: torch.backends.cudnn is left non-deterministic (its default), so
exact bit-for-bit reproduction on GPU is not guaranteed even with this seed
fixed — only the same order of operations up to cuDNN's own nondeterminism.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python's random, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
