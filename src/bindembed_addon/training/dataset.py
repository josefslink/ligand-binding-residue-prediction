"""
PyTorch Dataset that streams per-chain (embedding, label) pairs from an HDF5 file.
Label channel 2 is the binding class; channels 0 and 1 are always zero. File handles
are opened lazily in __getitem__ so the dataset can be pickled by DataLoader workers.
"""

from __future__ import annotations

import numpy as np
import h5py
import torch.utils.data


class H5ChainDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that streams per-chain embeddings from an HDF5 file.

    Parameters
    ----------
    chain_keys : list of h5 dataset keys to include (e.g. ["0_A", "0_B", …])
    h5_path    : path to the HDF5 embeddings file
    labels     : { chain_key: int8 (L,) } — SM2 output; chains missing from
                 this dict get all-zero labels (treated as non-binding)
    """

    def __init__(self, chain_keys: list[str], h5_path, labels: dict):
        self.keys    = list(chain_keys)
        self.h5_path = str(h5_path)
        self.labels  = labels
        self._f: h5py.File | None = None   # opened lazily per worker process

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")

        key = self.keys[idx]
        emb = np.array(self._f[key], dtype=np.float32)   # (L, 1024)
        L   = len(emb)

        lbl = np.zeros((L, 3), dtype=np.float32)
        if key in self.labels:
            lbl[:, 2] = self.labels[key].astype(np.float32)

        return emb, lbl    # (L, 1024), (L, 3)
