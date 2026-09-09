#!/usr/bin/env python3
"""
Embed train or test sequences with ProtT5, writing per-chain float32 (L, 1024)
arrays to HDF5 under data/processed/. The h5 key is "{protein_id}_{chain}"
(e.g. "0_A"). Already-embedded chains are skipped, so the script is resumable.

Usage:
    python scripts/embed_sequences.py --split train
    python scripts/embed_sequences.py --split test
    python scripts/embed_sequences.py --split train --max-len 1000
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bindembed_addon import config
from bindembed_addon.structure.extractor import extract_from_pdb_dir, load_fasta, load_residue_maps

PROTT5_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"

SPLIT_DIRS = {
    "train": config.TRAIN_DIR,
    "test":  config.TEST_DIR,
}
PROCESSED_DIR = config.PROCESSED_DIR


def _embed_sequence(model, tokenizer, seq: str, device, max_len: int) -> np.ndarray:
    """Embed one sequence → float32 (L, 1024). Chunks if len > max_len."""
    seq_clean = re.sub(r"[UZOB]", "X", seq)
    chunks = (
        [seq_clean] if len(seq_clean) <= max_len
        else [seq_clean[i: i + max_len] for i in range(0, len(seq_clean), max_len)]
    )

    parts = []
    for chunk in chunks:
        encoding = tokenizer(
            " ".join(list(chunk)),
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            out = model(**encoding)
        # Strip EOS token: last_hidden_state is (1, chunk_L+1, 1024)
        emb = out.last_hidden_state[0, : len(chunk), :].float().cpu().numpy()
        parts.append(emb)

    return np.concatenate(parts, axis=0)  # (L, 1024)


def embed_all(fasta_dict: dict, h5_path: Path, max_len: int, device):
    """Embed all chains in fasta_dict → h5_path. Resumable."""
    from transformers import T5EncoderModel, T5Tokenizer

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            done = set(f.keys())

    remaining = {k: v for k, v in fasta_dict.items() if k not in done}
    if not remaining:
        print("  all chains already embedded")
        return

    if done:
        print(f"  Resuming: {len(done)} chains done, {len(remaining)} remaining.")
    else:
        print(f"  Embedding {len(remaining)} chains…")

    tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL, torch_dtype=torch.float16)
    model = model.to(device).eval()
    print(f"  Model on {device}.")

    with h5py.File(h5_path, "a") as f:
        for idx, (key, seq) in enumerate(remaining.items()):
            if idx % 100 == 0:
                print(f"    {idx}/{len(remaining)}  key={key}  len={len(seq)}")
            emb = _embed_sequence(model, tokenizer, seq, device, max_len)
            assert emb.shape == (len(seq), 1024), \
                f"{key}: expected ({len(seq)},1024) got {emb.shape}"
            f.create_dataset(key, data=emb.astype(np.float32))

    size_gb = h5_path.stat().st_size / 1e9
    print(f"  {h5_path}  ({size_gb:.2f} GB)")

    del model
    torch.cuda.empty_cache()


def main():
    """Extract sequences for one split (cached) and embed them with ProtT5."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   choices=["train", "test"], required=True)
    parser.add_argument("--max-len", type=int, default=1000,
                        help="Max residues per ProtT5 chunk (default: 1000)")
    args = parser.parse_args()

    split     = args.split
    pdb_dir   = SPLIT_DIRS[split]
    fasta_out = PROCESSED_DIR / f"{split}_sequences.fasta"
    maps_out  = PROCESSED_DIR / f"{split}_residue_maps.json"
    h5_out    = PROCESSED_DIR / f"{split}_prott5_embeddings.h5"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Split : {split}")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  {torch.cuda.get_device_name(0)} | {props.total_memory/1e9:.1f} GB VRAM")

    # Stage 1 — extract sequences (cached)
    if fasta_out.exists() and maps_out.exists():
        print(f"\nstage 1: cached {split} sequences")
        fasta_dict = load_fasta(fasta_out)
        print(f"  {len(fasta_dict)} chains loaded.")
    else:
        print(f"\nstage 1: extracting {split} sequences from {pdb_dir}")
        fasta_dict, _ = extract_from_pdb_dir(pdb_dir, fasta_out, maps_out)

    # Stage 2 — embed
    print(f"\nstage 2: ProtT5 embeddings → {h5_out}")
    embed_all(fasta_dict, h5_out, args.max_len, device)


if __name__ == "__main__":
    main()
