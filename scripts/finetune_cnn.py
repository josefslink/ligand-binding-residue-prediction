#!/usr/bin/env python3
"""
Fine-tune (or train from scratch) the bindEmbed21DL CNN on the training split.
Runs 5 models — one per pretrained checkpoint, or 5 independent random seeds when
the architecture differs from the pretrained config (kernel=5, features=128).

Usage:
    python scripts/finetune_cnn.py                              # fine-tune pretrained
    python scripts/finetune_cnn.py --kernel 9 --features 256   # from scratch
    python scripts/finetune_cnn.py --kernel 9 --features 256 \\
        --out-prefix trained/ft_k9_f256_checkpoint
    python scripts/finetune_cnn.py --dry-run                    # 1 epoch, 1 run
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "bindPredict"))

from bindembed_addon import config
from bindembed_addon.structure.extractor import load_residue_maps
from bindembed_addon.labels.aligner      import load_all_labels
from bindembed_addon.splits.by_protein   import load_or_make_split
from bindembed_addon.metrics.iou         import compute_iou_fast
from bindembed_addon.training.dataset    import H5ChainDataset
from bindembed_addon.training.seeding    import set_seed

TRAIN_H5          = config.TRAIN_H5
TRAIN_MAPS        = config.TRAIN_MAPS
TRAIN_CSV         = config.TRAIN_CSV
SPLIT_OUT         = config.SPLIT_JSON
PRETRAINED_PREFIX = config.PRETRAINED_CHECKPOINT_PREFIX

# Pretrained architecture constants — changing either triggers from-scratch mode
PRETRAINED_KERNEL   = 5
PRETRAINED_FEATURES = 128


# Inference helpers

def _forward(model, emb: np.ndarray, device) -> np.ndarray:
    """emb: float32 (L, 1024) → float32 (L, 3) sigmoid probabilities."""
    x = torch.from_numpy(emb.T).unsqueeze(0).to(device)   # (1, 1024, L)
    with torch.no_grad():
        out = model(x)
    if out.dim() == 1:       # edge case: L=1
        out = out.unsqueeze(1)
    return torch.sigmoid(out).cpu().float().numpy().T      # (L, 3)


def run_val_inference(model, val_chains: list, device) -> dict:
    """Run inference on val_chains. Returns {chain_key: (L, 3) float32}."""
    model.eval()
    preds = {}
    with h5py.File(TRAIN_H5, "r") as f:
        for key in val_chains:
            if key not in f:
                continue
            preds[key] = _forward(model, np.array(f[key], dtype=np.float32), device)
    return preds


# Positive class weight

def compute_pos_weight(train_chains: list, labels: dict) -> float:
    """Positive class weight for BCEWithLogitsLoss: (n_negative / n_positive)."""
    n_pos = sum(int(labels[k].sum()) for k in train_chains if k in labels)
    n_tot = sum(len(labels[k])       for k in train_chains if k in labels)
    if n_pos == 0:
        raise ValueError("No positive labels found in train split.")
    return float(n_tot - n_pos) / float(n_pos)


# Per-checkpoint / per-seed fine-tuning

def train_one_model(
    run_idx: int,
    model,                    # already initialised (pretrained or random)
    train_chains: list,
    val_chains: list,
    train_labels: dict,
    val_labels: dict,
    residue_maps: dict,
    pos_weight: float,
    out_path: Path,
    args,
    device,
) -> float:
    """Train one model to convergence (early stopping on val IoU); save the best checkpoint."""
    label = "seed" if args.from_scratch else "checkpoint"
    print(f"\n  {label.capitalize()} {run_idx}/5"
          f"  kernel={args.kernel}  features={args.features}"
          f"  epochs={args.epochs}  lr={args.lr}  patience={args.patience}")

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing: lr decays from args.lr → eta_min over T_max epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    pw      = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    train_dataset = H5ChainDataset(train_chains, TRAIN_H5, train_labels)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    best_iou     = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history_rows = []

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps    = 0

        for emb_batch, lbl_batch in train_loader:
            emb = emb_batch.squeeze(0)              # (L, 1024)
            lbl = lbl_batch.squeeze(0)              # (L, 3)

            x  = emb.T.unsqueeze(0).to(device)     # (1, 1024, L)
            y2 = lbl[:, 2].to(device)               # (L,)

            optimizer.zero_grad()
            out = model(x)          # (3, L) — CNN2Layers squeezes the batch dim
            if out.dim() == 1:
                out = out.unsqueeze(1)
            loss = loss_fn(out[2], y2)   # channel 2 = small-molecule
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps    += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = total_loss / max(n_steps, 1)

        val_preds = run_val_inference(model, val_chains, device)
        val_iou   = compute_iou_fast(
            val_preds, residue_maps, val_labels,
            threshold=args.val_threshold,
        )

        is_best = val_iou > best_iou
        tag     = " ← best" if is_best else ""
        print(f"  epoch {epoch:3d}/{args.epochs}"
              f"  loss={avg_loss:.4f}  val_IoU={val_iou:.4f}"
              f"  lr={current_lr:.2e}{tag}")

        history_rows.append({
            label:        run_idx,
            "epoch":      epoch,
            "train_loss": avg_loss,
            "val_iou":    val_iou,
            "lr":         current_lr,
        })

        if is_best:
            best_iou     = val_iou
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), out_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stop: no improvement for {args.patience} epochs.")
                break

    print(f"\n  Best val IoU: {best_iou:.4f}  (epoch {best_epoch})")
    print(f"  Saved → {out_path}")

    reports_dir = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem   # e.g. "ft_k9_f256_checkpoint1"
    hist_path = reports_dir / f"history_{stem}.csv"
    pd.DataFrame(history_rows).to_csv(hist_path, index=False)
    print(f"  History → {hist_path}")

    return best_iou


def main():
    """Parse args, build the train/val split, and train 1 or 5 model(s)."""
    parser = argparse.ArgumentParser(
        description="Fine-tune (or train from scratch) the bindEmbed21DL CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=int, default=0,
                        help="Which pretrained checkpoint / seed to use (1–5, or 0 = all).")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=7,
                        help="Early-stopping patience in epochs.")
    parser.add_argument("--val-threshold", type=float, default=0.5,
                        help="Threshold for monitoring val IoU during training. "
                             "Does not affect final threshold (swept by validate_baseline.py).")
    # Tier-1: architecture args
    parser.add_argument("--kernel",   type=int,   default=PRETRAINED_KERNEL,
                        help="CNN kernel size. Changing from 5 forces --from-scratch.")
    parser.add_argument("--features", type=int,   default=PRETRAINED_FEATURES,
                        help="CNN feature channels. Changing from 128 forces --from-scratch.")
    parser.add_argument("--dropout",  type=float, default=0.5,
                        help="Dropout for from-scratch models (ignored when loading pretrained).")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Always train from random init, ignoring pretrained weights.")
    parser.add_argument("--out-prefix", type=str, default="",
                        help="Full output path prefix for checkpoint files "
                             "(e.g. trained/ft_k9_f256_checkpoint). "
                             "Auto-generated if empty.")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 epoch, run 1 only — smoke test.")
    args = parser.parse_args()

    arch_changed = (args.kernel != PRETRAINED_KERNEL or args.features != PRETRAINED_FEATURES)
    args.from_scratch = args.from_scratch or arch_changed

    if args.dry_run:
        args.checkpoint = 1
        args.epochs     = 1
        print("dry run: 1 epoch, run 1 only")

    if not args.out_prefix:
        if arch_changed:
            args.out_prefix = str(
                config.TRAINED_DIR
                / f"ft_k{args.kernel}_f{args.features}_checkpoint"
            )
        else:
            args.out_prefix = str(config.TRAINED_DIR / "finetune_checkpoint")

    print(f"Mode        : {'from scratch' if args.from_scratch else 'fine-tune from pretrained'}")
    print(f"Architecture: kernel={args.kernel}  features={args.features}"
          + (f"  dropout={args.dropout}" if args.from_scratch else "  dropout=0.7 (pretrained)"))
    print(f"Output prefix: {args.out_prefix}")

    for path in [TRAIN_H5, TRAIN_MAPS, TRAIN_CSV]:
        if not path.exists():
            sys.exit(f"ERROR: required file not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  {torch.cuda.get_device_name(0)} | {props.total_memory/1e9:.1f} GB VRAM")

    print("\ntrain/val split")
    train_ids, val_ids = load_or_make_split(TRAIN_CSV, split_out=SPLIT_OUT)

    print("\nloading residue maps and labels")
    residue_maps = load_residue_maps(TRAIN_MAPS)
    all_labels   = load_all_labels(TRAIN_CSV, residue_maps)

    with h5py.File(TRAIN_H5, "r") as f:
        h5_keys = set(f.keys())

    train_pid_set = set(str(p) for p in train_ids)
    val_pid_set   = set(str(p) for p in val_ids)

    train_chains = sorted(k for k in h5_keys
                          if k.rsplit("_", 1)[0] in train_pid_set and k in all_labels)
    val_chains   = sorted(k for k in h5_keys
                          if k.rsplit("_", 1)[0] in val_pid_set   and k in all_labels)

    train_labels = {k: all_labels[k] for k in train_chains}
    val_labels   = {k: all_labels[k] for k in val_chains}

    n_pos_tr = sum(int(v.sum()) for v in train_labels.values())
    n_tot_tr = sum(len(v)       for v in train_labels.values())
    print(f"  Train: {len(train_chains):,} chains  "
          f"({n_pos_tr:,}/{n_tot_tr:,} binding = {100*n_pos_tr/n_tot_tr:.1f}%)")
    print(f"  Val:   {len(val_chains):,} chains")

    pos_weight = compute_pos_weight(train_chains, train_labels)
    print(f"  pos_weight = {pos_weight:.2f}")

    run_indices = [args.checkpoint] if args.checkpoint > 0 else list(range(1, 6))
    results: dict[int, float] = {}

    for run_idx in run_indices:
        if args.from_scratch:
            from architectures import CNN2Layers
            set_seed(run_idx * 17)
            padding = (args.kernel - 1) // 2
            model = CNN2Layers(
                in_channels=1024,
                feature_channels=args.features,
                kernel_size=args.kernel,
                stride=1,
                padding=padding,
                dropout=args.dropout,
            )
        else:
            model_path = f"{PRETRAINED_PREFIX}{run_idx}.pt"
            model = torch.load(model_path, map_location=device, weights_only=False)

        out_path = Path(f"{args.out_prefix}{run_idx}.pt")

        iou = train_one_model(
            run_idx=run_idx,
            model=model,
            train_chains=train_chains,
            val_chains=val_chains,
            train_labels=train_labels,
            val_labels=val_labels,
            residue_maps=residue_maps,
            pos_weight=pos_weight,
            out_path=out_path,
            args=args,
            device=device,
        )
        results[run_idx] = iou

    for run_idx, iou in results.items():
        print(f"    run {run_idx}: best val IoU @ t={args.val_threshold:.3f} = {iou:.4f}")


if __name__ == "__main__":
    main()
