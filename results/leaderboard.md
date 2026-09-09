# Challenge-server leaderboard

Pooled (micro) Jaccard / IoU on the public test set (3,064 structures), scored by the
course challenge server via `metric.py`. This is the authoritative number for each
row — see `validation_metrics.md` for the corresponding (mildly optimistic) local
validation IoU.

| Model | Threshold | Public test Jaccard |
|---|---|---|
| Pretrained bindEmbed21 ensemble (unmodified) | 0.5 | 0.162 |
| Same, threshold swept on validation | 0.218 | 0.228 |
| Fine-tuned k5/f128 ensemble (5 checkpoints, fine-tuned) | 0.648 | 0.393 |
| From-scratch k9/f256 ensemble (5 seeds, trained from scratch) | 0.643 | **0.423** |

Progression: 0.162 → 0.423 over the unchanged pretrained baseline (2.6×). Threshold
calibration alone (no training) accounts for 0.162 → 0.228 of that.

## Not published

- **Checkpoints** (`finetune_checkpoint{1-5}.pt`, `ft_k9_f256_checkpoint{1-5}.pt`,
  58 MB total): excluded from the repo tree and from GitHub Releases. Decision:
  the leaderboard above is third-party evidence of the result and is stronger
  than a self-published binary; a Release would add a maintenance surface for
  no reviewer benefit.
- **Submission CSVs** (`submissions/*.csv`, 3.3 MB): these are model outputs,
  not challenge data, but they enumerate test-set protein IDs and residue
  tokens and add nothing a reviewer will read. Not published.
