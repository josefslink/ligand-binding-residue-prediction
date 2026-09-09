# Validation metrics

Evaluation setting: single held-out split, 1,965 proteins (protein-level 88/12
split at seed 42, `src/bindembed_addon/splits/by_protein.py`) — not cross-validation.
Val IoU is pooled Jaccard over the val split only, computed by
`src/bindembed_addon/metrics/iou.py` (same pooled-Jaccard logic as `metric.py`).

Threshold tuning was done on this same validation split used for early stopping,
so the ensemble's best-val-IoU numbers below are mildly optimistic relative to
the public leaderboard (`leaderboard.md`) — treat the leaderboard numbers as the
honest ones.

## Per-checkpoint / per-seed best val IoU

Individual models, monitored at threshold 0.5 during training (`--val-threshold`,
default). History files: `finetune_history_ckpt{1-5}.csv`,
`history_ft_k9_f256_checkpoint{1-5}.csv`.

| Run | Fine-tuned k5/f128 (best epoch) | From-scratch k9/f256 (best epoch) |
|---|---|---|
| 1 | 0.3573 (9)  | 0.4815 (14) |
| 2 | 0.3533 (7)  | 0.4777 (10) |
| 3 | 0.3577 (5)  | 0.4655 (5)  |
| 4 | 0.3381 (8)  | 0.4834 (20) |
| 5 | 0.3196 (1)  | 0.4601 (3)  |
| **range** | **0.3196–0.3577** | **0.4601–0.4834** |

In both cases the 5-checkpoint ensemble beats every individual member — 0.4596
(fine-tuned) and 0.4936 (from-scratch) below, both above any single member's peak.

## Threshold sweep (`threshold_sweep.csv`)

`scripts/validate_baseline.py` overwrites `reports/threshold_sweep.csv` on every
run, so only the **last** sweep run survives on disk: the from-scratch k9/f256
ensemble's, 200 values on `linspace(0.01, 0.95)`.

| Threshold (nearest grid point) | Val IoU | Note |
|---|---|---|
| 0.5013 | 0.4820 | nearest grid point to t=0.5 (no exact 0.5 in the 200-point grid); report states 0.4818 |
| **0.6430** | **0.4936** | best point on this sweep — the from-scratch model's reported result |

The pretrained-baseline (val IoU 0.161 at t=0.5, 0.238 at swept t=0.218) and
fine-tuned k5/f128 (val IoU 0.4596 at swept t=0.648) sweeps were run earlier in
the project, before the from-scratch model's sweep overwrote the same output
file; their val-IoU numbers are recorded here from `report/REPORT.md` prose,
not from a surviving sweep CSV — the underlying threshold/IoU curves for those
two runs were not preserved. Their leaderboard scores (`leaderboard.md`) are
independently confirmed by the challenge server regardless.

## Full progression (val IoU column mixes the two provenances above)

| Model | Threshold | Val IoU | Public test Jaccard |
|---|---|---|---|
| Pretrained bindEmbed21 ensemble, t=0.5 | 0.5 | 0.161 *(from report)* | 0.162 |
| Same, threshold swept | 0.218 | 0.238 *(from report)* | 0.228 |
| Fine-tuned k5/f128 ensemble, threshold swept | 0.648 | 0.4596 *(from report)* | 0.393 |
| From-scratch k9/f256 ensemble, threshold swept | 0.643 | **0.4936** *(this sweep CSV)* | **0.423** |
