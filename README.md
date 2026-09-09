# Ligand-Binding Residue Prediction

Small-molecule binding-residue prediction from PDB structures: ProtT5 embeddings + fine-tuned bindEmbed21 CNN.

## 1. Problem

Which residues of a protein line a small-molecule binding pocket? Binding-site
identification is the entry point to structure-based drug design — you cannot
dock or design a ligand without first knowing where it binds. Input is a PDB
structure; output is the set of pocket-lining residues, scored by pooled
(micro) Jaccard index / IoU over every residue of every protein.

## 2. Approach

16,379 training and 3,064 test structures. CA-atom records give each chain's
sequence and an ordered residue map, which is what lets an embedding position
be mapped back to a `chain_resnum[_icode]` submission token. Sequences are
embedded with ProtT5-XL-U50 in fp16 (chunked above 1,000 residues) and cached
as float32 `(L, 1024)` arrays in HDF5. A 2-layer CNN over those embeddings
predicts per-residue binding probability, read off the small-molecule channel;
final predictions are a 5-model ensemble average thresholded at a value swept
on validation.

## 3. Results

Evaluation setting for the public-test column: 3,064 test structures via the
course challenge server, pooled Jaccard. Val IoU: single held-out split, 1,965
proteins, protein-level 88/12 split at seed 42 — not cross-validation, and
tuned on the same split used for early stopping (see Limitations).

| Model | Val IoU (best t) | Public test Jaccard |
|---|---|---|
| Pretrained bindEmbed21 ensemble, t=0.5 (**baseline**) | 0.161 | **0.162** |
| Same, threshold swept to t=0.218 | 0.238 | 0.228 |
| Fine-tuned k5/f128 ensemble, t=0.648 | 0.4596 | 0.393 |
| From-scratch k9/f256 ensemble, t=0.643 | 0.4936 | **0.423** |

Lead with the progression, not the endpoint: **0.162 → 0.423 over the
unchanged pretrained baseline, a 2.6× improvement**, of which threshold
calibration alone accounts for 0.162 → 0.228.

Per-seed detail in [`results/`](results/): fine-tuned checkpoints peak at
0.3196–0.3577 individually, the from-scratch seeds at 0.4601–0.4834 — the
ensemble beats every member in both cases.

## 4. Reproduction

```bash
pip install -r requirements.txt          # Python 3.12
bash scripts/setup_bindpredict.sh        # fetches Rostlab/bindPredict @ c9b12e7
# obtain the challenge data — see data/README.md (not redistributable)
```

Then:

```bash
# embed (one-off; produces ~30.5 GB train / ~5.5 GB test HDF5, GPU strongly recommended)
python scripts/embed_sequences.py --split train
python scripts/embed_sequences.py --split test

# fine-tune from the pretrained checkpoints (reproduces the k5/f128 result)
python scripts/finetune_cnn.py --epochs 20 --patience 5

# train from scratch with a wider kernel (reproduces the final, submitted result)
python scripts/finetune_cnn.py --kernel 9 --features 256

# predict on the test set with the final model
python scripts/predict.py --skip-embed --threshold 0.643 \
    --model-prefix trained/ft_k9_f256_checkpoint
```

Seeds: the train/val split uses `random.Random(42)`; from-scratch training uses
`set_seed(run_idx * 17)` per run (17/34/51/68/85 for runs 1–5), covering
`random`, `numpy`, and `torch` (CPU + CUDA) — see
`src/bindembed_addon/training/seeding.py`. `torch.backends.cudnn` is left at
its default (non-deterministic), so bit-exact reproduction on GPU is not
guaranteed even with the seed fixed.

Note on `--epochs 20 --patience 5`: this is what was actually run to produce
the fine-tuned (k5/f128) result — confirmed from `results/finetune_history_ckpt*.csv`,
which stop 5 epochs past each run's best epoch. The script's own defaults are
`--epochs 30 --patience 7` (the values used for the from-scratch k9/f256 run) —
they are left unchanged rather than "corrected" to match the fine-tune run,
since that would edit the record of what was actually submitted. Pass
`--epochs 20 --patience 5` explicitly to reproduce the fine-tuned row above.

**Hardware and wall-clock, stated honestly:** GPU model **not recorded**.
Embedding wall-clock **not recorded**. Recovered from file timestamps in
`results/`: the five from-scratch (k9/f256) seeds took roughly 50 min – 1 h 15 min
each (2026-07-04 20:24 → 23:36 for all five); the five fine-tune runs took
roughly 15–35 min each (2026-07-03 15:46 → 17:36).

## 5. Repo structure

```
README.md
LICENSE                   MIT
THIRD_PARTY.md            non-original code accounted for
requirements.txt          pinned dependencies
metric.py                 course-provided scorer, unmodified (header comment added)
src/bindembed_addon/      own code — config, structure parsing, labels, metrics, splits, training
scripts/                  entry points — setup, embedding, training, validation, prediction, smoke test
third_party/              gitignored; populated by scripts/setup_bindpredict.sh (Rostlab/bindPredict)
data/README.md            provenance + acquisition; no data committed
results/                  leaderboard, validation metrics, training histories, threshold sweep
```

## 6. Limitations and next steps

Single held-out split, not k-fold — val IoU has no error bar. Threshold tuned
on the same validation split used for early stopping, so 0.4936 is mildly
optimistic; the 0.423 leaderboard number is the honest one, and the ~0.07 gap
between them is consistent with that. The kernel-9 gain is one experiment, not
a sweep — no hyperparameter search was run. Next: proper cross-validation,
calibrated per-protein thresholds instead of one global cut, and testing
whether a larger receptive field or a structure-aware (not sequence-only)
model closes the remaining gap.

## 7. Provenance

University challenge submission, Structural Bioinformatics, summer semester
2026. Solo work. All code under `src/` and `scripts/` is mine. `metric.py` is
the course-provided scorer, unmodified. The base CNN architecture and
pretrained checkpoints are `Rostlab/bindPredict` (bindEmbed21DL, Littmann et
al. 2021), MIT, fetched at setup rather than vendored. Embeddings use
ProtT5-XL-U50 (Elnaggar et al., ProtTrans) via HuggingFace. The course permits
pretrained models; this approach depends on two of them. **No grade stated.**
