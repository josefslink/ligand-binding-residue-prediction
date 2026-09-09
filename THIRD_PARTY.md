# Third-party components

Everything under `src/` and `scripts/` (except `metric.py`, noted below) is my
own work. Two external components are used unmodified; a third — a pretrained
language model — is fetched at runtime via HuggingFace and never vendored.

## 1. Rostlab/bindPredict

- **Source:** https://github.com/Rostlab/bindPredict, commit `c9b12e7`.
- **Licence:** MIT (`LICENSE` in that repository).
- **How it gets here:** fetched by `scripts/setup_bindpredict.sh` into
  `third_party/bindPredict/` (gitignored — never committed). Not vendored in
  this repo's git history.
- **What it supplies:** the `CNN2Layers` architecture (`architectures.py`) and
  the pretrained checkpoints `trained_models/checkpoint{1..5}.pt` (bindEmbed21DL,
  Littmann et al. 2021). Both are used unmodified.
- **Known incompatibility (not patched):** `bindPredict/config.py`'s
  `FileManager.load_classifier_torch` calls `torch.load()` without
  `weights_only=False`, which fails under torch ≥2.6 (default changed from
  `False` to `True`). This repo's own pipeline (`scripts/predict.py`,
  `scripts/validate_baseline.py`, `scripts/finetune_cnn.py`) never calls that
  function — each loads checkpoints directly with `weights_only=False`
  explicit, and is unaffected. `scripts/smoke_test.py` originally called
  bindPredict's own `BindEmbed21DL.prediction_pipeline()`, which does hit the
  incompatible code path; it has been rewritten to load checkpoints the same
  way the rest of this repo does, so the smoke test passes without touching
  anything under `third_party/`. See that script's module docstring.

## 2. `metric.py`

- **Source:** course-provided scorer for this challenge.
- **Status:** kept at the repo root, unmodified except for a header comment
  identifying it as course-provided. Computes pooled (micro) Jaccard/IoU —
  the exact metric used by the challenge server.
- `src/bindembed_addon/metrics/iou.py` imports `parse_submission` directly
  from this file so local scoring stays byte-for-byte in sync with the grader.

## 3. ProtT5-XL-U50

- **Source:** `Rostlab/prot_t5_xl_half_uniref50-enc` via HuggingFace
  (`transformers`). Elnaggar et al., ProtTrans.
- **How it gets here:** downloaded automatically by `transformers` on first
  use of `scripts/embed_sequences.py` or `scripts/predict.py`; not vendored.
- **What it supplies:** the per-residue protein language-model embeddings
  that are the sole input feature to the CNN.

The course permits use of pretrained models; this approach depends on two of
them (bindEmbed21's checkpoints and ProtT5).
