"""bindembed_addon — supporting pipeline for ligand-binding residue prediction.

Submodules:
    config   — path configuration, overridable by environment variables
    structure — PDB parsing (CA-atom extraction, residue maps)
    labels   — aligning train.csv binding tokens to residue maps
    metrics  — pooled-IoU scoring (wraps the course-provided metric.py)
    splits   — deterministic protein-level train/val split
    training — PyTorch Dataset for streaming embeddings from HDF5
"""
