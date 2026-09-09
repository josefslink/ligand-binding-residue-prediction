"""
Walks CA atoms in PDB files to produce a sequence dict and a residue-map dict,
both keyed by "{protein_id}_{chain}". Array index i in any embedding or prediction
corresponds to residue_maps[chain_key][i]. Token format: "{chain}_{resnum}" or
"{chain}_{resnum}_{icode}" when an insertion code is present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def resid_token(chain: str, resnum: int, icode: str) -> str:
    """Build the canonical chain_resid token used in submissions and train.csv."""
    return f"{chain}_{resnum}_{icode}" if icode else f"{chain}_{resnum}"


def _iter_ca_atoms(pdb_path: Path) -> Iterator[dict]:
    """Yield one dict per CA atom in PDB order."""
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            yield {
                "aa":     AA3_TO_1.get(line[17:20].strip(), "X"),
                "chain":  line[21],
                "resnum": int(line[22:26].strip()),
                "icode":  line[26].strip(),
            }


def extract_protein(pdb_path: Path, protein_id: str) -> tuple[dict, dict]:
    """
    Extract sequences and residue maps for all chains in one PDB file.

    Returns
    -------
    fasta_dict   : { "protid_chain": seq_str }
    residue_maps : { "protid_chain": [{"chain","resnum","icode"}, …] }
    """
    chains: dict[str, list[dict]] = {}
    for atom in _iter_ca_atoms(pdb_path):
        chains.setdefault(atom["chain"], []).append(atom)

    fasta_dict: dict[str, str] = {}
    residue_maps: dict[str, list[dict]] = {}
    for chain_id, residues in chains.items():
        key = f"{protein_id}_{chain_id}"
        fasta_dict[key] = "".join(r["aa"] for r in residues)
        residue_maps[key] = [
            {"chain": r["chain"], "resnum": r["resnum"], "icode": r["icode"]}
            for r in residues
        ]
    return fasta_dict, residue_maps


def extract_from_pdb_dir(
    pdb_dir: Path,
    fasta_out: Path | None = None,
    maps_out:  Path | None = None,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """
    Extract sequences + residue maps for every PDB in pdb_dir.

    Parameters
    ----------
    pdb_dir   : directory containing *_protein.pdb files
    fasta_out : if given, write FASTA to this path
    maps_out  : if given, write residue maps JSON to this path
    verbose   : print progress every 500 proteins

    Returns
    -------
    fasta_dict   : { "protid_chain": seq_str }
    residue_maps : { "protid_chain": [{"chain","resnum","icode"}, …] }
    """
    fasta_dict:   dict[str, str]        = {}
    residue_maps: dict[str, list[dict]] = {}

    pdbs = sorted(pdb_dir.glob("*.pdb"))
    for idx, pdb_path in enumerate(pdbs):
        if verbose and idx % 500 == 0:
            print(f"  {idx}/{len(pdbs)} proteins processed…")

        protein_id = pdb_path.stem.replace("_protein", "")
        fd, rm = extract_protein(pdb_path, protein_id)
        fasta_dict.update(fd)
        residue_maps.update(rm)

    if verbose:
        print(f"  {len(fasta_dict)} chains across {len(pdbs)} proteins")

    if fasta_out is not None:
        fasta_out.parent.mkdir(parents=True, exist_ok=True)
        with open(fasta_out, "w") as fh:
            for key, seq in fasta_dict.items():
                fh.write(f">{key}\n{seq}\n")
        if verbose:
            print(f"  FASTA → {fasta_out}")

    if maps_out is not None:
        maps_out.parent.mkdir(parents=True, exist_ok=True)
        with open(maps_out, "w") as fh:
            json.dump(residue_maps, fh)
        if verbose:
            print(f"  Residue maps → {maps_out}")

    return fasta_dict, residue_maps


def load_fasta(fasta_path: Path) -> dict[str, str]:
    """Load a FASTA file into {header: sequence} dict."""
    result: dict[str, str] = {}
    key: str | None = None
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                key = line[1:]
            elif key is not None:
                result[key] = line
    return result


def load_residue_maps(maps_path: Path) -> dict[str, list[dict]]:
    """Load residue maps JSON from disk."""
    with open(maps_path) as fh:
        return json.load(fh)
