"""Convert structure-determination CSV input into the AnthroFold job list
consumed by SageMaker async inference.

The CSV columns are:

    antigen_seq      Amino-acid sequence of the antigen.
                     Multiple antigen chains can be passed by separating them with '/'.
    binder_seq       Binder sequence. Multiple chains separated with '/'.
    binder_mode      1 for antibody-like binders, 0 for general protein binders.
    antibody_form    'VH/VL', 'SS', or 'None'.
    antigen_temp     Antigen template field from the input schema, or 'None'.
    exp_structure    Ground-truth structure filename, used as the round-trip
                     identifier for downstream DockQ scoring.
"""

import csv
import re
from pathlib import Path


def _split_chains(value):
    """Split a slash-separated sequence field into individual chain sequences."""
    if not value:
        return []
    return [chain.strip() for chain in value.split("/") if chain.strip()]


def _job_name(row, row_index):
    """Pick a deterministic job name. Prefer exp_structure (stripped of .cif),
    fall back to a zero-padded row index."""
    exp = (row.get("exp_structure") or "").strip()
    if exp and exp.lower() != "none":
        return re.sub(r"\.cif$", "", exp, flags=re.IGNORECASE)
    return f"row_{row_index:04d}"


def _protein_chain(sequence):
    return {
        "proteinChain": {
            "sequence": sequence,
            "count": 1,
            "modifications": [],
        }
    }


def csv_to_jobs(path):
    """Read the structure-determination CSV and return AnthroFold prediction jobs.

    Each job dict matches the SageMaker endpoint payload shape:
    ``{"name": ..., "sequences": [{"proteinChain": {...}}, ...], "covalent_bonds": []}``.
    Antigen chains appear first in the sequences list, followed by binder chains.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "antigen_seq" not in reader.fieldnames or "binder_seq" not in reader.fieldnames:
            raise ValueError(
                f"CSV at {path} must include 'antigen_seq' and 'binder_seq' columns; "
                f"found columns: {reader.fieldnames}"
            )
        rows = list(reader)

    jobs = []
    for i, row in enumerate(rows):
        antigen_chains = _split_chains(row.get("antigen_seq", ""))
        binder_chains = _split_chains(row.get("binder_seq", ""))
        if not antigen_chains or not binder_chains:
            raise ValueError(
                f"Row {i} in {path} is missing antigen_seq or binder_seq: {row!r}"
            )
        jobs.append(
            {
                "name": _job_name(row, i),
                "sequences": [_protein_chain(seq) for seq in antigen_chains + binder_chains],
                "covalent_bonds": [],
            }
        )
    return jobs
