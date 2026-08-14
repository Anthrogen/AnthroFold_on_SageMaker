"""Convert structure-determination CSV input into the AnthroFold job list
consumed by SageMaker async inference.

The CSV columns are:

    antigen_seq      Amino-acid sequence of the antigen.
                     Multiple antigen chains can be passed by separating them with '|'.
    binder_seq       Binder sequence. Multiple chains separated with '|'.
    binder_mode      1 for antibody-like binders, 0 for general protein binders.
    antibody_form    'VH/VL', 'SS', or 'None'.
    antigen_temp     Antigen template field from the input schema, or 'None'.
    exp_structure    Ground-truth structure filename, used as the round-trip
                     identifier for downstream DockQ scoring.
    antigen_msa      Required for folding. Pipe-separated MSA directories, one
                     per antigen chain, in the same order as antigen_seq.
    binder_msa       Required for folding. Same, for binder_seq.

``csv_to_jobs`` will parse a CSV without the two MSA columns, but ``validate_msas``
refuses to submit one: AnthroFold folds only from supplied MSAs.

Each MSA directory is one produced by the AnthroFold MSA pipeline
(``src/msa_client.py``) and holds
``pairing.a3m`` and ``non_pairing.a3m``.
"""

import csv
import re
from pathlib import Path

PAIRED_A3M = "pairing.a3m"
UNPAIRED_A3M = "non_pairing.a3m"

# The two sequence columns, and the MSA column that mirrors each.
SEQ_COLUMNS = ("antigen_seq", "binder_seq")


def split_chains(value):
    """Split a chain-separated sequence field into individual chain sequences.

    '|' is the documented separator: the antibody_form column carries literal
    values like 'VH/VL', so a slash separator collided with real field content.
    '/' is still accepted here for CSVs written before that change — amino-acid
    sequences never contain either character, so this is unambiguous.
    """
    if not value:
        return []
    return [chain.strip() for chain in re.split(r"[|/]", value) if chain.strip()]


def _split_msa_dirs(value):
    """Split a chain-separated MSA-directory field.

    Pipe only, unlike split_chains: these are filesystem paths and '/' is a
    path separator.
    """
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _job_name(row, row_index):
    """Pick a deterministic job name. Prefer exp_structure (stripped of .cif),
    fall back to a zero-padded row index."""
    exp = (row.get("exp_structure") or "").strip()
    if exp and exp.lower() != "none":
        return re.sub(r"\.cif$", "", exp, flags=re.IGNORECASE)
    return f"row_{row_index:04d}"


def _protein_chain(sequence, msa_dir=None):
    chain = {
        "sequence": sequence,
        "count": 1,
        "modifications": [],
    }
    if msa_dir:
        chain["unpairedMsaPath"] = str(Path(msa_dir) / UNPAIRED_A3M)
        chain["pairedMsaPath"] = str(Path(msa_dir) / PAIRED_A3M)
    return {"proteinChain": chain}


def _chain_msa_dirs(row, seq_column, chains, name, path):
    """MSA directory per chain for one side, or Nones when the column is absent.

    The link between a chain and its MSA is positional, so a length mismatch is
    not recoverable — it would silently fold one chain against another chain's
    MSA.
    """
    msa_column = seq_column.replace("_seq", "_msa")
    dirs = _split_msa_dirs(row.get(msa_column, ""))
    if not dirs:
        return [None] * len(chains)
    if len(dirs) != len(chains):
        raise ValueError(
            f"Row {name!r} in {path}: {msa_column} lists {len(dirs)} MSA director"
            f"{'y' if len(dirs) == 1 else 'ies'} but {seq_column} has {len(chains)} "
            f"chain{'' if len(chains) == 1 else 's'}. They are matched by position, "
            "so they must line up one-to-one."
        )
    return dirs


def csv_to_jobs(path):
    """Read the structure-determination CSV and return AnthroFold prediction jobs.

    Each job dict matches the SageMaker endpoint payload shape:
    ``{"name": ..., "sequences": [{"proteinChain": {...}}, ...], "covalent_bonds": []}``.
    Antigen chains appear first in the sequences list, followed by binder chains.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if "antigen_seq" not in fieldnames or "binder_seq" not in fieldnames:
            raise ValueError(
                f"CSV at {path} must include 'antigen_seq' and 'binder_seq' columns; "
                f"found columns: {fieldnames}"
            )
        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV at {path} has a header but no data rows.")

    jobs = []
    seen_names = {}
    for i, row in enumerate(rows):
        name = _job_name(row, i)
        antigen_chains = split_chains(row.get("antigen_seq", ""))
        binder_chains = split_chains(row.get("binder_seq", ""))
        if not antigen_chains or not binder_chains:
            empty = "antigen_seq" if not antigen_chains else "binder_seq"
            raise ValueError(
                f"Row {i} (name={name!r}) in {path} has an empty {empty} column."
            )
        # Job names key every downstream output (CIFs, confidence, DockQ), so a
        # duplicate silently overwrites an earlier row's results.
        if name in seen_names:
            raise ValueError(
                f"Row {i} in {path} repeats the job name {name!r} from row "
                f"{seen_names[name]}. Names come from exp_structure and must be "
                "unique — downstream outputs are keyed by them."
            )
        seen_names[name] = i

        antigen_msas = _chain_msa_dirs(row, "antigen_seq", antigen_chains, name, path)
        binder_msas = _chain_msa_dirs(row, "binder_seq", binder_chains, name, path)
        chains = list(zip(antigen_chains + binder_chains, antigen_msas + binder_msas))
        jobs.append(
            {
                "name": name,
                "sequences": [_protein_chain(seq, msa_dir) for seq, msa_dir in chains],
                "covalent_bonds": [],
            }
        )
    return jobs
