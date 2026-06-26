"""Predict the antigen epitope from a returned AnthroFold complex structure.

Given a predicted complex CIF and the **chain ID** of the antigen,
``predict_epitope`` reports every antigen residue that has at least one heavy
atom within ``contact_threshold_a`` Å of any heavy atom on a different chain.

**Antigen chain identification.** The caller passes ``antigen_chain_id``
explicitly. For predictions returned by SageMaker, ``find_antigen_chain_id``
auto-detects the chain by exact polymer-sequence match against the antigen
sequence from the input CSV row. AnthroFold preserves submitted polymer
sequences verbatim in the returned mmCIF, so this match is deterministic and
does not depend on any chain-ordering convention. ``ValueError`` is raised if
the chain ID isn't present or no chain matches the antigen sequence; a
homo-multimer antigen (several identical matching chains) resolves to the first.

**Epitope definition.** A residue is on the epitope if **any** of its heavy
atoms is within ``DEFAULT_EPITOPE_CONTACT_THRESHOLD_A`` Å of **any** heavy atom
on a non-antigen **polymer** residue. Waters, ligands, ions, and glycans on
non-antigen chains are skipped — only protein/nucleic-acid polymer atoms count
as binder contacts. The 4.5 Å heavy-atom default is a common choice for
antibody-antigen contact definition.

**Epitope correctness scoring.** ``epitope_correctness`` compares a predicted
epitope (set of antigen residue IDs) against a ground-truth epitope and
returns continuous precision / recall / F1 / Jaccard. The recall component is
the direct epitope analogue of DockQ's Fnat (fraction of native contact
residues recovered); F1 is the headline summary number.
"""

from typing import Dict, Iterable, List, Tuple

import gemmi

# Default heavy-atom contact threshold (Å). Common choice for antibody-antigen contacts.
DEFAULT_EPITOPE_CONTACT_THRESHOLD_A: float = 4.5


def _chain_polymer_one_letter(chain) -> str:
    """One-letter polymer sequence for a gemmi chain. Modified residues that
    gemmi recognises (MSE, SEC, PYL) map to their parent code; unknowns → X."""
    out = []
    for r in chain.get_polymer():
        info = gemmi.find_tabulated_residue(r.name)
        out.append(info.one_letter_code.upper() if info and info.one_letter_code else "X")
    return "".join(out)


def find_antigen_chain_ids(cif_path: str, antigen_seq: str) -> List[str]:
    """Return **every** chain ID in ``cif_path`` whose polymer sequence matches
    the antigen.

    ``antigen_seq`` is the CSV antigen field and may be a single sequence or a
    ``/``-separated list of sequences (a multi-chain antigen). This covers a
    homo-multimer (one sequence appearing on several chains) and a
    hetero-multimer (several different antigen chains) uniformly. AnthroFold
    predictions preserve the submitted sequence verbatim, so the match is exact
    and does not depend on chain ordering. Raises ``ValueError`` if no chain
    matches any antigen sequence.

    Note: this is for **predicted** CIFs where the sequence round-trips exactly.
    Ground-truth CIFs may have unresolved residues and won't exact-match.
    """
    targets = {s.upper().replace(" ", "") for s in antigen_seq.split("/") if s.strip()}
    if not targets:
        raise ValueError(f"antigen_seq is empty for {cif_path}")

    structure = gemmi.read_structure(cif_path)
    structure.setup_entities()

    matches = []
    chain_summary = []
    for chain in structure[0]:
        seq = _chain_polymer_one_letter(chain)
        chain_summary.append(f"{chain.name}: {len(seq)} aa")
        if seq in targets:
            matches.append(chain.name)

    if not matches:
        raise ValueError(
            f"No chain in {cif_path} matches the antigen sequence(s). "
            f"Chains present: {chain_summary}"
        )
    return matches


def find_antigen_chain_id(cif_path: str, antigen_seq: str) -> str:
    """Return a single antigen chain ID (the first match). See
    ``find_antigen_chain_ids`` for the multi-chain version used by DockQ scoring.
    """
    return find_antigen_chain_ids(cif_path, antigen_seq)[0]


def predict_epitope(
    cif_path: str,
    antigen_chain_ids,
    contact_threshold_a: float = DEFAULT_EPITOPE_CONTACT_THRESHOLD_A,
) -> List[Tuple[str, int, str]]:
    """Return the epitope residues for a predicted complex.

    ``antigen_chain_ids`` is an iterable of the antigen chain IDs (one, or
    several for a multi-chain antigen; e.g. from ``find_antigen_chain_ids``). An
    antigen residue is in the epitope if any of its heavy atoms is within
    ``contact_threshold_a`` Å of any heavy atom on a chain that is NOT an antigen
    chain (a binder chain). Excluding *all* antigen chains is what prevents
    antigen-antigen contacts from being miscounted as epitope.

    Returns a list of ``(chain_id, residue_seq_id, residue_name)`` tuples in
    primary-sequence order; ``residue_seq_id`` is gemmi's ``residue.seqid.num``
    (the auth-seq-id, consistent between predicted and native structures).

    Raises ``ValueError`` if any antigen chain ID isn't present in the structure.
    """
    antigen_ids = set(antigen_chain_ids)
    if not antigen_ids:
        raise ValueError("antigen_chain_ids is empty")

    structure = gemmi.read_structure(cif_path)
    structure.setup_entities()
    model = structure[0]

    present = {c.name for c in model}
    missing = antigen_ids - present
    if missing:
        raise ValueError(
            f"Antigen chain(s) {sorted(missing)} not found in {cif_path}. "
            f"Chains in the structure: {sorted(present)}"
        )

    ns = gemmi.NeighborSearch(model, structure.cell, contact_threshold_a + 1.0)
    ns.populate(include_h=False)

    epitope: List[Tuple[str, int, str]] = []
    for chain in model:
        if chain.name not in antigen_ids:
            continue
        for residue in chain.get_polymer():
            in_contact = False
            for atom in residue:
                if atom.element.name == "H":
                    continue
                for mark in ns.find_atoms(atom.pos, "\0", radius=contact_threshold_a):
                    cra = mark.to_cra(model)
                    if cra.chain.name in antigen_ids:
                        continue  # self or another antigen chain — not a binder contact
                    if cra.residue.entity_type != gemmi.EntityType.Polymer:
                        continue  # skip waters / ligands / ions / glycans
                    if atom.pos.dist(cra.atom.pos) <= contact_threshold_a:
                        in_contact = True
                        break
                if in_contact:
                    break
            if in_contact:
                epitope.append((chain.name, residue.seqid.num, residue.name.upper().strip()))
    return epitope


def epitope_correctness(
    predicted_residues: Iterable,
    ground_truth_residues: Iterable,
) -> Dict[str, float]:
    """Continuous epitope-prediction correctness scoring.

    Both arguments are iterables of antigen residue identifiers — typically the
    residue sequence numbers (``int``) returned by ``predict_epitope`` (first
    element of each tuple), but any hashable identifier (e.g.
    ``(seq_id, residue_name)``) works as long as the same scheme is used on
    both sides.

    Returns a dict with:

    - ``precision`` — |pred ∩ gt| / |pred|. Fraction of predicted epitope
      residues that are actually on the experimental epitope.
    - ``recall`` — |pred ∩ gt| / |gt|. Fraction of experimental epitope
      residues recovered — the direct DockQ-Fnat analogue at residue level.
    - ``f1`` — harmonic mean of precision and recall.
    - ``jaccard`` — |pred ∩ gt| / |pred ∪ gt|.
    - ``true_positive`` / ``false_positive`` / ``false_negative`` — raw counts.
    - ``predicted_size`` / ``ground_truth_size`` — raw set sizes.
    - ``degenerate_empty`` — True iff both inputs are empty (in which case the
      four ratio metrics are set to 1.0 by convention but the result is not
      informative; downstream aggregators should filter these out).
    """
    pred = set(predicted_residues)
    gt = set(ground_truth_residues)

    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    degenerate_empty = not pred and not gt
    if degenerate_empty:
        precision = recall = f1 = jaccard = 1.0
    else:
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gt) if gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        union = len(pred | gt)
        jaccard = tp / union if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_size": len(pred),
        "ground_truth_size": len(gt),
        "degenerate_empty": degenerate_empty,
    }
