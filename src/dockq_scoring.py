"""DockQ scoring helper, vendored from FoldBench's evaluation/eval_by_dockqv2.py.

The wrapper logic (chain-type detection, het-residue reformatting, best-mapping
search across chain permutations) is preserved so that per-interface DockQ /
LRMSD / iRMSD / Fnat values match what FoldBench produces on the same input.
The underlying DockQ implementation comes from the public ``DockQ`` PyPI
package (https://github.com/bjornwallner/DockQ), so this repository can use
``pip install DockQ`` without needing the FoldBench tree.

The headline ``best_dockq`` returned by ``dockq()`` here is a *mean* across
the reported interfaces (bounded [0, 1]), which is a convenience this wrapper
adds — FoldBench's original writes one row per interface and doesn't aggregate.

The original FoldBench file also includes a multiprocessing harness that reads
prediction CSVs and writes detail JSONs; that orchestration is intentionally
not vendored. Callers should drive ``dockq()`` directly per (model, native)
pair.
"""

import itertools
from functools import partial

from DockQ.DockQ import (
    count_chain_combinations,
    get_all_chain_maps,
    group_chains,
    load_PDB,
    run_on_all_native_interfaces,
)

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}

NUCLEOTIDES = {
    "DA", "DT", "DG", "DC", "DI", "DU",
    "A", "U", "G", "C", "I",
}


def determine_chain_type(chain_id, residues):
    if not residues:
        return "unk"
    protein_count = sum(1 for res in residues if res in AMINO_ACIDS)
    na_count = sum(1 for res in residues if res in NUCLEOTIDES)
    total = len(residues)
    threshold = 0.8
    if protein_count / total >= threshold:
        return "protein"
    if na_count / total >= threshold:
        return "na"
    return "unk"


def reformat_type(structure):
    for chain_id, chain_value in structure.child_dict.items():
        residues = []
        ctype = "ligand" if chain_value.is_het else None
        if ctype is None:
            for _, res_value in chain_value.child_dict.items():
                residues.append(str(res_value.resname).upper())
            ctype = determine_chain_type(chain_id, residues)
        structure.child_dict[chain_id].type = ctype
        for idx, res in enumerate(structure.child_list):
            if res.id == chain_id:
                structure.child_list[idx].type = ctype
    return structure


def reformat_het(structure):
    """Treat modified-residue chains as polymer rather than ligand.

    DockQv2 marks chains with any non-blank residue hetfield as het, which
    incorrectly classifies polymers containing MSE / phosphorylated residues
    as ligand. We flip them back if any standard polymer residue is present.
    """
    for chain_id, chain_value in structure.child_dict.items():
        is_polymer = False
        for res_id, _ in chain_value.child_dict.items():
            if res_id[0] == " ":
                is_polymer = True
                break
        if is_polymer:
            structure.child_dict[chain_id].is_het = None
            for idx, res in enumerate(structure.child_list):
                if res.id == chain_id:
                    structure.child_list[idx].is_het = None
                    break
    return structure


def dockq(
    model_path,
    native_path,
    model_chains=None,
    native_chains=None,
    small_molecule=False,
    allowed_mismatches=0,
    antigen_chain_ids=None,
):
    """Compute DockQ between a predicted structure and a native structure.

    If ``antigen_chain_ids`` is provided (the model chain IDs of the antigen —
    one, or several for a multi-chain antigen), the returned ``best_result`` is
    filtered to only Ab-Ag interfaces — those where exactly one side is an
    antigen chain. The intra-antibody (VH-VL) interface is excluded: it is
    usually well packed regardless of antigen binding, and including it inflates
    the headline number. The chain-mapping search still runs over all chains; we
    narrow the per-interface report after the fact.

    Returns a dict with::

        {
          "model":         basename of the model file,
          "native":        basename of the native file,
          "best_dockq":    mean per-interface DockQ across (filtered, if antigen_chain_ids given)
                           interfaces of the chosen mapping. Bounded [0, 1].
          "best_result":   per-interface detail dict,
          "best_mapping":  the chain mapping that maximised the (unfiltered) DockQ sum,
        }

    If multiple chain permutations are possible (e.g. homo-multimers), all are
    enumerated and the one with the highest **summed** (unfiltered) DockQ is
    chosen as the best mapping — but the reported ``best_dockq`` is a per-
    interface mean, not the sum.
    """
    model_structure = load_PDB(model_path, small_molecule=small_molecule)
    native_structure = load_PDB(native_path, small_molecule=small_molecule)

    native_structure = reformat_het(native_structure)
    model_structure = reformat_het(model_structure)
    model_structure = reformat_type(model_structure)
    native_structure = reformat_type(native_structure)

    model_chains = [c.id for c in model_structure] if model_chains is None else model_chains
    native_chains = [c.id for c in native_structure] if native_chains is None else native_chains

    chain_clusters, reverse_map = group_chains(
        model_structure,
        native_structure,
        model_chains,
        native_chains,
        allowed_mismatches=allowed_mismatches,
    )
    chain_maps = get_all_chain_maps(
        chain_clusters,
        {},  # no chains are pre-mapped
        reverse_map,
        model_chains,
        native_chains,
    )

    num_chain_combinations = count_chain_combinations(chain_clusters)
    chain_maps, chain_maps_replay = itertools.tee(chain_maps)
    run_chain_map = partial(run_on_all_native_interfaces, model_structure, native_structure)

    best_dockq = -1
    best_result = None
    best_mapping = None

    if num_chain_combinations > 1:
        results_per_mapping = [run_chain_map(cm) for cm in chain_maps]
        for cm, (result, total_dockq) in zip(chain_maps_replay, results_per_mapping):
            if total_dockq > best_dockq:
                best_dockq = total_dockq
                best_result = result
                best_mapping = cm
    else:
        best_mapping = next(chain_maps)
        best_result, best_dockq = run_chain_map(best_mapping)

    if antigen_chain_ids is not None and best_result and best_mapping:
        # best_result is keyed by the *native* interface chains and best_mapping
        # is {native: model}, so resolve the antigen in native space — matching
        # the model chain id against native keys breaks whenever the two
        # structures letter their chains differently. antigen_chain_ids is the
        # model chain id(s) of the antigen; find_antigen_chain_ids returns every
        # chain of a homo- or hetero-multimer antigen.
        native_chains_present = set(best_mapping.keys())
        native_antigens = {native for native, model in best_mapping.items() if model in antigen_chain_ids}

        def _interface_chains(key):
            # A DockQ interface key concatenates its two native chain IDs; split
            # using the known native chains so multi-character chain IDs work too.
            for i in range(1, len(key)):
                a, b = key[:i], key[i:]
                if a in native_chains_present and b in native_chains_present:
                    return {a, b}
            return set(key)  # single-character fallback

        # Keep only Ab-Ag interfaces (exactly one side is an antigen chain) —
        # scoring every antibody-to-antigen contact while dropping the
        # intra-antibody (VH-VL) and antigen-antigen interfaces. If the antigen
        # can't be mapped to a native chain, native_antigens is empty and this
        # yields no interfaces -> best_dockq=None, rather than silently scoring
        # all interfaces (incl. VH-VL).
        best_result = {k: v for k, v in best_result.items()
                       if len(_interface_chains(k) & native_antigens) == 1}

    # Headline DockQ: mean across the reported interfaces (so the number is
    # bounded [0, 1] regardless of how many interfaces the complex has).
    headline_dockq = (
        sum(v.get("DockQ", 0) for v in best_result.values()) / len(best_result)
        if best_result
        else None
    )

    return {
        "model": model_path.split("/")[-1],
        "native": native_path.split("/")[-1],
        "best_dockq": headline_dockq,
        "best_result": best_result,
        "best_mapping": best_mapping,
    }
