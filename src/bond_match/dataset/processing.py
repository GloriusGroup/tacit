"""Conversion of raw reaction entries into training samples.

Implements BE-matrix-based mechanism replay, LP-step handling, negative
samples, trajectory grouping, and pathway-based sample generation.
"""

import warnings
import torch
import numpy as np
from rdkit import Chem
import networkx as nx
import hashlib
from bond_match.blockspace import reconstruct_step as _reconstruct_step
from bond_match.blockspace import apply_step as _apply_step
from bond_match.blockspace import mol_to_be_matrix as _mol_to_be_matrix
from bond_match.catalyst_utils import fix_catalyst_smiles
import torch

from bond_match.dataset.metadata import (
    _extract_entry_yield,
    _sample_tensors_to_numpy,
)
from bond_match.dataset.mol_graph import (
    _be_to_nx,
    apply_elemental_step,
    augment_with_lp_edges,
    canonicalize_mol_ordering,
    connect_components_with_head,
    find_atom_by_identifier,
    get_graph_features_nx,
    mol_to_nx,
    nx_to_mol,
    smiles_match,
)

def _process_v3_mechanism(
    steps,
    initial_G,
    include_product,
    extended_features,
    has_catalyst,
    mech_idx=0,
    be_matrix=None,
    atoms_array=None,
    enable_history=False,
    reactant_smi="",
    product_smi="",
    enable_blockspace_product=True,
    canonical_rank=False,
):
    """Process a single v3 mechanism (list of steps) into training samples.

    Uses BE matrix operations (reconstruct_step + apply_step) to correctly
    track state through steps including LP/formal-charge changes. After each
    step, all changes (including LP changes) are applied to the BE matrix and
    the graph is rebuilt from it for feature extraction; catalyst flags are
    re-applied after each rebuild.

    Each step's changes are split into off-diagonal break pairs (value < 0)
    and form pairs (value > 0). A step with exactly two breaks and two forms
    is emitted as a standard two-bond-swap sample with per-bond labels
    ``swap_type + 1`` on the two breaking bonds (break pairs are sorted with
    i < j by reconstruct_step; swap type is inferred from the first form
    pair). Otherwise, diagonal BE changes are interpreted as lone-pair
    changes in units of two electrons; if the totals of electron sources
    (bond breaks plus LP losses) and sinks (bond forms plus LP gains) both
    equal two, an LP-step sample is emitted in which LP changes are encoded
    as breaks/forms of virtual LP-node edges added by augment_with_lp_edges.

    If ``include_product`` is set, a final stable-state sample (type
    ``"product"``, all-zero labels, ``is_over_target=True``) is appended; by
    default this is the blockspace-applied final state so side products and
    fragments produced by the mechanism are retained, unless
    ``enable_blockspace_product`` is False and a product SMILES is given.

    Args:
        steps: List of step dicts from mechanisms[i].steps
        initial_G: NetworkX graph of the initial reactant (with head node)
        include_product: Whether to add a final stable-state sample
        extended_features: Whether to include extended atom features
        has_catalyst: Whether the reaction has a catalyst
        mech_idx: Index of the mechanism (for sample type labeling)
        be_matrix: Initial BE matrix (n x n) from _mol_to_be_matrix
        atoms_array: Atom identity array (n x 2) with [map_number, atomic_number]
        enable_history: Whether to store previous step graph data for history context
        reactant_smi: Reactant SMILES stored in each sample
        product_smi: Product SMILES used when enable_blockspace_product is False
        enable_blockspace_product: If True, the final stable-state sample is
            built from the blockspace-applied state; otherwise from product_smi
        canonical_rank: Whether to include the canonical-rank node feature

    Returns:
        List of training sample dicts.
    """
    samples = []
    current_G = initial_G.copy()
    be_current = be_matrix.copy() if be_matrix is not None else None
    history_graphs = []

    catalyst_flags = None
    if extended_features:
        catalyst_flags = {}
        for node in initial_G.nodes():
            catalyst_flags[node] = initial_G.nodes[node].get("is_catalyst", 0)

    for step_idx, step in enumerate(steps):
        try:
            changes = _reconstruct_step(step, atoms_array, be_current)
        except Exception:
            break

        break_pairs = [(i, j) for i, j, v in changes if i != j and v < 0]
        form_pairs = [(i, j) for i, j, v in changes if i != j and v > 0]

        is_2bond_swap = len(break_pairs) == 2 and len(form_pairs) == 2

        if is_2bond_swap:
            try:
                x, edge_index, edge_attr, bonds_list, edge_map = get_graph_features_nx(
                    current_G,
                    extended_features=extended_features,
                    canonical_rank=canonical_rank,
                )
            except Exception:
                break

            if len(bonds_list) > 0:
                node_indices = sorted(list(current_G.nodes()))
                node_map_inv = {i: node_idx for i, node_idx in enumerate(node_indices)}

                x, edge_index, edge_attr, bonds_list, edge_map, _lp_node_map = (
                    augment_with_lp_edges(
                        x,
                        edge_index,
                        edge_attr,
                        bonds_list,
                        edge_map,
                        be_current,
                        node_map_inv,
                    )
                )

                label_matrix = np.zeros((len(bonds_list),), dtype=np.long)

                u_1, u_2 = break_pairs[0]
                v_1, v_2 = break_pairs[1]

                f1 = set(form_pairs[0])
                swap_type = -1

                if f1 == {u_1, v_1} or f1 == {u_2, v_2}:
                    swap_type = 0
                elif f1 == {u_1, v_2} or f1 == {u_2, v_1}:
                    swap_type = 1

                if swap_type != -1:
                    idx_a = edge_map.get((u_1, u_2))
                    idx_b = edge_map.get((v_1, v_2))

                    if idx_a is not None and idx_b is not None:
                        target = swap_type + 1
                        label_matrix[idx_a] = target
                        label_matrix[idx_b] = target

                        sample = {
                            "x": x,
                            "edge_index": edge_index,
                            "edge_attr": edge_attr,
                            "y": label_matrix,
                            "bonds": bonds_list,
                            "type": f"mech{mech_idx}_step_{step_idx}",
                            "is_over_target": False,
                            "has_catalyst": has_catalyst,
                            "step_index": step_idx,
                            "reactant_smi": reactant_smi,
                        }

                        if enable_history:
                            sample["history"] = list(history_graphs)
                            hist_entry = {
                                "x": x.clone()
                                if isinstance(x, torch.Tensor)
                                else torch.tensor(x),
                                "edge_index": edge_index.clone()
                                if isinstance(edge_index, torch.Tensor)
                                else torch.tensor(edge_index),
                                "edge_attr": edge_attr.clone()
                                if isinstance(edge_attr, torch.Tensor)
                                else torch.tensor(edge_attr),
                                "bonds": list(bonds_list),
                                "step_index": step_idx,
                                "reactant_smi": reactant_smi,
                            }
                            history_graphs.append(hist_entry)

                        samples.append(sample)

        else:
            diag_changes = [(i, v) for i, j, v in changes if i == j]

            lp_loss_units = []
            lp_gain_units = []
            for atom_idx, val in diag_changes:
                count = int(abs(val)) // 2
                if val < 0:
                    lp_loss_units.extend([atom_idx] * count)
                elif val > 0:
                    lp_gain_units.extend([atom_idx] * count)

            n_sources = len(break_pairs) + len(lp_loss_units)
            n_sinks = len(form_pairs) + len(lp_gain_units)

            is_lp_step = (
                n_sources == 2
                and n_sinks == 2
                and not is_2bond_swap
                and len(diag_changes) > 0
            )

            if is_lp_step:
                try:
                    x, edge_index, edge_attr, bonds_list, edge_map = (
                        get_graph_features_nx(
                            current_G,
                            extended_features=extended_features,
                            canonical_rank=canonical_rank,
                        )
                    )
                except Exception:
                    be_current = _apply_step(be_current, changes)
                    current_G = _be_to_nx(be_current, atoms_array, catalyst_flags)
                    continue

                if len(bonds_list) > 0:
                    node_indices = sorted(list(current_G.nodes()))
                    node_map_local = {
                        node_idx: i for i, node_idx in enumerate(node_indices)
                    }
                    node_map_inv = {
                        i: node_idx for i, node_idx in enumerate(node_indices)
                    }

                    x, edge_index, edge_attr, bonds_list, edge_map, lp_node_map = (
                        augment_with_lp_edges(
                            x,
                            edge_index,
                            edge_attr,
                            bonds_list,
                            edge_map,
                            be_current,
                            node_map_inv,
                        )
                    )

                    if lp_node_map:
                        label_matrix = np.zeros((len(bonds_list),), dtype=np.long)

                        virtual_breaks = list(break_pairs)
                        for atom_idx in lp_loss_units:
                            if atom_idx in lp_node_map:
                                virtual_breaks.append(
                                    tuple(
                                        sorted(
                                            (
                                                atom_idx,
                                                -1
                                                - list(lp_node_map.keys()).index(
                                                    atom_idx
                                                ),
                                            )
                                        )
                                    )
                                )

                        virtual_forms = list(form_pairs)
                        for atom_idx in lp_gain_units:
                            if atom_idx in lp_node_map:
                                virtual_forms.append(
                                    tuple(
                                        sorted(
                                            (
                                                atom_idx,
                                                -1
                                                - list(lp_node_map.keys()).index(
                                                    atom_idx
                                                ),
                                            )
                                        )
                                    )
                                )

                        if len(virtual_breaks) == 2 and len(virtual_forms) == 2:
                            idx_a = edge_map.get(virtual_breaks[0])
                            idx_b = edge_map.get(virtual_breaks[1])

                            if idx_a is not None and idx_b is not None:
                                b0_u, b0_v = bonds_list[idx_a]
                                b1_u, b1_v = bonds_list[idx_b]

                                vf_mapped = []
                                for vf in virtual_forms:
                                    if all(v >= 0 for v in vf):
                                        vf_mapped.append(
                                            set(node_map_local.get(a, a) for a in vf)
                                        )
                                    else:
                                        real_atom = [a for a in vf if a >= 0][0]
                                        lp_sentinel = [a for a in vf if a < 0][0]
                                        lp_atom_idx = list(lp_node_map.keys())[
                                            -(lp_sentinel + 1)
                                        ]
                                        vf_mapped.append(
                                            {
                                                node_map_local.get(
                                                    real_atom, real_atom
                                                ),
                                                lp_node_map[lp_atom_idx],
                                            }
                                        )

                                f1_set = vf_mapped[0] if vf_mapped else set()
                                swap_type = -1

                                if f1_set == {b0_u, b1_u} or f1_set == {b0_v, b1_v}:
                                    swap_type = 0
                                elif f1_set == {b0_u, b1_v} or f1_set == {b0_v, b1_u}:
                                    swap_type = 1

                                if swap_type != -1:
                                    target = swap_type + 1
                                    label_matrix[idx_a] = target
                                    label_matrix[idx_b] = target

                                    sample = {
                                        "x": x,
                                        "edge_index": edge_index,
                                        "edge_attr": edge_attr,
                                        "y": label_matrix,
                                        "bonds": bonds_list,
                                        "type": f"mech{mech_idx}_step_{step_idx}_lp",
                                        "is_over_target": False,
                                        "has_catalyst": has_catalyst,
                                        "step_index": step_idx,
                                        "reactant_smi": reactant_smi,
                                    }

                                    if enable_history:
                                        sample["history"] = list(history_graphs)
                                        hist_entry = {
                                            "x": x.clone()
                                            if isinstance(x, torch.Tensor)
                                            else torch.tensor(x),
                                            "edge_index": edge_index.clone()
                                            if isinstance(edge_index, torch.Tensor)
                                            else torch.tensor(edge_index),
                                            "edge_attr": edge_attr.clone()
                                            if isinstance(edge_attr, torch.Tensor)
                                            else torch.tensor(edge_attr),
                                            "bonds": list(bonds_list),
                                            "step_index": step_idx,
                                            "reactant_smi": reactant_smi,
                                        }
                                        history_graphs.append(hist_entry)

                                    samples.append(sample)

        be_current = _apply_step(be_current, changes)

        current_G = _be_to_nx(be_current, atoms_array, catalyst_flags)

    if include_product and len(current_G.nodes) > 0:
        try:
            product_G = current_G
            product_be = be_current
            if not enable_blockspace_product and product_smi:
                product_mol = Chem.MolFromSmiles(product_smi)
                if product_mol is not None:
                    product_mol = Chem.AddHs(product_mol)
                    product_mol = canonicalize_mol_ordering(product_mol)
                    product_G = mol_to_nx(product_mol)
                    product_be, _ = _mol_to_be_matrix(product_mol)

            x_p, edge_index_p, edge_attr_p, bonds_p, edge_map_p = get_graph_features_nx(
                product_G,
                extended_features=extended_features,
                canonical_rank=canonical_rank,
            )
        except Exception:
            return samples

        if len(bonds_p) > 0:
            node_indices_p = sorted(list(product_G.nodes()))
            node_map_inv_p = {i: node_idx for i, node_idx in enumerate(node_indices_p)}

            x_p, edge_index_p, edge_attr_p, bonds_p, edge_map_p, _lp_map_p = (
                augment_with_lp_edges(
                    x_p,
                    edge_index_p,
                    edge_attr_p,
                    bonds_p,
                    edge_map_p,
                    product_be,
                    node_map_inv_p,
                )
            )

            label_p = np.zeros((len(bonds_p),), dtype=np.long)
            product_sample = {
                "x": x_p,
                "edge_index": edge_index_p,
                "edge_attr": edge_attr_p,
                "y": label_p,
                "bonds": bonds_p,
                "type": "product",
                "is_over_target": True,
                "has_catalyst": has_catalyst,
                "step_index": len(steps),
                "reactant_smi": "",
            }
            if enable_history:
                product_sample["history"] = list(history_graphs)
            samples.append(product_sample)

    return samples

def prepare_v3_reactant(entry, extended_features=False):
    """Prepare the initial reactant mol, graph, and BE matrix for a v3 entry.

    Shared between training (process_reaction_entry_v3) and evaluation
    (InferenceDataset._process_v3) to guarantee identical preprocessing.
    Catalyst SMILES from ``metadata.cat`` are repaired with
    fix_catalyst_smiles and appended to the reactant; unfixable catalysts
    are dropped with a warning. Explicit hydrogens are added, atom ordering
    is canonicalized, and with ``extended_features`` an ``is_catalyst``
    node attribute is assigned (atoms with index >= the reactant heavy-atom
    count, and hydrogens attached to such atoms, are flagged).

    Args:
        entry: Raw reaction entry with ``reaction_smiles_mapped`` (fallback
            ``reaction_smiles``) and optional ``metadata.cat``.
        extended_features: Whether to annotate ``is_catalyst`` node flags.

    Returns:
        Dict with keys ``mol``, ``G``, ``be_matrix``, ``atoms_array``,
        ``has_catalyst``, ``n_reactant_heavy``, ``reactant_smi``,
        ``product_smi``, and ``full_ranks``, or None on failure.
    """
    rxn_smiles = entry.get("reaction_smiles_mapped") or entry.get("reaction_smiles", "")
    if not rxn_smiles or ">>" not in rxn_smiles:
        return None

    try:
        reactant_smi, product_smi = rxn_smiles.split(">>")
        mol_r = Chem.MolFromSmiles(reactant_smi)
        if not mol_r:
            return None
    except Exception:
        return None

    cat = ""
    raw_cat = ""
    if entry.get("metadata") and entry.get("metadata").get("cat"):
        raw_cat = entry["metadata"]["cat"]
        fixed_cat = fix_catalyst_smiles(raw_cat)
        if fixed_cat:
            cat = "." + fixed_cat
        else:
            warnings.warn(
                f"Catalyst SMILES unfixable, dropping catalyst from entry: {raw_cat}",
                stacklevel=2,
            )

    has_catalyst = bool(cat)

    if cat:
        mol_r = Chem.MolFromSmiles(reactant_smi + cat)
        if not mol_r:
            warnings.warn(
                f"Reactant+catalyst concat failed even after fix: cat={raw_cat}",
                stacklevel=2,
            )
            return None

    n_reactant_heavy = 0
    if extended_features and has_catalyst:
        reactant_only = Chem.MolFromSmiles(reactant_smi)
        if reactant_only:
            n_reactant_heavy = reactant_only.GetNumAtoms()

    mol_r = Chem.AddHs(mol_r)
    mol_r, full_ranks = canonicalize_mol_ordering(mol_r, return_mapping=True)
    initial_G = mol_to_nx(mol_r)

    be_matrix, atoms_array = _mol_to_be_matrix(mol_r)

    if extended_features:
        if has_catalyst and n_reactant_heavy > 0:
            for node in initial_G.nodes():
                atom_data = initial_G.nodes[node]
                atomic_num = atom_data.get("atomic_number", 0)
                if atomic_num == 0:
                    atom_data["is_catalyst"] = 0
                elif atomic_num == 1:
                    is_cat = 0
                    for nbr in initial_G.neighbors(node):
                        if (
                            initial_G.nodes[nbr].get("atomic_number", 0) > 1
                            and nbr >= n_reactant_heavy
                        ):
                            is_cat = 1
                            break
                    atom_data["is_catalyst"] = is_cat
                elif node < n_reactant_heavy:
                    atom_data["is_catalyst"] = 0
                else:
                    atom_data["is_catalyst"] = 1
        else:
            for node in initial_G.nodes():
                initial_G.nodes[node]["is_catalyst"] = 0

    return {
        "mol": mol_r,
        "G": initial_G,
        "be_matrix": be_matrix,
        "atoms_array": atoms_array,
        "has_catalyst": has_catalyst,
        "n_reactant_heavy": n_reactant_heavy,
        "reactant_smi": reactant_smi,
        "product_smi": product_smi,
        "full_ranks": full_ranks,
    }

def _process_negative_reaction(entry, extended_features=False, canonical_rank=False):
    """Generate a single training sample for a non-working reaction (Yield == 0).

    All bond targets are zero (no reactive bonds), and the stop action is the
    target. Reuses prepare_v3_reactant / get_graph_features_nx /
    augment_with_lp_edges for consistent preprocessing.
    """
    prep = prepare_v3_reactant(entry, extended_features=extended_features)
    if prep is None:
        return []

    try:
        x, edge_index, edge_attr, bonds_list, edge_map = get_graph_features_nx(
            prep["G"],
            extended_features=extended_features,
            canonical_rank=canonical_rank,
        )
    except Exception:
        return []

    if len(bonds_list) < 2:
        return []

    node_indices = sorted(list(prep["G"].nodes()))
    node_map_inv = {i: node_idx for i, node_idx in enumerate(node_indices)}
    x, edge_index, edge_attr, bonds_list, edge_map, _ = augment_with_lp_edges(
        x,
        edge_index,
        edge_attr,
        bonds_list,
        edge_map,
        prep["be_matrix"],
        node_map_inv,
    )

    sample = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": np.zeros(len(bonds_list), dtype=np.long),
        "bonds": bonds_list,
        "type": "negative_reaction",
        "is_negative_reaction": True,
        "is_over_target": 1,
        "has_catalyst": prep["has_catalyst"],
        "step_index": 0,
        "reactant_smi": prep.get("reactant_smi", ""),
    }

    return [_sample_tensors_to_numpy(sample)]

def _is_reactant_equals_product(entry):
    """Return True when the stored reaction explicitly encodes no change."""
    reaction_fields = (
        entry.get("reaction_smiles_mapped"),
        entry.get("reaction_smiles"),
        entry.get("metadata", {}).get("Reaction SMILES"),
    )
    for rxn in reaction_fields:
        if not rxn or ">>" not in rxn:
            continue
        reactants, products = rxn.split(">>", 1)
        return reactants == products
    return False

def process_reaction_entry_v3(
    entry,
    include_product=True,
    extended_features=False,
    use_all_mechanisms=False,
    enable_history=False,
    enable_blockspace_product=True,
    canonical_rank=False,
):
    """Process a v3 blockspace reaction entry into training samples.

    Key differences from v2:
    - Uses reaction_smiles_mapped (fallback reaction_smiles) to preserve atom maps
    - Reads mechanisms[i].steps instead of pathway
    - Bond labels are strings like "C:12", "H:0~N:15" instead of dicts
    - Supports catalyst via metadata.cat (same as v2)
    - use_all_mechanisms: if True, processes all alternative mechanisms as data augmentation

    Yield==0 entries are handled in two ways: explicit no-reaction entries
    (reactant equals product) become a single stop-target negative sample via
    _process_negative_reaction, while yield==0 entries that still carry a
    mechanism are processed normally but their mechanistic samples are
    flagged with ``is_negative_reaction=True``.

    Args:
        entry: Raw v3 blockspace reaction entry.
        include_product: Whether to append a final stable-state sample.
        extended_features: Whether to include extended atom features.
        use_all_mechanisms: Process all mechanisms instead of only the first.
        enable_history: Store previous step graphs as history context.
        enable_blockspace_product: Build the final state from the
            blockspace-applied BE matrix rather than the product SMILES.
        canonical_rank: Whether to include the canonical-rank node feature.

    Returns:
        List of sample dicts with tensors converted to numpy arrays.
    """
    samples = []
    if entry.get("success") is False:
        return samples

    yield_val = _extract_entry_yield(entry)
    is_mechanistic_negative = False
    if yield_val is not None and yield_val == 0 and _is_reactant_equals_product(entry):
        return _process_negative_reaction(entry, extended_features, canonical_rank)
    if yield_val is not None and yield_val == 0:
        is_mechanistic_negative = True

    mechanisms = entry.get("mechanisms", [])
    if not mechanisms or not mechanisms[0].get("steps"):
        return samples

    prep = prepare_v3_reactant(entry, extended_features=extended_features)
    if prep is None:
        return samples

    initial_G = prep["G"]
    be_matrix = prep["be_matrix"]
    atoms_array = prep["atoms_array"]
    has_catalyst = prep["has_catalyst"]
    reactant_smi = prep.get("reactant_smi", "")
    product_smi = prep.get("product_smi", "")

    mechs_to_process = mechanisms if use_all_mechanisms else mechanisms[:1]
    for mech_idx, mechanism in enumerate(mechs_to_process):
        steps = mechanism.get("steps", [])
        if not steps:
            continue
        generated_samples = _process_v3_mechanism(
                steps,
                initial_G,
                include_product,
                extended_features,
                has_catalyst,
                mech_idx=mech_idx,
                be_matrix=be_matrix,
                atoms_array=atoms_array,
                reactant_smi=reactant_smi,
                product_smi=product_smi,
                enable_blockspace_product=enable_blockspace_product,
                enable_history=enable_history,
                canonical_rank=canonical_rank,
            )
        if is_mechanistic_negative:
            for sample in generated_samples:
                if sample.get("type", "").startswith("mech"):
                    sample["is_negative_reaction"] = True
        samples.extend(generated_samples)

    return [_sample_tensors_to_numpy(s) for s in samples]

def process_reaction_entry_v3_trajectory(
    entry,
    include_product=True,
    extended_features=False,
    use_all_mechanisms=False,
    enable_blockspace_product=True,
    canonical_rank=False,
):
    """Process a v3 entry into trajectory samples for trajectory-mode training.

    Runs process_reaction_entry_v3 (without history) and groups all
    resulting mechanistic/product states into one trajectory dict with type
    ``"trajectory"``.

    Args:
        entry: Raw v3 blockspace reaction entry.
        include_product: Whether to append a final stable-state sample.
        extended_features: Whether to include extended atom features.
        use_all_mechanisms: Process all mechanisms instead of only the first.
        enable_blockspace_product: Build the final state from the
            blockspace-applied BE matrix rather than the product SMILES.
        canonical_rank: Whether to include the canonical-rank node feature.

    Returns:
        List of trajectory dicts, each with ``states`` and summary metadata.
    """
    states = process_reaction_entry_v3(
        entry,
        include_product=include_product,
        extended_features=extended_features,
        use_all_mechanisms=use_all_mechanisms,
        enable_history=False,
        enable_blockspace_product=enable_blockspace_product,
        canonical_rank=canonical_rank,
    )
    if not states:
        return []

    return [
        {
            "states": states,
            "type": "trajectory",
            "has_catalyst": any(s.get("has_catalyst", False) for s in states),
            "step_index": max(int(s.get("step_index", 0)) for s in states),
            "reactant_smi": next(
                (s.get("reactant_smi", "") for s in states if s.get("reactant_smi")),
                "",
            ),
        }
    ]

def process_reaction_entry(
    entry, include_product=True, extended_features=False, enable_history=False
):
    """Process a pathway-based reaction entry into training samples.

    Replays the ``pathway`` step list on a NetworkX graph: for each step,
    the two breaking bonds are resolved with find_atom_by_identifier, the
    swap type is inferred from the first formed bond, per-bond labels
    ``swap_type + 1`` are assigned, and the step is applied with
    apply_elemental_step. Head-node connectivity is re-established after
    every step because applying a step may split a molecule and isolate a
    fragment from the previous head node. Steps with
    ``fragments_discarded`` are validated by checking that one connected
    component matches the step's ``result_smiles``; replay stops on validation
    failure. If
    ``include_product`` is set, a final stable-state sample with all-zero
    labels is appended.

    Args:
        entry: Raw reaction entry with ``reaction_smiles`` and ``pathway``.
        include_product: Whether to append a final stable-state sample.
        extended_features: Whether to include extended atom features and
            ``is_catalyst`` flags.
        enable_history: Store previous step graphs as history context.

    Returns:
        List of sample dicts with tensors converted to numpy arrays.
    """
    samples = []
    history_graphs = []
    if entry.get("success") is False:
        return samples

    rxn_smiles = entry["reaction_smiles"]
    try:
        reactant_smi, _ = rxn_smiles.split(">>")
        reactant_smi = Chem.MolToSmiles(
            Chem.MolFromSmiles(reactant_smi), ignoreAtomMapNumbers=True, canonical=True
        )
    except:
        return []

    cat = ""
    raw_cat = ""
    if entry.get("metadata") and entry.get("metadata").get("cat"):
        raw_cat = entry["metadata"]["cat"]
        fixed_cat = fix_catalyst_smiles(raw_cat)
        if fixed_cat:
            cat = "." + fixed_cat
        else:
            warnings.warn(
                f"Catalyst SMILES unfixable, dropping catalyst from entry: {raw_cat[:80]}",
                stacklevel=2,
            )

    has_catalyst = bool(cat)

    mol_r = Chem.MolFromSmiles(reactant_smi + cat)
    if not mol_r:
        warnings.warn(
            f"Reactant+catalyst concat failed even after fix: cat={raw_cat[:80]}",
            stacklevel=2,
        )
        return []

    n_reactant_heavy = 0
    if extended_features and has_catalyst:
        reactant_only = Chem.MolFromSmiles(reactant_smi)
        if reactant_only:
            n_reactant_heavy = reactant_only.GetNumAtoms()

    mol_r = Chem.AddHs(mol_r)

    current_G = mol_to_nx(mol_r)

    # Annotate graph nodes with catalyst membership when extended features are enabled.
    if extended_features:
        if has_catalyst and n_reactant_heavy > 0:
            for node in current_G.nodes():
                atom_data = current_G.nodes[node]
                atomic_num = atom_data.get("atomic_number", 0)
                if atomic_num == 0:
                    atom_data["is_catalyst"] = 0
                elif atomic_num == 1:
                    is_cat = 0
                    for nbr in current_G.neighbors(node):
                        if (
                            current_G.nodes[nbr].get("atomic_number", 0) > 1
                            and nbr >= n_reactant_heavy
                        ):
                            is_cat = 1
                            break
                    atom_data["is_catalyst"] = is_cat
                elif node < n_reactant_heavy:
                    atom_data["is_catalyst"] = 0
                else:
                    atom_data["is_catalyst"] = 1
        else:
            for node in current_G.nodes():
                current_G.nodes[node]["is_catalyst"] = 0

    pathway = entry.get("pathway", [])

    for step_idx, step in enumerate(pathway):
        try:
            x, edge_index, edge_attr, bonds_list, edge_map = get_graph_features_nx(
                current_G,
                extended_features=extended_features,
                canonical_rank=canonical_rank,
            )
        except Exception as e:
            print(f"Ignoring error {e} for smiles {rxn_smiles}")
            break

        if len(bonds_list) > 0:
            label_matrix = np.zeros((len(bonds_list),), dtype=np.long)

            try:
                n1 = find_atom_by_identifier(current_G, step["break"][0]["atom1"])
                n2 = find_atom_by_identifier(current_G, step["break"][0]["atom2"])
                n3 = find_atom_by_identifier(current_G, step["break"][1]["atom1"])
                n4 = find_atom_by_identifier(current_G, step["break"][1]["atom2"])
            except:
                break

            if None in [n1, n2, n3, n4]:
                break

            # Determine which bond-pairing pattern describes the two formed bonds.
            try:
                f1_n1 = find_atom_by_identifier(current_G, step["form"][0]["atom1"])
                f1_n2 = find_atom_by_identifier(current_G, step["form"][0]["atom2"])
            except:
                break

            form_pair = {f1_n1, f1_n2}
            swap_type = -1

            u_1, u_2 = sorted([n1, n2])
            v_1, v_2 = sorted([n3, n4])

            if form_pair == {u_1, v_1} or form_pair == {u_2, v_2}:
                swap_type = 0
            elif form_pair == {u_1, v_2} or form_pair == {u_2, v_1}:
                swap_type = 1

            if swap_type != -1:
                idx_a = edge_map.get((u_1, u_2))
                idx_b = edge_map.get((v_1, v_2))

                if idx_a is not None and idx_b is not None:
                    target = swap_type + 1
                    label_matrix[idx_a] = target
                    label_matrix[idx_b] = target

                    sample = {
                        "x": x,
                        "edge_index": edge_index,
                        "edge_attr": edge_attr,
                        "y": label_matrix,
                        "bonds": bonds_list,
                        "type": f"step_{step_idx}",
                        "is_over_target": False,
                        "has_catalyst": has_catalyst,
                        "step_index": step_idx,
                    }

                    if enable_history:
                        sample["history"] = list(history_graphs)
                        history_graphs.append(
                            {
                                "x": x.clone()
                                if isinstance(x, torch.Tensor)
                                else torch.tensor(x),
                                "edge_index": edge_index.clone()
                                if isinstance(edge_index, torch.Tensor)
                                else torch.tensor(edge_index),
                                "edge_attr": edge_attr.clone()
                                if isinstance(edge_attr, torch.Tensor)
                                else torch.tensor(edge_attr),
                                "bonds": list(bonds_list),
                                "step_index": step_idx,
                                "reactant_smi": reactant_smi,
                            }
                        )

                    samples.append(sample)

            try:
                next_G = apply_elemental_step(
                    current_G, (u_1, u_2), (v_1, v_2), swap_type
                )
            except:
                break

            # Reconnect virtual head nodes after a step that may split components.
            next_G = connect_components_with_head(next_G)

            if step.get("fragments_discarded"):
                expected = step.get("result_smiles")

                # Compare SMILES using only real atoms; virtual head nodes are excluded.
                real_nodes = [
                    n
                    for n, d in next_G.nodes(data=True)
                    if d.get("atomic_number", 0) > 0
                ]
                check_G = next_G.subgraph(real_nodes)  # This splits into components

                comps = [check_G.subgraph(c) for c in nx.connected_components(check_G)]

                found_match = False
                for c in comps:
                    mol_c = nx_to_mol(c)
                    smi_c = Chem.MolToSmiles(mol_c) if mol_c else ""
                    if smiles_match(smi_c, expected):
                        found_match = True
                        break

                if not found_match:
                    break

                current_G = next_G
            else:
                current_G = next_G

    if include_product and len(current_G.nodes) > 0:
        try:
            x_p, edge_index_p, edge_attr_p, bonds_p, _ = get_graph_features_nx(
                current_G,
                extended_features=extended_features,
                canonical_rank=canonical_rank,
            )
        except Exception as e:
            print(f"Ignoring error for product {e} for smiles {rxn_smiles}")
            return samples

        if len(bonds_p) > 0:
            label_p = np.zeros((len(bonds_p),), dtype=np.long)
            product_sample = {
                "x": x_p,
                "edge_index": edge_index_p,
                "edge_attr": edge_attr_p,
                "y": label_p,
                "bonds": bonds_p,
                "type": "product",
                "is_over_target": True,
                "has_catalyst": has_catalyst,
                "step_index": len(pathway),
            }
            if enable_history:
                product_sample["history"] = list(history_graphs)
            samples.append(product_sample)

    return [_sample_tensors_to_numpy(s) for s in samples]
