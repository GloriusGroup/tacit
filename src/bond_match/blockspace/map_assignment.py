"""
Atom-map number assignment for partially mapped reactions.

``assign_full_mapping`` gives every unmapped non-H atom a map number so that
substrate and product can be matched index-free.  When the same element
appears unmapped on both sides, ``find_best_unmapped_assignment`` searches
pairings and keeps the one whose resulting R matrix has the fewest bond
changes (i.e. the most chemically economical interpretation).
"""

from collections import defaultdict
from itertools import combinations, permutations
from typing import Dict, Optional, Tuple

import numpy as np
from rdkit import Chem

from atom_matching import match_atoms_by_map
from be_core import smiles_to_matrices
from r_matrix import find_r_matrix


def assign_full_mapping(
    sub_smi: str,
    prod_smi: str,
    atoms_sub_orig: np.ndarray,
    forced_heavy_matches: Optional[Dict[int, int]] = None,
) -> Tuple[str, str, np.ndarray]:
    """
    Assign map numbers to all unmapped non-H atoms in substrate and product.

    New map numbers start above the maximum existing map number.  Forced
    heavy-atom matches receive the same number on both sides; remaining
    unmapped substrate atoms (byproducts) and product atoms (new atoms) each
    receive independent numbers.  Unmapped hydrogens are removed from the
    output SMILES; mapped hydrogens are kept.

    Args:
        sub_smi: Substrate SMILES
        prod_smi: Product SMILES
        atoms_sub_orig: Atom array from smiles_to_matrices (same parse as sub_smi)
        forced_heavy_matches: Optional dict {sub_idx: prod_idx} forcing specific
            unmapped heavy atom pairings. If None, unmapped heavy atoms get
            independent numbers (byproduct/new).

    Returns:
        (mapped_sub_smi, mapped_prod_smi, updated_atoms_sub)
    """
    sub_mol = Chem.MolFromSmiles(sub_smi, sanitize=False)
    prod_mol = Chem.MolFromSmiles(prod_smi, sanitize=False)
    if sub_mol is None or prod_mol is None:
        return sub_smi, prod_smi, atoms_sub_orig
    Chem.SanitizeMol(sub_mol)
    Chem.SanitizeMol(prod_mol)

    max_map = 0
    for atom in sub_mol.GetAtoms():
        max_map = max(max_map, atom.GetAtomMapNum())
    for atom in prod_mol.GetAtoms():
        max_map = max(max_map, atom.GetAtomMapNum())

    sub_mol_h = Chem.AddHs(sub_mol)
    prod_mol_h = Chem.AddHs(prod_mol)

    atoms_mapped = atoms_sub_orig.copy()
    next_map = max_map + 1

    matched_sub = set()
    if forced_heavy_matches:
        for si, pi in forced_heavy_matches.items():
            sub_mol_h.GetAtomWithIdx(si).SetAtomMapNum(next_map)
            prod_mol_h.GetAtomWithIdx(pi).SetAtomMapNum(next_map)
            atoms_mapped[si][0] = next_map
            matched_sub.add(si)
            next_map += 1

    be_sub, atoms_sub = smiles_to_matrices(
        sub_smi, kekulize=True, include_hydrogens=True
    )

    for si in range(len(atoms_sub)):
        map_num, z = int(atoms_sub[si][0]), int(atoms_sub[si][1])
        if map_num != 0 or z == 1 or si in matched_sub:
            continue
        sub_mol_h.GetAtomWithIdx(si).SetAtomMapNum(next_map)
        atoms_mapped[si][0] = next_map
        next_map += 1

    be_prod, atoms_prod = smiles_to_matrices(
        prod_smi, kekulize=True, include_hydrogens=True
    )
    for pi in range(len(atoms_prod)):
        z = int(atoms_prod[pi][1])
        if z == 1:
            continue
        if prod_mol_h.GetAtomWithIdx(pi).GetAtomMapNum() == 0:
            prod_mol_h.GetAtomWithIdx(pi).SetAtomMapNum(next_map)
            next_map += 1

    ps = Chem.RemoveHsParameters()
    ps.removeMapped = False
    sub_mapped = Chem.MolToSmiles(Chem.RemoveHs(sub_mol_h, ps))
    prod_mapped = Chem.MolToSmiles(Chem.RemoveHs(prod_mol_h, ps))

    return sub_mapped, prod_mapped, atoms_mapped


def find_best_unmapped_assignment(
    sub_smi: str,
    prod_smi: str,
    atoms_sub: np.ndarray,
) -> Optional[Dict[int, int]]:
    """
    Search pairings of unmapped heavy atoms between substrate and product.

    For each element with unmapped atoms on both sides, all combinations and
    permutations of pairings are generated.  Each candidate assignment is
    scored by computing the full R matrix and counting non-zero off-diagonal
    entries (each representing a bond change); invalid R matrices score
    infinity.  If the cartesian product across elements exceeds 10000
    options, a per-element greedy search is used instead of exhaustive
    enumeration.

    Args:
        sub_smi: Substrate SMILES
        prod_smi: Product SMILES
        atoms_sub: Substrate atom array (passed through to assign_full_mapping)

    Returns:
        The assignment dict {sub_idx: prod_idx} that minimizes R matrix
        changes, or None if there are no unmapped heavy atoms on both sides.
    """
    be_sub, atoms_sub_local = smiles_to_matrices(
        sub_smi, kekulize=True, include_hydrogens=True
    )
    be_prod, atoms_prod = smiles_to_matrices(
        prod_smi, kekulize=True, include_hydrogens=True
    )
    s2p, byp_idx, new_idx = match_atoms_by_map(
        atoms_sub_local, atoms_prod, be_sub, be_prod
    )

    matched_sub = set(s2p.keys())
    matched_prod = set(s2p.values())

    unmapped_sub_by_elem = defaultdict(list)
    unmapped_prod_by_elem = defaultdict(list)
    for si in range(len(atoms_sub_local)):
        m, z = int(atoms_sub_local[si][0]), int(atoms_sub_local[si][1])
        if m == 0 and z != 1 and si not in matched_sub:
            unmapped_sub_by_elem[z].append(si)
    for pi in range(len(atoms_prod)):
        m, z = int(atoms_prod[pi][0]), int(atoms_prod[pi][1])
        if m == 0 and z != 1 and pi not in matched_prod:
            unmapped_prod_by_elem[z].append(pi)

    shared_elems = [z for z in unmapped_sub_by_elem if z in unmapped_prod_by_elem]
    if not shared_elems:
        return None

    def _r_matrix_cost(forced_matches):
        """Compute full R matrix for this assignment, return non-zero entry count."""
        mapped_sub_smi, mapped_prod_smi, _ = assign_full_mapping(
            sub_smi, prod_smi, atoms_sub, forced_heavy_matches=forced_matches
        )
        try:
            be_s, at_s = smiles_to_matrices(
                mapped_sub_smi, kekulize=True, include_hydrogens=True
            )
            be_p, at_p = smiles_to_matrices(
                mapped_prod_smi, kekulize=True, include_hydrogens=True
            )
            r, valid = find_r_matrix(be_s, at_s, be_p, at_p)
            if not valid:
                return float("inf")
            n = len(r)
            cost = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(r[i, j]) > 1e-6:
                        cost += 1
            return cost
        except Exception:
            return float("inf")

    per_elem_options = []
    for z in shared_elems:
        s_indices = unmapped_sub_by_elem[z]
        p_indices = unmapped_prod_by_elem[z]
        n_match = min(len(s_indices), len(p_indices))
        if n_match == 0:
            continue
        elem_options = []
        for sub_combo in combinations(s_indices, n_match):
            for prod_perm in permutations(p_indices, n_match):
                elem_options.append(list(zip(sub_combo, prod_perm)))
        per_elem_options.append(elem_options)

    if not per_elem_options:
        return None

    total = 1
    for opts in per_elem_options:
        total *= len(opts)
    if total > 10000:
        best_matches = {}
        for elem_opts in per_elem_options:
            best_cost = float("inf")
            best_opt = None
            for opt in elem_opts:
                trial = dict(best_matches)
                trial.update(dict(opt))
                c = _r_matrix_cost(trial)
                if c < best_cost:
                    best_cost = c
                    best_opt = opt
            if best_opt:
                best_matches.update(dict(best_opt))
        return best_matches if best_matches else None

    from itertools import product as cart_product

    best_cost = float("inf")
    best_matches = None
    for combo in cart_product(*per_elem_options):
        trial = {}
        for pairs in combo:
            trial.update(dict(pairs))
        c = _r_matrix_cost(trial)
        if c < best_cost:
            best_cost = c
            best_matches = trial
    return best_matches
