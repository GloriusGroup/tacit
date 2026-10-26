"""
R matrix (reaction difference) computation and completion.

The R matrix is the elementwise difference between product and substrate BE
matrices for matched atoms (``BE_product - BE_substrate``).  Byproduct atoms
(present in substrate but not product) have no product reference, so their
entries must be *completed* such that the full R matrix conserves electrons
(``sum(R) == 0``).  Two completion strategies are provided:

- LP-only: electrons from broken bonds become lone pairs.
- Bond-forming: byproduct atoms bond with each other (e.g. H + Cl -> HCl),
  falling back to LP-only if that would produce invalid lone pair counts.

``find_r_matrix`` additionally searches over ambiguous unmapped-hydrogen source
assignments to find an LP-balanced R matrix.
"""

from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np

from atom_matching import _get_h_candidates_and_base_matching, match_atoms_by_map
from be_core import PT


def compute_r_matrix_partial(
    be_substrate: np.ndarray,
    atoms_substrate: np.ndarray,
    be_product: np.ndarray,
    atoms_product: np.ndarray,
) -> Tuple[np.ndarray, List[int], int]:
    """
    Compute the R matrix for matched atoms (determined by product structure).

    Args:
        be_substrate: Substrate BE matrix
        atoms_substrate: Substrate atom identity
        be_product: Product BE matrix
        atoms_product: Product atom identity

    Returns:
        r_matrix: Partial R matrix (same size as substrate, byproduct entries are 0)
        byproduct_indices: Indices of byproduct atoms in substrate
        electron_deficit: Electrons needed to balance (should go to byproducts)
    """
    n_sub = len(atoms_substrate)
    r_matrix = np.zeros((n_sub, n_sub), dtype=np.float32)

    substrate_to_product, byproduct_indices, _ = match_atoms_by_map(
        atoms_substrate, atoms_product, be_substrate, be_product
    )

    # Matched atoms are compared directly between substrate and product.
    for sub_i, prod_i in substrate_to_product.items():
        # Diagonal entries encode lone-pair changes.
        r_matrix[sub_i, sub_i] = be_product[prod_i, prod_i] - be_substrate[sub_i, sub_i]

        # Off-diagonal entries encode bond-order changes between matched atoms.
        for sub_j, prod_j in substrate_to_product.items():
            if sub_i < sub_j:
                diff = be_product[prod_i, prod_j] - be_substrate[sub_i, sub_j]
                r_matrix[sub_i, sub_j] = diff
                r_matrix[sub_j, sub_i] = diff

        # Bonds from matched atoms to byproduct atoms are removed.
        for by_idx in byproduct_indices:
            if be_substrate[sub_i, by_idx] > 0:
                r_matrix[sub_i, by_idx] = -be_substrate[sub_i, by_idx]
                r_matrix[by_idx, sub_i] = -be_substrate[sub_i, by_idx]

    electron_deficit = -np.sum(r_matrix)

    return r_matrix, byproduct_indices, electron_deficit


def _complete_lp_only(r_matrix: np.ndarray, byproduct_indices: List[int]) -> np.ndarray:
    """LP-only byproduct completion: broken bond electrons go to lone pairs."""
    r_complete = r_matrix.copy()
    participating = []

    for by_idx in byproduct_indices:
        electrons = 0
        for j in range(len(r_matrix)):
            if j != by_idx and r_matrix[by_idx, j] < 0:
                electrons += abs(r_matrix[by_idx, j]) * 2
        r_complete[by_idx, by_idx] = int(round(electrons))
        if electrons > 0:
            participating.append(by_idx)

    # Distribute imbalance in electron-pair units to keep lone-pair counts even.
    imbalance = np.sum(r_complete)
    if abs(imbalance) > 1e-6:
        targets = participating if participating else list(byproduct_indices)
        if targets:
            total = int(round(-imbalance))
            units = total // 2  # number of lone pairs to distribute
            n = len(targets)
            base_units = units // n
            leftover_units = units - base_units * n
            for i, by_idx in enumerate(targets):
                extra = 1 if i < leftover_units else 0
                r_complete[by_idx, by_idx] += (base_units + extra) * 2

    return r_complete


def _complete_with_bonds(
    r_matrix: np.ndarray,
    be_substrate: np.ndarray,
    atoms_substrate: np.ndarray,
    byproduct_indices: List[int],
) -> Optional[np.ndarray]:
    """
    Bond-forming byproduct completion: form bonds between byproduct atoms,
    remaining electrons go to lone pairs. Returns None if result has invalid LP.
    """
    r_complete = r_matrix.copy()

    # Count electrons contributed by bonds broken from byproduct atoms.
    broken_electrons = {}
    for by_idx in byproduct_indices:
        electrons = 0
        for j in range(len(r_matrix)):
            if j != by_idx and r_matrix[by_idx, j] < 0:
                electrons += abs(r_matrix[by_idx, j]) * 2
        broken_electrons[by_idx] = int(round(electrons))

    # Bonding capacity is derived from broken bond orders.
    capacity = {i: broken_electrons[i] // 2 for i in byproduct_indices}

    # Detect unmapped H-H byproduct pairs that can supply hydrogen atoms.
    byproduct_set = set(byproduct_indices)
    h2_pairs_internal = []
    for by_idx in byproduct_indices:
        if int(atoms_substrate[by_idx][1]) != 1:
            continue
        if int(atoms_substrate[by_idx][0]) != 0:
                continue
        for j in range(len(be_substrate)):
            if j != by_idx and be_substrate[by_idx, j] > 0:
                if (
                    j in byproduct_set
                    and int(atoms_substrate[j][1]) == 1
                    and int(atoms_substrate[j][0]) == 0
                ):
                    pair = (min(by_idx, j), max(by_idx, j))
                    if pair not in set(h2_pairs_internal):
                        h2_pairs_internal.append(pair)
                break

    # Negatively charged byproducts can receive additional H atoms when H2 is
    # present and available for bond reorganization.
    if h2_pairs_internal:
        n = len(be_substrate)
        for by_idx in byproduct_indices:
            z = int(atoms_substrate[by_idx][1])
            if z == 1:
                continue
            v = PT.GetNOuterElecs(z)
            lp = be_substrate[by_idx, by_idx]
            b = sum(float(be_substrate[by_idx, j]) for j in range(n) if j != by_idx)
            formal_charge = v - lp - b
            if formal_charge < -0.5:
                extra = int(round(abs(formal_charge)))
                capacity[by_idx] = capacity.get(by_idx, 0) + extra

    h_byproducts = [
        i
        for i in byproduct_indices
        if atoms_substrate[i][1] == 1 and capacity.get(i, 0) > 0
    ]
    heavy_byproducts = [
        i
        for i in byproduct_indices
        if atoms_substrate[i][1] != 1 and capacity.get(i, 0) > 0
    ]

    # Prefer hydrogen-heavy atom bonds, then pair remaining hydrogens.
    bond_pairs = []
    heavy_start = 0
    for h_idx in h_byproducts:
        if capacity[h_idx] <= 0:
            continue
        for k in range(len(heavy_byproducts)):
            idx = (heavy_start + k) % len(heavy_byproducts)
            heavy_idx = heavy_byproducts[idx]
            if capacity[heavy_idx] <= 0:
                continue
            bond_pairs.append((heavy_idx, h_idx))
            capacity[heavy_idx] -= 1
            capacity[h_idx] -= 1
            heavy_start = (idx + 1) % len(heavy_byproducts)
            break

    remaining_h = [i for i in h_byproducts if capacity.get(i, 0) > 0]
    for k in range(0, len(remaining_h) - 1, 2):
        bond_pairs.append((remaining_h[k], remaining_h[k + 1]))
        capacity[remaining_h[k]] -= 1
        capacity[remaining_h[k + 1]] -= 1

    # Break internal H-H bonds when electronegative byproducts can accept hydrogen.
    ELECTRONEG_Z = {7, 8, 9, 15, 16, 17, 35, 53}  # N, O, F, P, S, Cl, Br, I
    remaining_electroneg = [
        i
        for i in heavy_byproducts
        if capacity.get(i, 0) > 0 and int(atoms_substrate[i][1]) in ELECTRONEG_Z
    ]
    if remaining_electroneg and h2_pairs_internal:
        for h_a, h_b in h2_pairs_internal:
            if not remaining_electroneg:
                break
            r_complete[h_a, h_b] -= 1
            r_complete[h_b, h_a] -= 1
            broken_electrons[h_a] = broken_electrons.get(h_a, 0) + 2
            broken_electrons[h_b] = broken_electrons.get(h_b, 0) + 2
            capacity[h_a] = capacity.get(h_a, 0) + 1
            capacity[h_b] = capacity.get(h_b, 0) + 1

            # Form bonds between released hydrogens and electronegative byproducts.
            for h_idx in (h_a, h_b):
                if capacity[h_idx] <= 0:
                    continue
                for k, heavy_idx in enumerate(remaining_electroneg):
                    if capacity[heavy_idx] <= 0:
                        continue
                    bond_pairs.append((heavy_idx, h_idx))
                    capacity[heavy_idx] -= 1
                    capacity[h_idx] -= 1
                    break
            remaining_electroneg = [
                i for i in remaining_electroneg if capacity.get(i, 0) > 0
            ]

    # Pair heavy byproducts when residual bonding capacity remains.
    remaining_heavy = [i for i in heavy_byproducts if capacity.get(i, 0) > 0]
    for k in range(0, len(remaining_heavy) - 1, 2):
        bond_pairs.append((remaining_heavy[k], remaining_heavy[k + 1]))
        capacity[remaining_heavy[k]] -= 1
        capacity[remaining_heavy[k + 1]] -= 1

    if not bond_pairs:
        return None

    for i, j in bond_pairs:
        r_complete[i, j] += 1
        r_complete[j, i] += 1

    # Update lone-pair changes after byproduct bond formation.
    participating = []
    for by_idx in byproduct_indices:
        new_bond_el = sum(2 for i, j in bond_pairs if i == by_idx or j == by_idx)
        r_complete[by_idx, by_idx] = broken_electrons.get(by_idx, 0) - new_bond_el
        if broken_electrons.get(by_idx, 0) > 0:
            participating.append(by_idx)

    # Distribute imbalance in electron-pair units to keep lone-pair counts even.
    imbalance = np.sum(r_complete)
    if abs(imbalance) > 1e-6:
        targets = participating if participating else list(byproduct_indices)
        if targets:
            total = int(round(-imbalance))
            units = total // 2  # number of lone pairs to distribute
            n = len(targets)
            base_units = units // n
            leftover_units = units - base_units * n
            for i, by_idx in enumerate(targets):
                extra = 1 if i < leftover_units else 0
                r_complete[by_idx, by_idx] += (base_units + extra) * 2

    for by_idx in byproduct_indices:
        result_lp = be_substrate[by_idx, by_idx] + r_complete[by_idx, by_idx]
        if result_lp < -0.01:
            return None

    return r_complete


def complete_r_matrix_simple(
    r_matrix: np.ndarray,
    be_substrate: np.ndarray,
    atoms_substrate: np.ndarray,
    byproduct_indices: List[int],
) -> np.ndarray:
    """
    Complete R matrix for byproduct atoms, enforcing electron conservation.

    Tries bond-forming completion first (byproduct H bonds to heavy atoms).
    Falls back to LP-only completion if bonds would create invalid LP values.

    Args:
        r_matrix: Partial R matrix (byproduct diagonal entries are 0)
        be_substrate: Substrate BE matrix
        atoms_substrate: Substrate atom identity
        byproduct_indices: Indices of byproduct atoms

    Returns:
        Completed R matrix with sum = 0
    """
    r_bonds = _complete_with_bonds(
        r_matrix, be_substrate, atoms_substrate, byproduct_indices
    )
    if r_bonds is not None:
        return r_bonds

    return _complete_lp_only(r_matrix, byproduct_indices)


def _compute_r_from_mapping(
    be_substrate: np.ndarray,
    be_product: np.ndarray,
    s2p: dict,
    byproduct_indices: List[int],
) -> np.ndarray:
    """Compute partial R matrix from a given substrate-to-product mapping."""
    n = len(be_substrate)
    r = np.zeros((n, n), dtype=np.float32)
    for si, pi in s2p.items():
        r[si, si] = be_product[pi, pi] - be_substrate[si, si]
        for sj, pj in s2p.items():
            if si < sj:
                diff = be_product[pi, pj] - be_substrate[si, sj]
                r[si, sj] = diff
                r[sj, si] = diff
        for by in byproduct_indices:
            if be_substrate[si, by] > 0:
                r[si, by] = -be_substrate[si, by]
                r[by, si] = -be_substrate[si, by]
    return r


def find_r_matrix(
    be_substrate: np.ndarray,
    atoms_substrate: np.ndarray,
    be_product: np.ndarray,
    atoms_product: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Find an R matrix such that applying it to substrate yields a superset of product.

    Tries multiple unmapped H assignments to find one that produces an LP-balanced
    R matrix (required for decomposition into elemental steps).
    """
    NO_LP = {1, 6}  # H, C don't have meaningful LP changes

    s2p_base, msub_base, mprod_base, h_candidates, h_targets = (
        _get_h_candidates_and_base_matching(
            atoms_substrate, atoms_product, be_substrate, be_product
        )
    )

    def _build(ordering):
        """Build a complete R matrix for a given hydrogen-source ordering."""
        s2p = s2p_base.copy()
        msub = msub_base.copy()
        mprod = mprod_base.copy()
        for sub_h, prod_h in zip(ordering, h_targets):
            s2p[sub_h] = prod_h
            msub.add(sub_h)
            mprod.add(prod_h)
        byproducts = [i for i in range(len(atoms_substrate)) if i not in msub]
        r_partial = _compute_r_from_mapping(be_substrate, be_product, s2p, byproducts)
        r_complete = complete_r_matrix_simple(
            r_partial, be_substrate, atoms_substrate, byproducts
        )
        be_result = be_substrate + r_complete
        valid = bool(np.all(be_result >= -1e-6) and abs(np.sum(r_complete)) < 1e-6)
        # LP imbalance on heteroatoms
        lp = 0.0
        for i in range(len(r_complete)):
            v = r_complete[i, i]
            if abs(v) > 1e-6 and int(atoms_substrate[i][1]) not in NO_LP:
                lp += v
        return r_complete, valid, abs(lp)

    default_order = h_candidates[: len(h_targets)]
    best_r, best_valid, best_lp = _build(default_order)

    if best_lp < 1e-6 or not h_targets or len(h_candidates) <= len(h_targets):
        return best_r, best_valid

    # If the default assignment is lone-pair imbalanced, test alternative
    # hydrogen assignments grouped by parent atom.
    n_tgt = len(h_targets)
    h_to_parent = {}
    parent_to_hs = {}
    for h in h_candidates:
        for j in range(len(be_substrate)):
            if j != h and be_substrate[h, j] > 0:
                h_to_parent[h] = j
                parent_to_hs.setdefault(j, []).append(h)
                break
    reduced = []
    for hs in parent_to_hs.values():
        reduced.extend(hs[:n_tgt])

    n_reduced = len(reduced)
    n_perms = 1
    for i in range(n_tgt):
        n_perms *= max(1, n_reduced - i)
    if n_perms > 5000:
        return best_r, best_valid

    seen_parent_combos = set()
    for ordering in permutations(reduced, n_tgt):
        # Deduplicate by parent combination while preserving target-assignment order.
        parent_key = tuple(h_to_parent.get(h) for h in ordering)
        if parent_key in seen_parent_combos:
            continue
        seen_parent_combos.add(parent_key)
        if list(ordering) == default_order:
            continue
        r, valid, lp = _build(ordering)
        if not valid:
            continue
        if lp < best_lp:
            best_r, best_valid, best_lp = r, valid, lp
            if lp < 1e-6:
                break

    return best_r, best_valid


def apply_r_matrix(
    be_substrate: np.ndarray, atoms_substrate: np.ndarray, r_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply R matrix to substrate to get result.

    Args:
        be_substrate: Substrate BE matrix
        atoms_substrate: Substrate atom identity (unchanged)
        r_matrix: Change matrix

    Returns:
        be_result: Result BE matrix
        atoms_result: Result atom identity (same as substrate)
    """
    be_result = be_substrate + r_matrix
    return be_result, atoms_substrate.copy()
