"""
R matrix cleanup transforms.

Removes artifacts from a raw R matrix before mechanism search:

- Kekule shuffling noise in aromatic rings (bond changes that sum to zero
  around a ring correspond to an alternative Kekule structure, not a
  chemical change).
- Lone pair changes on byproduct atoms, converted into inter-byproduct bonds
  so mechanisms decompose into clean bond swaps.
- Radical byproducts (odd lone pair entries), paired up into bonds.
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem


def find_aromatic_rings(smiles: str) -> List[set]:
    """
    Find aromatic rings in a molecule.

    Args:
        smiles: SMILES string.

    Returns:
        List of sets of atom-map numbers, one set per fully aromatic ring
        (rings whose atoms carry no map numbers are omitted).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    rings = []
    ring_info = mol.GetRingInfo()

    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            map_nums = set()
            for idx in ring:
                atom = mol.GetAtomWithIdx(idx)
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    map_nums.add(map_num)
            if map_nums:
                rings.append(map_nums)

    return rings


def clean_aromatic_kekule_noise(
    r_matrix: np.ndarray,
    atoms: np.ndarray,
    aromatic_rings: List[set],
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Remove Kekule shuffling artifacts from an R matrix.

    For each aromatic ring, the off-diagonal bond changes between ring atoms
    are summed (each counted twice for the symmetric matrix).  If the sum is
    zero, the changes only represent a different Kekule structure and are
    zeroed out.

    Args:
        r_matrix: R matrix (net changes).
        atoms: (n, 2) atom identity array.
        aromatic_rings: List of sets of atom-map numbers per aromatic ring.
        tol: Numerical tolerance.

    Returns:
        Cleaned copy of the R matrix.
    """
    r_clean = r_matrix.copy()
    n = len(r_matrix)

    map_to_idx = {}
    for i, (map_num, _) in enumerate(atoms):
        if map_num > 0:
            map_to_idx[map_num] = i

    for ring_maps in aromatic_rings:
        ring_indices = [map_to_idx[m] for m in ring_maps if m in map_to_idx]

        if len(ring_indices) < 2:
            continue

        ring_change_sum = 0.0
        ring_changes = []

        for i in ring_indices:
            for j in ring_indices:
                if i < j:
                    val = r_matrix[i, j]
                    if abs(val) > tol:
                        ring_changes.append((i, j, val))
                        ring_change_sum += 2 * val

        if abs(ring_change_sum) < tol and ring_changes:
            for i, j, val in ring_changes:
                r_clean[i, j] = 0
                r_clean[j, i] = 0

    return r_clean


def eliminate_byproduct_lp(
    r_matrix: np.ndarray,
    atoms: np.ndarray,
    byproduct_idx: Optional[List[int]] = None,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Replace lone pair changes on byproduct atoms with inter-byproduct bonds.

    Byproduct atoms are unmapped atoms (map = 0) plus any mapped atoms listed
    in ``byproduct_idx`` (present in substrate but absent from product).
    LP changes on mapped non-byproduct atoms are left untouched — they
    represent genuine formal charge changes (e.g. N-oxide formation).

    Algorithm:

    1. Phase 0: eliminate LP changes on C and H atoms (always artifacts) by
       converting them into bonds with atoms whose row sum has the opposite
       sign (uncompensated bond changes).
    2. Compute row sums of byproduct atoms, splitting them into excess
       (positive row sum, LP to shed) and deficit (negative row sum, needing
       bond compensation).
    3. Phase 1: pair excess atoms with deficit atoms.  Each bond applies
       ``R[e,d] += 1`` and ``R[e,e] -= 2``, lowering the excess row sum by 1
       and raising the deficit row sum by 1.
    4. Phase 2: pair remaining excess atoms with each other.  Each bond
       applies ``R[a,b] += 1``, ``R[a,a] -= 2``, ``R[b,b] -= 2``, lowering
       both row sums by 1.
    5. Phase 3: zero out byproduct rows describing pure detachment (LP gain
       exactly compensating broken bonds).  Only applied when the atom's
       total electron contribution (diagonal plus twice the off-diagonal
       sum) is zero and every non-byproduct bond partner remains balanced
       after removal.

    Args:
        r_matrix: R matrix (net changes).
        atoms: (n, 2) atom identity array.
        byproduct_idx: Optional indices of mapped byproduct atoms.
        tol: Numerical tolerance.

    Returns:
        Cleaned copy of the R matrix.
    """
    NO_LP = {1, 6}
    byproduct_set = set(byproduct_idx) if byproduct_idx else set()

    def is_byproduct(i):
        """Return True if atom i is unmapped or listed as a mapped byproduct."""
        return int(atoms[i][0]) == 0 or i in byproduct_set

    r = r_matrix.copy()
    n = len(r)

    for i in range(n):
        z = int(atoms[i][1])
        if z not in NO_LP:
            continue
        if abs(r[i, i]) < tol:
            continue
        lp = r[i, i]
        if lp > tol:
            for j in range(n):
                if j == i:
                    continue
                row_sum_j = sum(r[j, k] for k in range(n))
                while row_sum_j < -tol and lp >= 2 - tol:
                    r[i, j] += 1
                    r[j, i] += 1
                    r[i, i] -= 2
                    lp -= 2
                    row_sum_j += 1
        elif lp < -tol:
            for j in range(n):
                if j == i:
                    continue
                row_sum_j = sum(r[j, k] for k in range(n))
                while row_sum_j > tol and lp <= -2 + tol:
                    r[i, j] += 1
                    r[j, i] += 1
                    r[i, i] += 2
                    lp += 2
                    row_sum_j -= 1

    excess = []
    deficit = []

    for i in range(n):
        if not is_byproduct(i):
            continue
        row_sum = sum(r[i, j] for j in range(n))
        if row_sum > tol:
            excess.append([i, row_sum])
        elif row_sum < -tol:
            deficit.append([i, -row_sum])

    for d_entry in deficit:
        di, d_need = d_entry
        for e_entry in excess:
            ei, e_have = e_entry
            while d_need >= 1 - tol and e_have >= 1 - tol and r[ei, ei] >= 2 - tol:
                r[ei, di] += 1
                r[di, ei] += 1
                r[ei, ei] -= 2
                e_have -= 1
                d_need -= 1
            e_entry[1] = e_have
            d_entry[1] = d_need
            if d_need < 1 - tol:
                break

    for i in range(len(excess)):
        ai, a_have = excess[i]
        if a_have < 1 - tol:
            continue
        for j in range(i + 1, len(excess)):
            bj, b_have = excess[j]
            if b_have < 1 - tol:
                continue
            while (
                a_have >= 1 - tol
                and b_have >= 1 - tol
                and r[ai, ai] >= 2 - tol
                and r[bj, bj] >= 2 - tol
            ):
                r[ai, bj] += 1
                r[bj, ai] += 1
                r[ai, ai] -= 2
                r[bj, bj] -= 2
                a_have -= 1
                b_have -= 1
            excess[i][1] = a_have
            excess[j][1] = b_have

    for i in range(n):
        if not is_byproduct(i):
            continue
        off_diag_sum = sum(r[i, j] for j in range(n) if j != i)
        total = r[i, i] + 2 * off_diag_sum
        if abs(total) > tol:
            continue
        if abs(r[i, i]) < tol and abs(off_diag_sum) < tol:
            continue
        safe = True
        for j in range(n):
            if j == i or is_byproduct(j) or abs(r[i, j]) < tol:
                continue
            j_row_sum_without_i = sum(r[j, k] for k in range(n) if k != i)
            if abs(j_row_sum_without_i) > tol:
                safe = False
                break
        if safe:
            r[i, :] = 0
            r[:, i] = 0

    return r


def find_radicals(r_matrix: np.ndarray, tol: float = 1e-6) -> List[int]:
    """
    Return indices of atoms with odd diagonal (lone pair) entries.

    An odd LP change indicates an unpaired electron (radical).

    Args:
        r_matrix: R matrix (net changes).
        tol: Numerical tolerance.

    Returns:
        List of radical atom indices.
    """
    radicals = []
    n = len(r_matrix)
    for i in range(n):
        lp = r_matrix[i, i]
        if abs(lp) > tol and abs(lp % 2) > tol:
            radicals.append(i)
    return radicals


def generate_pairings(items: List[int]) -> List[List[Tuple[int, int]]]:
    """
    Generate all ways to pair up items.

    For [a,b,c,d] returns:
        [[(a,b), (c,d)], [(a,c), (b,d)], [(a,d), (b,c)]]

    Args:
        items: Items to pair (even count expected).

    Returns:
        List of complete pairings, each a list of (item, item) tuples.
    """
    if len(items) == 0:
        return [[]]
    if len(items) == 2:
        return [[(items[0], items[1])]]

    first = items[0]
    rest = items[1:]
    result = []

    for i, partner in enumerate(rest):
        pair = (first, partner)
        remaining = rest[:i] + rest[i + 1 :]
        for sub_pairing in generate_pairings(remaining):
            result.append([pair] + sub_pairing)

    return result


def fix_radicals(r_matrix: np.ndarray, pairing: List[Tuple[int, int]]) -> np.ndarray:
    """
    Fix radical pairs by forming bonds between them.

    For each pair (i, j): add a bond (``R[i,j] += 1``) and reduce both lone
    pair entries (``R[i,i] -= 1``, ``R[j,j] -= 1``).

    Args:
        r_matrix: R matrix (net changes).
        pairing: List of radical index pairs.

    Returns:
        Fixed copy of the R matrix.
    """
    r_fixed = r_matrix.copy()
    for i, j in pairing:
        r_fixed[i, j] += 1
        r_fixed[j, i] += 1
        r_fixed[i, i] -= 1
        r_fixed[j, j] -= 1
    return r_fixed
