"""
Elemental step primitives.

An elemental step moves two electron pairs (4 electrons) and is expressed as
a list of unit changes ``(i, j, value)`` on the upper triangle of an
R matrix.  Exactly two step shapes are allowed:

- Type 1 (bond swap): break 2 bonds, form 2 bonds (4 off-diagonal changes).
- Type 2 (bond-LP swap): break 1 bond, form 1 bond, shift 1 lone pair
  (2 off-diagonal + 2 diagonal changes).

Every step has sources summing to -4 and sinks summing to +4 (net 0,
electron conservation).  This module extracts unit changes from an R matrix,
enumerates candidate steps, validates their electron flow, and applies or
removes steps from BE/R matrices.

Type aliases:

- ``Change``: ``Tuple[int, int, float]`` — an ``(i, j, value)`` entry on the
  upper triangle (``i == j`` for lone pair changes).
- ``Step``: ``List[Change]`` — the changes of one elemental step.
"""

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

Change = Tuple[int, int, float]
Step = List[Change]


def contribution(change: Change) -> float:
    """
    Calculate the contribution of a change to the full matrix sum.

    Off-diagonal entries are counted twice (symmetric matrix).
    Diagonal entries are counted once.

    Args:
        change: (i, j, value) change tuple.

    Returns:
        Contribution to the full matrix sum.
    """
    i, j, val = change
    return val if i == j else 2 * val


def extract_changes(
    r_matrix: np.ndarray,
    tol: float = 1e-6,
    atoms: Optional[np.ndarray] = None,
) -> Tuple[List[Change], List[Change]]:
    """
    Extract unit changes from an R matrix (upper triangle + diagonal only).

    Multi-electron changes are split into unit changes:
    - Bonds (off-diagonal): split into ±1 units
    - Lone pairs (diagonal): split into ±2 units

    If atoms is provided, LP changes on C (Z=6) and H (Z=1) are skipped
    (only heteroatoms should have LP changes in elemental steps).

    Args:
        r_matrix: R matrix (net changes).
        tol: Numerical tolerance.
        atoms: Optional (n, 2) atom identity array.

    Returns:
        (sources, sinks) - sources are negative, sinks are positive
    """
    NO_LP = {1, 6}

    sources, sinks = [], []

    rows, cols = np.nonzero(np.triu(np.abs(r_matrix) >= tol))
    for i, j in zip(rows.tolist(), cols.tolist()):
        val = float(r_matrix[i, j])
        if i == j:
            if atoms is not None and int(atoms[i][1]) in NO_LP:
                continue
            unit = 2.0 if val > 0 else -2.0
            count = int(round(abs(val) / 2))
        else:
            unit = 1.0 if val > 0 else -1.0
            count = int(round(abs(val)))

        for _ in range(count):
            if unit < 0:
                sources.append((i, j, unit))
            else:
                sinks.append((i, j, unit))

    return sources, sinks


def find_subsets_with_sum(
    changes: List[Change], target: float, tol: float = 1e-6
) -> List[List[Change]]:
    """
    Find all subsets of changes whose contributions sum to target.

    Uses brute-force enumeration over all non-empty subsets, which is
    adequate for small change lists.

    Args:
        changes: Candidate changes.
        target: Required contribution sum.
        tol: Numerical tolerance.

    Returns:
        List of matching subsets.
    """
    results = []
    n = len(changes)

    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            subset = [changes[k] for k in combo]
            total = sum(contribution(c) for c in subset)
            if abs(total - target) < tol:
                results.append(subset)

    return results


def is_valid_electron_flow(step: Step, atoms: Optional[np.ndarray] = None) -> bool:
    """
    Check if a step is one of exactly two allowed elemental step types.

    Type 1 (bond swap):
        4 off-diagonal changes: 2x +1 and 2x -1. No diagonal changes.
        The two broken bonds must involve 4 distinct atoms (rejects
        duplicate-bond and 3-center cases), and the formed bonds must
        cross-pair the same 4 atoms: breaking (A,B) and (C,D) allows forming
        {(A,C),(B,D)} or {(A,D),(B,C)}.

    Type 2 (bond-LP swap):
        2 off-diagonal changes: 1x +1 and 1x -1.
        2 diagonal changes: 1x +2 and 1x -2.
        For broken bond (A,B) and LP-loss atom C (with C not in {A,B}), the
        formed bond must connect C to one endpoint of the broken bond and the
        LP gain must land on the other endpoint.

    Additional constraints:
    - No LP changes on C (Z=6) or H (Z=1)
    - All changes must form a connected graph (checked by BFS over bond
      edges; diagonal entries contribute their atom as a node)

    Args:
        step: List of change tuples.
        atoms: Optional (n, 2) atom identity array for the LP element check.

    Returns:
        True if the step is a valid elemental step.
    """
    NO_LP = {1, 6}

    bond_pos = []
    bond_neg = []
    lp_pos = []
    lp_neg = []
    atoms_in_step = set()
    edges = []

    for i, j, val in step:
        v = float(val)
        if i == j:
            if abs(v - 2.0) < 1e-6:
                lp_pos.append((i, j, v))
            elif abs(v + 2.0) < 1e-6:
                lp_neg.append((i, j, v))
            else:
                return False
            if atoms is not None and int(atoms[i][1]) in NO_LP:
                return False
            atoms_in_step.add(i)
        else:
            if abs(v - 1.0) < 1e-6:
                bond_pos.append((i, j, v))
            elif abs(v + 1.0) < 1e-6:
                bond_neg.append((i, j, v))
            else:
                return False
            atoms_in_step.add(i)
            atoms_in_step.add(j)
            edges.append((i, j))

    if not atoms_in_step:
        return True

    n_bp = len(bond_pos)
    n_bn = len(bond_neg)
    n_lp = len(lp_pos)
    n_ln = len(lp_neg)

    is_type1 = n_bp == 2 and n_bn == 2 and n_lp == 0 and n_ln == 0
    is_type2 = n_bp == 1 and n_bn == 1 and n_lp == 1 and n_ln == 1

    if not (is_type1 or is_type2):
        return False

    if is_type1:
        A, B = bond_neg[0][0], bond_neg[0][1]
        C, D = bond_neg[1][0], bond_neg[1][1]
        if len({A, B, C, D}) != 4:
            return False
        f0 = {bond_pos[0][0], bond_pos[0][1]}
        f1 = {bond_pos[1][0], bond_pos[1][1]}
        cross_a = ({A, C}, {B, D})
        cross_b = ({A, D}, {B, C})
        if not (
            (f0 == cross_a[0] and f1 == cross_a[1])
            or (f0 == cross_a[1] and f1 == cross_a[0])
            or (f0 == cross_b[0] and f1 == cross_b[1])
            or (f0 == cross_b[1] and f1 == cross_b[0])
        ):
            return False

    if is_type2:
        A, B = bond_neg[0][0], bond_neg[0][1]
        C = lp_neg[0][0]
        D, E = bond_pos[0][0], bond_pos[0][1]
        F = lp_pos[0][0]
        if C in {A, B}:
            return False
        form_set = {D, E}
        if form_set == {C, A}:
            if F != B:
                return False
        elif form_set == {C, B}:
            if F != A:
                return False
        else:
            return False

    adj = {a: set() for a in atoms_in_step}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    start = next(iter(atoms_in_step))
    visited = {start}
    queue = [start]
    while queue:
        node = queue.pop()
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited == atoms_in_step


def generate_elemental_steps(
    r_matrix: np.ndarray,
    intermediates: Optional[List[Tuple[int, int]]] = None,
    atoms: Optional[np.ndarray] = None,
) -> List[Step]:
    """
    Generate all valid elemental steps from the current R matrix.

    Combines every source subset summing to -4 with every sink subset
    summing to +4 and keeps combinations that pass
    ``is_valid_electron_flow``; this enforces the two allowed step types
    (bond swap and bond-LP swap).  Intermediate bonds are offered as both
    breakable sources (-1) and formable sinks (+1).

    Args:
        r_matrix: Current R matrix (remaining changes)
        intermediates: Optional list of intermediate bonds (i, j) that can
                      be formed (+1) or broken (-1) in steps
        atoms: Optional atom identity array for LP heteroatom constraint

    Returns:
        List of valid steps (deduplicated).
    """
    sources, sinks = extract_changes(r_matrix, atoms=atoms)

    if intermediates:
        for i, j in intermediates:
            sources.append((i, j, -1.0))
            sinks.append((i, j, 1.0))

    if not sources or not sinks:
        return []

    source_4 = find_subsets_with_sum(sources, -4.0)
    sink_4 = find_subsets_with_sum(sinks, +4.0)

    steps = []
    seen = set()

    for src in source_4:
        for snk in sink_4:
            step = src + snk
            key = frozenset(step)
            if key not in seen:
                seen.add(key)
                if is_valid_electron_flow(step, atoms):
                    steps.append(step)

    return steps


def apply_step(be: np.ndarray, step: Step) -> np.ndarray:
    """Apply an elemental step to a BE matrix, returning a new matrix."""
    result = be.copy()
    for i, j, val in step:
        result[i, j] += val
        if i != j:
            result[j, i] += val
    return result


def remove_step(r: np.ndarray, step: Step) -> np.ndarray:
    """Subtract a step's changes from an R matrix, returning a new matrix."""
    result = r.copy()
    for i, j, val in step:
        result[i, j] -= val
        if i != j:
            result[j, i] -= val
    return result


def is_valid_be(be: np.ndarray, tol: float = -1e-6) -> bool:
    """Check if a BE matrix has all non-negative entries (within tolerance)."""
    return np.all(be >= tol)


def is_zero(r: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if an R matrix is effectively zero."""
    return np.allclose(r, 0, atol=tol)


def matrix_key(m: np.ndarray) -> Tuple:
    """Convert a matrix to a hashable key (flattened, rounded to 6 decimals)."""
    return tuple(np.round(m.flatten(), 6))


def find_active_atoms(r_matrix: np.ndarray, tol: float = 1e-6) -> List[int]:
    """Return indices of atoms involved in any R matrix change."""
    active = set()
    n = len(r_matrix)
    for i in range(n):
        for j in range(n):
            if abs(r_matrix[i, j]) > tol:
                active.add(i)
                active.add(j)
    return list(active)
