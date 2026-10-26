"""
Mechanism search over elemental steps.

Decomposes an R matrix into a sequence of valid elemental steps:

- ``check_decomposable``: fast preconditions (balance, integrality,
  divisibility by 4, LP balance).
- ``find_intermediate_bonds``: candidate transient bonds (formed then broken)
  for reactions whose change sum is not divisible by 4.
- ``find_minimum_mechanism``: BFS for a shortest step sequence whose
  intermediate BE matrices all stay valid (non-negative).
- ``find_all_minimum_mechanisms``: DFS enumeration of every mechanism with
  exactly the minimum step count.
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from mech_steps import (
    Step,
    apply_step,
    contribution,
    extract_changes,
    find_active_atoms,
    generate_elemental_steps,
    is_valid_be,
    is_zero,
    matrix_key,
    remove_step,
)


def add_intermediate_bond(r_matrix: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Return an unchanged copy of the R matrix.

    The function adds and then immediately subtracts the same bond entry at
    (i, j), so the net effect is a plain copy.  A transient bond cannot be
    represented inside a single R matrix (forming and breaking cancel);
    intermediate bonds are instead handled by the search routines via the
    ``intermediates`` parameter of ``find_minimum_mechanism`` and
    ``find_all_minimum_mechanisms``.

    Args:
        r_matrix: R matrix to copy.
        i, j: Bond indices (no net change is applied).

    Returns:
        Copy of ``r_matrix``.
    """
    r_new = r_matrix.copy()
    r_new[i, j] += 1
    r_new[j, i] += 1
    r_new[i, j] -= 1
    r_new[j, i] -= 1
    return r_new


def find_intermediate_bonds(
    r_matrix: np.ndarray,
    be_substrate: np.ndarray,
    atoms_substrate: np.ndarray,
    max_intermediates: int = 2,
    tol: float = 1e-6,
) -> List[List[Tuple[int, int]]]:
    """
    Find candidate intermediate bonds for reactions whose positive change sum
    is not divisible by 4.

    Each intermediate bond contributes +2 (formed) and -2 (broken) to the
    change sum, so a remainder of 2 (mod 4) is fixable with one intermediate
    (e.g. +6/-6 becomes +8/-8, a 2-step mechanism).  Other remainders are not
    handled and yield no candidates.  Candidate bonds connect active atoms
    and exclude bonds already present in the substrate or already changed in
    the R matrix.

    Args:
        r_matrix: R matrix (net changes).
        be_substrate: Substrate BE matrix.
        atoms_substrate: Substrate atom identity array (unused in candidate
            generation, kept for interface consistency).
        max_intermediates: Maximum number of intermediates allowed.
        tol: Numerical tolerance.

    Returns:
        List of intermediate bond sets to try: ``[[]]`` if no intermediates
        are needed, one single-bond list per candidate if one is needed, or
        ``[]`` if the remainder cannot be fixed.
    """
    sources, sinks = extract_changes(r_matrix)
    pos_sum = sum(contribution(c) for c in sinks)

    if abs(pos_sum % 4) < tol:
        return [[]]

    remainder = int(round(pos_sum)) % 4
    if remainder == 2:
        num_intermediates = 1
    else:
        return []

    if num_intermediates > max_intermediates:
        return []

    active = find_active_atoms(r_matrix, tol)

    candidates = []
    for i in active:
        for j in active:
            if i >= j:
                continue
            if be_substrate[i, j] > tol:
                continue
            if abs(r_matrix[i, j]) > tol:
                continue
            candidates.append((i, j))

    if num_intermediates == 1:
        return [[(i, j)] for i, j in candidates]

    return []


def check_decomposable(
    r_matrix: np.ndarray,
    tol: float = 1e-6,
    atoms: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    """
    Check if an R matrix can be decomposed into +4/-4 elemental steps.

    Conditions checked:

    1. Sources and sinks balance (total change sum is zero).
    2. All bond and lone pair changes are integral.
    3. The positive sum is divisible by 4 (each step moves 4 electrons).
    4. Total lone pair change is zero; an unbalanced LP indicates a missing
       electron source/sink in the reaction.

    Args:
        r_matrix: R matrix (net changes).
        tol: Numerical tolerance.
        atoms: Optional atom identity array forwarded to ``extract_changes``.

    Returns:
        (is_decomposable, message)
    """
    sources, sinks = extract_changes(r_matrix, atoms=atoms)

    pos_sum = sum(contribution(c) for c in sinks)
    neg_sum = sum(contribution(c) for c in sources)

    if abs(pos_sum + neg_sum) > tol:
        return False, f"R matrix not balanced: +{pos_sum:.1f}, {neg_sum:.1f}"

    for c in sources + sinks:
        i, j, v = c
        if i != j:
            if abs(v - round(v)) > tol:
                return False, f"Non-integer bond change: {v:.2f}"
        else:
            if abs(v - round(v)) > tol:
                return False, f"Non-integer lone pair change: {v:.2f}"

    if abs(pos_sum % 4) > tol:
        return False, f"Positive sum {pos_sum:.1f} not divisible by 4"

    lp_total = sum(c[2] for c in sources + sinks if c[0] == c[1])
    if abs(lp_total) > tol:
        return (
            False,
            f"LP not balanced (net {lp_total:+.0f}): missing electron source/sink",
        )

    expected_steps = int(round(pos_sum / 4))
    return True, f"Expecting {expected_steps} elemental steps"


def find_minimum_mechanism(
    be_start: np.ndarray,
    r_matrix: np.ndarray,
    max_steps: int = 50,
    intermediates: Optional[List[Tuple[int, int]]] = None,
    atoms: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[Step]], int]:
    """
    Find a mechanism with the minimum number of steps using BFS.

    BFS explores shorter paths first, so the first solution found is
    minimal.  Each intermediate bond carries a binary state (0 = not formed,
    1 = formed and pending breakage); an unformed intermediate is offered to
    the step generator as a formable bond, while breaking a formed one is
    handled through the R matrix update.  A solution requires the remaining
    R matrix to be zero and all intermediate states to be 0.  Visited states
    are keyed on (BE matrix, remaining R, intermediate states).

    Args:
        be_start: Starting BE matrix
        r_matrix: R matrix (net changes)
        max_steps: Maximum steps to search
        intermediates: Optional intermediate bonds that can be formed/broken
        atoms: Optional atom identity array for LP heteroatom constraint

    Returns:
        (steps, num_steps) or (None, -1) if not found
    """
    if is_zero(r_matrix):
        return [], 0

    initial_int_state = tuple([0] * len(intermediates)) if intermediates else ()

    queue = deque([(be_start.copy(), r_matrix.copy(), [], initial_int_state)])
    visited = {(matrix_key(be_start), matrix_key(r_matrix), initial_int_state)}

    while queue:
        be, r, steps, int_state = queue.popleft()

        if len(steps) >= max_steps:
            continue

        available_intermediates = []
        if intermediates:
            for idx, (i, j) in enumerate(intermediates):
                if int_state[idx] == 0:
                    available_intermediates.append((i, j))

        for step in generate_elemental_steps(
            r, available_intermediates if intermediates else None, atoms=atoms
        ):
            new_be = apply_step(be, step)

            if not is_valid_be(new_be):
                continue

            new_r = remove_step(r, step)
            new_steps = steps + [step]

            new_int_state = list(int_state)
            if intermediates:
                for idx, (i, j) in enumerate(intermediates):
                    for si, sj, sv in step:
                        if (si, sj) == (i, j) or (si, sj) == (j, i):
                            if sv > 0:
                                new_int_state[idx] = 1
                            elif sv < 0:
                                new_int_state[idx] = 0
            new_int_state = tuple(new_int_state)

            all_intermediates_resolved = all(s == 0 for s in new_int_state)
            if is_zero(new_r) and all_intermediates_resolved:
                return new_steps, len(new_steps)

            key = (matrix_key(new_be), matrix_key(new_r), new_int_state)
            if key not in visited:
                visited.add(key)
                queue.append((new_be.copy(), new_r.copy(), new_steps, new_int_state))

    return None, -1


def find_all_minimum_mechanisms(
    be_start: np.ndarray,
    r_matrix: np.ndarray,
    min_steps: int,
    max_results: int = 1000,
    atoms: Optional[np.ndarray] = None,
    intermediates: Optional[List[Tuple[int, int]]] = None,
) -> List[List[Step]]:
    """
    Find all mechanisms with exactly min_steps using DFS.

    Uses the same intermediate-state tracking as ``find_minimum_mechanism``.

    Args:
        be_start: Starting BE matrix
        r_matrix: Full R matrix to decompose
        min_steps: Target number of steps
        max_results: Stop after finding this many
        atoms: Optional atom identity array for LP heteroatom constraint
        intermediates: Optional intermediate bonds that can be formed/broken

    Returns:
        List of mechanisms (each is a list of steps)
    """
    results = []
    initial_int_state = tuple([0] * len(intermediates)) if intermediates else ()

    def dfs(
        be: np.ndarray,
        r: np.ndarray,
        steps: List[Step],
        int_state: tuple,
    ):
        """Recursively extend the current step sequence, recording complete mechanisms of length min_steps."""
        if len(results) >= max_results:
            return

        all_resolved = all(s == 0 for s in int_state) if int_state else True
        if is_zero(r) and all_resolved:
            if len(steps) == min_steps:
                results.append([list(s) for s in steps])
            return

        if len(steps) >= min_steps:
            return

        available = []
        if intermediates:
            for idx, (i, j) in enumerate(intermediates):
                if int_state[idx] == 0:
                    available.append((i, j))

        for step in generate_elemental_steps(
            r, available if intermediates else None, atoms=atoms
        ):
            new_be = apply_step(be, step)
            if not is_valid_be(new_be):
                continue

            new_r = remove_step(r, step)

            new_int_state = list(int_state)
            if intermediates:
                for idx, (i, j) in enumerate(intermediates):
                    for si, sj, sv in step:
                        if (si, sj) == (i, j) or (si, sj) == (j, i):
                            if sv > 0:
                                new_int_state[idx] = 1
                            elif sv < 0:
                                new_int_state[idx] = 0
            new_int_state = tuple(new_int_state)

            steps.append(step)
            dfs(new_be, new_r, steps, new_int_state)
            steps.pop()

    dfs(be_start, r_matrix, [], initial_int_state)
    return results
