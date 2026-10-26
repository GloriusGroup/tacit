"""
Connected-component utilities for BE matrices.

A BE matrix may describe several disconnected molecules at once (a "system").
This module finds those components, extracts per-molecule submatrices, and
provides helpers to permute or subsample a multi-molecule system.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def get_connected_components(be_matrix: np.ndarray) -> List[List[int]]:
    """
    Find connected components (individual molecules) in a BE matrix.

    Adjacency is defined by positive off-diagonal entries; diagonal (lone
    pair) entries are ignored.  Uses ``scipy.sparse.csgraph``.

    Args:
        be_matrix: (n_atoms, n_atoms) bond-electron matrix

    Returns:
        List of lists, each containing atom indices for one component
    """
    n_atoms = len(be_matrix)

    adjacency = (be_matrix > 0).astype(np.int8)
    np.fill_diagonal(adjacency, 0)

    sparse_adj = csr_matrix(adjacency)
    n_components, labels = connected_components(sparse_adj, directed=False)

    components = [[] for _ in range(n_components)]
    for idx, label in enumerate(labels):
        components[label].append(idx)

    return components


def extract_component(
    be_matrix: np.ndarray, atom_identity: np.ndarray, indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a submatrix for a single connected component.

    Args:
        be_matrix: Full BE matrix
        atom_identity: Full atom identity matrix
        indices: Atom indices belonging to this component

    Returns:
        (sub_be_matrix, sub_atom_identity) for the selected component
    """
    indices = sorted(indices)
    n = len(indices)

    sub_be = np.zeros((n, n), dtype=be_matrix.dtype)
    sub_atoms = np.zeros((n, 2), dtype=atom_identity.dtype)

    for new_i, old_i in enumerate(indices):
        sub_atoms[new_i] = atom_identity[old_i]
        for new_j, old_j in enumerate(indices):
            sub_be[new_i, new_j] = be_matrix[old_i, old_j]

    return sub_be, sub_atoms


def decompose_molecules(
    be_matrix: np.ndarray, atom_identity: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Decompose a multi-molecule system into individual molecules.

    Args:
        be_matrix: BE matrix (possibly block-diagonal or scrambled)
        atom_identity: Atom identity matrix

    Returns:
        List of (be_matrix, atom_identity) tuples, one per molecule
    """
    components = get_connected_components(be_matrix)
    molecules = []

    for component_indices in components:
        sub_be, sub_atoms = extract_component(
            be_matrix, atom_identity, component_indices
        )
        molecules.append((sub_be, sub_atoms))

    return molecules


def scramble_matrix_pair(
    be_matrix: np.ndarray, atom_identity: np.ndarray, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly permute the node ordering of a matrix pair.

    The resulting matrices represent the same molecular system with a different
    atom order. The BE matrix is permuted symmetrically across rows and columns.

    Args:
        be_matrix: Original BE matrix
        atom_identity: Original atom identity matrix
        seed: Random seed for reproducibility

    Returns:
        (scrambled_be, scrambled_atoms)
    """
    if seed is not None:
        np.random.seed(seed)

    n_atoms = len(be_matrix)
    perm = np.random.permutation(n_atoms)

    scrambled_atoms = atom_identity[perm]

    scrambled_be = be_matrix[perm][:, perm]

    return scrambled_be, scrambled_atoms


def remove_random_components(
    be_matrix: np.ndarray,
    atom_identity: np.ndarray,
    n_keep: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove random molecules from a multi-molecule system.

    Kept molecules are recombined into block-diagonal matrices.

    Args:
        be_matrix: Original BE matrix
        atom_identity: Original atom identity matrix
        n_keep: Number of molecules to keep (None = random)
        seed: Random seed

    Returns:
        (reduced_be, reduced_atoms) with fewer molecules
    """
    if seed is not None:
        np.random.seed(seed)

    molecules = decompose_molecules(be_matrix, atom_identity)
    n_molecules = len(molecules)

    if n_molecules <= 1:
        return be_matrix.copy(), atom_identity.copy()

    if n_keep is None:
        n_keep = np.random.randint(1, n_molecules)

    n_keep = min(n_keep, n_molecules)

    keep_indices = np.random.choice(n_molecules, size=n_keep, replace=False)
    kept_molecules = [molecules[i] for i in keep_indices]

    total_atoms = sum(len(atoms) for be, atoms in kept_molecules)

    new_be = np.zeros((total_atoms, total_atoms), dtype=be_matrix.dtype)
    new_atoms = np.zeros((total_atoms, 2), dtype=atom_identity.dtype)

    offset = 0
    for be, atoms in kept_molecules:
        n = len(atoms)
        new_be[offset : offset + n, offset : offset + n] = be
        new_atoms[offset : offset + n] = atoms
        offset += n

    return new_be, new_atoms
