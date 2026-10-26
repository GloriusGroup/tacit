"""
Permutation-invariant hashing and multiset comparison for BE matrices.

Provides Weisfeiler-Lehman graph hashes for single molecules, multiset
signatures for multi-molecule systems, and a subset check used to verify
that a product system is contained in a substrate-plus-R result.
"""

from collections import Counter

import networkx as nx
import numpy as np

from be_components import decompose_molecules


def be_to_networkx(be_matrix: np.ndarray, atom_identity: np.ndarray) -> nx.Graph:
    """
    Convert BE matrix + atom identity to a NetworkX graph.

    Each node carries a ``label`` attribute encoding atomic number, map
    number, and lone pair count; each edge carries a ``label`` attribute
    encoding the bond electron entry.

    Args:
        be_matrix: BE matrix for molecule
        atom_identity: Atom identity matrix

    Returns:
        NetworkX Graph
    """
    n_atoms = len(be_matrix)
    G = nx.Graph()

    for i in range(n_atoms):
        map_num, atomic_num = atom_identity[i]
        lone_pairs = be_matrix[i, i]
        G.add_node(i, label=f"{atomic_num}:{map_num}:{lone_pairs:.1f}")

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if be_matrix[i, j] > 0:
                G.add_edge(i, j, label=f"{be_matrix[i, j]:.1f}")

    return G


def wl_hash(
    be_matrix: np.ndarray, atom_identity: np.ndarray, iterations: int = 3
) -> str:
    """
    Compute Weisfeiler-Lehman hash for a molecule.

    Uses NetworkX's C-optimized implementation.

    This produces a permutation-invariant hash that incorporates:
    - Atomic numbers
    - Map numbers
    - Bond structure (from BE matrix)
    - Lone pairs

    Args:
        be_matrix: BE matrix for single molecule
        atom_identity: Atom identity matrix
        iterations: Number of WL iterations

    Returns:
        Hex string hash
    """
    G = be_to_networkx(be_matrix, atom_identity)
    return nx.weisfeiler_lehman_graph_hash(
        G, iterations=iterations, node_attr="label", edge_attr="label"
    )


def molecule_hash(be_matrix: np.ndarray, atom_identity: np.ndarray) -> str:
    """
    Get a canonical hash for a single molecule.

    Alias for ``wl_hash`` with default parameters.
    """
    return wl_hash(be_matrix, atom_identity)


def system_signature(be_matrix: np.ndarray, atom_identity: np.ndarray) -> Counter:
    """
    Get a multiset signature for a molecular system.

    Decomposes into molecules and hashes each one.

    Args:
        be_matrix: BE matrix (possibly multi-molecule)
        atom_identity: Atom identity matrix

    Returns:
        Counter of molecule hashes
    """
    molecules = decompose_molecules(be_matrix, atom_identity)
    hashes = [molecule_hash(be, atoms) for be, atoms in molecules]
    return Counter(hashes)


def is_subset(
    be_small: np.ndarray,
    atoms_small: np.ndarray,
    be_big: np.ndarray,
    atoms_big: np.ndarray,
) -> bool:
    """
    Check if the small system is a multiset subset of the big system.

    Handles scrambled node ordering and non-block-diagonal matrices.

    Args:
        be_small, atoms_small: The potentially smaller system
        be_big, atoms_big: The potentially larger system

    Returns:
        True if all molecules in small are present in big (with multiplicity)
    """
    sig_small = system_signature(be_small, atoms_small)
    sig_big = system_signature(be_big, atoms_big)

    for mol_hash, count in sig_small.items():
        if sig_big.get(mol_hash, 0) < count:
            return False

    return True
