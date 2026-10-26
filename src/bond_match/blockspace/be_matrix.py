"""
BE-matrix public API.

Modules:

- ``be_core``: SMILES/mol <-> BE matrix conversion
- ``be_components``: connected-component decomposition
- ``be_hash``: WL hashing, system signatures, subset check
- ``atom_matching``: substrate <-> product atom matching
- ``r_matrix``: R matrix computation and completion
"""

from be_core import (  # noqa: F401
    BOND_TO_ELECTRONS,
    PT,
    matrices_to_smiles,
    mol_to_be_matrix,
    smiles_to_matrices,
)

from be_components import (  # noqa: F401
    decompose_molecules,
    extract_component,
    get_connected_components,
    remove_random_components,
    scramble_matrix_pair,
)

from be_hash import (  # noqa: F401
    be_to_networkx,
    is_subset,
    molecule_hash,
    system_signature,
    wl_hash,
)

from atom_matching import (  # noqa: F401
    _get_h_candidates_and_base_matching,
    match_atoms_by_map,
)

from r_matrix import (  # noqa: F401
    _complete_lp_only,
    _complete_with_bonds,
    _compute_r_from_mapping,
    apply_r_matrix,
    complete_r_matrix_simple,
    compute_r_matrix_partial,
    find_r_matrix,
)
