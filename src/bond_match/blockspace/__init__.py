"""
Blockspace package API.

The blockspace modules use bare imports (``from be_core import ...``) so
they can also run standalone (e.g. ``python mechanism.py``).  This
``__init__.py`` adds the package directory to ``sys.path`` so those bare
imports resolve, then re-exports the key public API used by the parent
project.

Module layout:

- ``be_core``: SMILES/mol <-> BE matrix conversion
- ``be_components``: connected-component decomposition
- ``be_hash``: WL hashing, system signatures, subset check
- ``atom_matching``: substrate <-> product atom matching
- ``r_matrix``: R matrix computation and completion
- ``map_assignment``: atom-map number assignment
- ``atom_labels``: atom/step label formatting
- ``r_cleanup``: R matrix cleanup transforms
- ``mech_steps``: elemental step primitives
- ``mech_search``: BFS/DFS mechanism search
- ``mech_serialize``: mechanism JSON serialization
- ``mech_analyze``: end-to-end analysis and CLI
- ``verify``: serialized-mechanism verification
"""

import os
import sys

__all__ = [
    "PT",
    "matrices_to_smiles",
    "mol_to_be_matrix",
    "smiles_to_matrices",
    "decompose_molecules",
    "apply_step",
    "is_valid_be",
    "parse_atom_label",
    "reconstruct_step",
    "resolve_atom_index",
]

_submodule_dir = os.path.dirname(os.path.abspath(__file__))
if _submodule_dir not in sys.path:
    sys.path.insert(0, _submodule_dir)

from be_core import (  # noqa: E402
    PT,
    matrices_to_smiles,
    mol_to_be_matrix,
    smiles_to_matrices,
)

from be_components import decompose_molecules  # noqa: E402

from mech_steps import apply_step, is_valid_be  # noqa: E402

from verify import (  # noqa: E402
    parse_atom_label,
    reconstruct_step,
    resolve_atom_index,
)
