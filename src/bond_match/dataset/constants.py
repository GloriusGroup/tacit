"""Shared constants and RDKit configuration for the dataset package.

Importing this module disables RDKit logging, aliases ``np.long`` to
``np.int64``, and replaces ``Chem.MolToSmiles`` with
:func:`custom_mol_to_smiles` so dative bonds are written by default.

Constants:
    BASE_DIR: Directory containing this module; cache files are stored
        relative to it.
    ATOM_MAX_NUM: Maximum supported atomic number for feature encoding.
    BOND_TYPE_MAP: RDKit bond type to integer bond category (dative bonds
        map to the single-bond category).
    BOND_ORDER_TO_TYPE: Numeric bond order to edge-type index. 0.0 denotes
        the virtual head-node edge, 1.5 aromatic, 5.0 dative, and 7.0 the
        lone-pair (LP) edge to a virtual lone-pair node.
    NUM_BOND_TYPES: Number of distinct edge-type indices.
    _ELEMENT_TO_ATOMIC_NUM: Lazily populated element-symbol to atomic-number
        cache.
    PT: RDKit periodic table singleton.
    SHARD_SAMPLE_LIMIT: Default number of samples per dataset shard.
"""

import logging
import os
import numpy as np
from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")
_logger = logging.getLogger(__name__)
_original_mol_to_smiles = Chem.MolToSmiles
def custom_mol_to_smiles(mol, **kwargs):
    """Write a SMILES string with dative bonds included by default.

    Wraps the original ``Chem.MolToSmiles``. If a ``params`` object is
    supplied it is used unchanged; otherwise a ``SmilesWriteParams`` with
    ``includeDativeBonds=True`` is built, recognized keyword arguments
    (including the ``isomericSmiles``/``kekuleSmiles`` aliases) are
    transferred onto it, and the remaining keyword arguments are forwarded.

    Args:
        mol: RDKit Mol to serialize.
        **kwargs: Keyword arguments accepted by ``Chem.MolToSmiles``.

    Returns:
        SMILES string for ``mol``.
    """
    if "params" in kwargs:
        return _original_mol_to_smiles(mol, **kwargs)

    params = Chem.SmilesWriteParams()
    params.includeDativeBonds = True
    arg_map = {
        "isomericSmiles": "doIsomericSmiles",
        "kekuleSmiles": "doKekule",
    }

    keys_to_remove = []

    for key, value in kwargs.items():
        attr_name = arg_map.get(key, key)
        if hasattr(params, attr_name):
            setattr(params, attr_name, value)
            keys_to_remove.append(key)

    for key in keys_to_remove:
        kwargs.pop(key)
    kwargs["params"] = params

    return _original_mol_to_smiles(mol, **kwargs)
Chem.MolToSmiles = custom_mol_to_smiles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
np.long = np.int64
ATOM_MAX_NUM = 100
BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DATIVE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
BOND_ORDER_TO_TYPE = {
    0.0: 0,
    1.0: 1,
    2.0: 2,
    3.0: 3,
    1.5: 4,
    5.0: 5,
    7.0: 6,
}
NUM_BOND_TYPES = 7
_ELEMENT_TO_ATOMIC_NUM = {}
PT = Chem.GetPeriodicTable()
SHARD_SAMPLE_LIMIT = 50_000
