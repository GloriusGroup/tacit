"""
BE matrix core: SMILES/RDKit mol <-> bond-electron matrix conversion.

Bond-Electron (BE) matrix conventions (following FlowER):

- Off-diagonal ``BE[i, j]``: half the bond electrons
  (single = 1, double = 2, triple = 3, aromatic = 1.5).
- Diagonal ``BE[i, i]``: lone pair electrons = v - c - b - h
  (outer electrons - formal charge - bond order sum - implicit H count).
- Atom identity matrix: ``(n_atoms, 2)`` int array with
  ``[map_number, atomic_number]`` per atom.

These conventions ensure ``sum(BE matrix) == total valence electrons``
(electron conservation).

``BOND_TO_ELECTRONS`` maps RDKit bond types to total bond electron counts.
"""

from typing import Tuple

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PT = Chem.GetPeriodicTable()

BOND_TO_ELECTRONS = {
    Chem.BondType.SINGLE: 2,
    Chem.BondType.DOUBLE: 4,
    Chem.BondType.TRIPLE: 6,
    Chem.BondType.AROMATIC: 3,
}


def smiles_to_matrices(
    smiles: str, kekulize: bool = True, include_hydrogens: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SMILES to (BE matrix, atom identity matrix).

    Half the bond electrons are stored in each of ``BE[i, j]`` and
    ``BE[j, i]`` (FlowER convention); diagonal entries hold lone pair
    electrons computed as v - c - b - h.

    Args:
        smiles: SMILES string (can contain multiple molecules separated by '.')
        kekulize: If True, convert aromatic bonds to alternating single/double
        include_hydrogens: If True, include explicit H atoms in matrix.
                          If False, only heavy atoms (H count implicit in lone pairs).

    Returns:
        be_matrix: (n_atoms, n_atoms) float array
        atom_identity: (n_atoms, 2) int array with [map_number, atomic_number]

    Raises:
        ValueError: If the SMILES cannot be parsed.
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = True

    mol = Chem.MolFromSmiles(smiles, ps)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    if include_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)

    n_atoms = mol.GetNumAtoms()

    be_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    atom_identity = np.zeros((n_atoms, 2), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        map_num = atom.GetAtomMapNum()
        atomic_num = atom.GetAtomicNum()
        atom_identity[idx] = [map_num, atomic_num]

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = bond.GetBondType()
        electrons = BOND_TO_ELECTRONS.get(bond_type, 2)

        be_matrix[i, j] = electrons / 2
        be_matrix[j, i] = electrons / 2

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()

        v = PT.GetNOuterElecs(atomic_num)
        c = atom.GetFormalCharge()
        b = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        h = atom.GetTotalNumHs() if not include_hydrogens else 0

        lone_pairs = v - c - b - h
        be_matrix[idx, idx] = lone_pairs

    return be_matrix, atom_identity


def matrices_to_smiles(be_matrix: np.ndarray, atom_identity: np.ndarray) -> str:
    """
    Convert (BE matrix, atom identity matrix) back to SMILES.

    Bond orders are read from one side of the symmetric matrix and binned
    (half-electron value <= 1.25 single, <= 2.5 double, else triple).
    Formal charges are recovered from the diagonal via
    ``c = v - b - lone_pairs``.  Isolated atoms are marked no-implicit-H so
    sanitization does not add hydrogens.  If sanitization fails, a
    non-canonical SMILES is returned without sanitization.

    Args:
        be_matrix: (n_atoms, n_atoms) bond-electron matrix
        atom_identity: (n_atoms, 2) with [map_number, atomic_number]

    Returns:
        SMILES string
    """
    n_atoms = len(be_matrix)

    mol = Chem.RWMol()

    for i in range(n_atoms):
        map_num, atomic_num = atom_identity[i]
        atom = Chem.Atom(int(atomic_num))
        if map_num > 0:
            atom.SetAtomMapNum(int(map_num))
        mol.AddAtom(atom)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            half_electrons = be_matrix[i, j]
            if half_electrons >= 0.5:
                if half_electrons <= 1.25:
                    bond_type = Chem.BondType.SINGLE
                elif half_electrons <= 2.5:
                    bond_type = Chem.BondType.DOUBLE
                else:
                    bond_type = Chem.BondType.TRIPLE
                mol.AddBond(i, j, bond_type)

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()
        v = PT.GetNOuterElecs(atomic_num)
        b = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        lone_pairs = be_matrix[i, i]

        formal_charge = int(v - b - lone_pairs)
        atom.SetFormalCharge(formal_charge)

        if atom.GetDegree() == 0 and atom.GetNumExplicitHs() == 0:
            atom.SetNoImplicit(True)

    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except:
        return Chem.MolToSmiles(mol, canonical=False)


def mol_to_be_matrix(mol):
    """
    Build BE matrix directly from an RDKit mol, preserving atom indices.

    Indices match ``mol.GetAtomWithIdx(i)`` and thus the graph node indices
    from ``mol_to_nx``.  Uses kekulized bonds for integer bond orders
    required by ``apply_step``.

    Args:
        mol: RDKit molecule with explicit hydrogens, typically from ``AddHs``

    Returns:
        be_matrix: (n, n) float32 array
        atom_identity: (n, 2) int32 array with [map_number, atomic_number]
    """
    mol_copy = Chem.RWMol(mol)
    try:
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
    except Exception:
        pass

    n = mol_copy.GetNumAtoms()
    be = np.zeros((n, n), dtype=np.float32)
    atoms = np.zeros((n, 2), dtype=np.int32)

    for atom in mol_copy.GetAtoms():
        idx = atom.GetIdx()
        atoms[idx] = [atom.GetAtomMapNum(), atom.GetAtomicNum()]

    for bond in mol_copy.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bo = bond.GetBondTypeAsDouble()
        be[i, j] = bo
        be[j, i] = bo

    for atom in mol_copy.GetAtoms():
        idx = atom.GetIdx()
        v = PT.GetNOuterElecs(atom.GetAtomicNum())
        c = atom.GetFormalCharge()
        b = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        be[idx, idx] = v - c - b

    return be, atoms
