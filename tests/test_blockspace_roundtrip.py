"""Unit tests for blockspace BE-matrix algebra: SMILES<->matrix roundtrips,
decomposition, step application and validity checks."""
import numpy as np
from rdkit import Chem

import bond_match.blockspace as B


def _canon(smi):
    """Canonicalize a SMILES string via an explicit Mol round-trip; Chem.CanonSmiles is unreliable when MolToSmiles is monkeypatched elsewhere in the suite."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def test_mol_to_be_matrix_is_symmetric_and_valid():
    """mol_to_be_matrix returns a square, symmetric BE matrix that passes is_valid_be."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    be, atoms = B.mol_to_be_matrix(mol)
    assert be.shape[0] == be.shape[1] == atoms.shape[0]
    assert np.allclose(be, be.T)
    assert B.is_valid_be(be)


def test_smiles_matrix_roundtrip_with_hydrogens():
    """smiles_to_matrices followed by matrices_to_smiles recovers the canonical input SMILES."""
    be, atoms = B.smiles_to_matrices("CCO", include_hydrogens=True)
    recovered = B.matrices_to_smiles(be, atoms)
    assert _canon(recovered) == _canon("CCO")


def test_decompose_molecules_splits_components():
    """decompose_molecules separates a two-component system into the correct individual molecules."""
    be, atoms = B.smiles_to_matrices("CCO.O", include_hydrogens=True)
    comps = B.decompose_molecules(be, atoms)
    assert len(comps) == 2
    recovered = {_canon(B.matrices_to_smiles(c_be, c_at)) for c_be, c_at in comps}
    assert _canon("CCO") in recovered
    assert _canon("O") in recovered


def test_apply_step_symmetrizes_changes():
    """apply_step applies off-diagonal changes symmetrically and diagonal changes once: breaking the (0,1) bond zeroes both (0,1) and (1,0), and adding 2 electrons to atom 1 raises its diagonal entry to 4."""
    be = np.array([[2, 1], [1, 2]], dtype=np.float32)
    out = B.apply_step(be, [(0, 1, -1), (1, 1, 2)])
    assert out[0, 1] == 0 and out[1, 0] == 0
    assert out[1, 1] == 4


def test_is_valid_be_rejects_negatives():
    """is_valid_be accepts a non-negative matrix and rejects one with negative entries."""
    assert B.is_valid_be(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert not B.is_valid_be(np.array([[1.0, -2.0], [-2.0, 1.0]]))


def test_parse_atom_label():
    """parse_atom_label parses plain and tilde-qualified atom labels into (symbol, index, parent symbol, parent index) tuples."""
    assert B.parse_atom_label("C:15") == ("C", 15, None, None)
    assert B.parse_atom_label("H:0~C:12") == ("H", 0, "C", 12)
