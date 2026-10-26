"""Tests for the blockspace package adapter and mol_to_be_matrix."""

import numpy as np
import pytest
from rdkit import Chem


class TestImports:
    """Verify all re-exported functions are importable."""

    def test_be_matrix_exports(self):
        """BE-matrix conversion functions are importable and callable."""
        from bond_match.blockspace import decompose_molecules, matrices_to_smiles, smiles_to_matrices
        assert callable(decompose_molecules)
        assert callable(matrices_to_smiles)
        assert callable(smiles_to_matrices)

    def test_mechanism_exports(self):
        """Mechanism-step functions are importable and callable."""
        from bond_match.blockspace import apply_step, is_valid_be
        assert callable(apply_step)
        assert callable(is_valid_be)

    def test_verify_exports(self):
        """Verification helpers are importable and callable."""
        from bond_match.blockspace import reconstruct_step, resolve_atom_index, parse_atom_label
        assert callable(reconstruct_step)
        assert callable(resolve_atom_index)
        assert callable(parse_atom_label)

    def test_mol_to_be_matrix_export(self):
        """mol_to_be_matrix is importable and callable."""
        from bond_match.blockspace import mol_to_be_matrix
        assert callable(mol_to_be_matrix)


class TestMolToBeMatrix:
    """Tests for mol_to_be_matrix correctness."""

    def test_shape(self, simple_mol):
        """BE matrix is n x n and the atom array is n x 2 for an n-atom molecule."""
        from bond_match.blockspace import mol_to_be_matrix
        be, atoms = mol_to_be_matrix(simple_mol)
        n = simple_mol.GetNumAtoms()
        assert be.shape == (n, n)
        assert atoms.shape == (n, 2)

    def test_symmetry(self, simple_mol):
        """BE matrix is symmetric."""
        from bond_match.blockspace import mol_to_be_matrix
        be, _ = mol_to_be_matrix(simple_mol)
        np.testing.assert_array_equal(be, be.T)

    def test_non_negative(self, simple_mol):
        """All BE matrix entries are non-negative within numerical tolerance."""
        from bond_match.blockspace import mol_to_be_matrix
        be, _ = mol_to_be_matrix(simple_mol)
        assert np.all(be >= -1e-6), f"Negative entries in BE matrix: {be}"

    def test_electron_conservation(self, simple_mol):
        """Sum of BE matrix entries equals the total valence electron count minus formal charges."""
        from bond_match.blockspace import mol_to_be_matrix
        be, atoms = mol_to_be_matrix(simple_mol)

        PT = Chem.GetPeriodicTable()
        expected_electrons = 0
        for atom in simple_mol.GetAtoms():
            expected_electrons += PT.GetNOuterElecs(atom.GetAtomicNum())
            expected_electrons -= atom.GetFormalCharge()

        actual = be.sum()
        np.testing.assert_allclose(actual, expected_electrons, atol=1e-4)

    def test_atom_identity_matches(self, mapped_mol):
        """Second column of the atom array stores each atom's atomic number."""
        from bond_match.blockspace import mol_to_be_matrix
        _, atoms = mol_to_be_matrix(mapped_mol)
        for atom in mapped_mol.GetAtoms():
            idx = atom.GetIdx()
            assert atoms[idx, 1] == atom.GetAtomicNum()

    def test_ethanol(self):
        """All off-diagonal BE entries for ethanol are 0 or 1, since ethanol contains only single bonds."""
        from bond_match.blockspace import mol_to_be_matrix
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        be, atoms = mol_to_be_matrix(mol)

        n = mol.GetNumAtoms()
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert be[i, j] in (0.0, 1.0), f"Unexpected bond order at ({i},{j}): {be[i,j]}"


@pytest.mark.slow
class TestReconstructStepRoundtrip:
    """Round-trip of reconstruct_step followed by apply_step on a dataset entry."""

    def test_roundtrip(self, sample_v3_entry):
        """reconstruct_step yields a non-empty change set and apply_step produces a valid BE matrix."""
        from bond_match.blockspace import reconstruct_step, apply_step, is_valid_be, mol_to_be_matrix

        rxn_smi = sample_v3_entry.get("reaction_smiles_mapped") or sample_v3_entry.get("reaction_smiles", "")
        reactant_smi = rxn_smi.split(">>")[0]
        mol = Chem.MolFromSmiles(reactant_smi)
        assert mol is not None, f"Could not parse reactant: {reactant_smi}"
        mol = Chem.AddHs(mol)

        be, atoms = mol_to_be_matrix(mol)
        step = sample_v3_entry["mechanisms"][0]["steps"][0]

        changes = reconstruct_step(step, atoms, be)
        assert len(changes) > 0, "No changes found in step"

        be_new = apply_step(be, changes)
        assert is_valid_be(be_new), "BE matrix after apply_step is invalid"
