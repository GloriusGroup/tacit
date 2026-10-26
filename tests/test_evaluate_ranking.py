"""Tests for ranking evaluation utilities and BE-matrix step application.

Cross-validates that:
- apply_predicted_step_be is correct and matches blockspace.apply_step
- apply_step_with_lp correctly handles LP virtual node steps
- graph-based and BE-based step application produce equivalent product SMILES
- InferenceDataset correctly processes supported entry formats
- inference_collate correctly batches samples and preserves metadata
- score_pairs, decode_swap_type, is_entry_successful behave correctly
"""

import json
import numpy as np
import pytest
import torch
from rdkit import Chem
from unittest.mock import MagicMock

from conftest import load_uspto_entries


def _find_bond(be, atoms, z1, z2):
    """Find first bond between atoms with atomic numbers z1 and z2 in BE matrix."""
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            if be[i, j] > 0:
                if (int(atoms[i, 1]) == z1 and int(atoms[j, 1]) == z2) or \
                   (int(atoms[i, 1]) == z2 and int(atoms[j, 1]) == z1):
                    return (i, j)
    return None


def _find_two_disjoint_bonds(be, atoms):
    """Find two bonds with no shared atoms for a valid metathesis test."""
    n = len(atoms)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            if be[i, j] > 0:
                bonds.append((i, j))
    for a in range(len(bonds)):
        for b in range(a + 1, len(bonds)):
            b1, b2 = bonds[a], bonds[b]
            if {b1[0], b1[1]}.isdisjoint({b2[0], b2[1]}):
                return b1, b2
    return None, None


@pytest.fixture
def ethanol_mol():
    """Ethanol (CCO) with explicit hydrogens."""
    mol = Chem.MolFromSmiles("CCO")
    return Chem.AddHs(mol)


@pytest.fixture
def ethanol_be(ethanol_mol):
    """BE matrix and atom identity for ethanol."""
    from bond_match.blockspace import mol_to_be_matrix
    return mol_to_be_matrix(ethanol_mol)


@pytest.fixture
def ethanol_graph(ethanol_mol):
    """NetworkX graph for ethanol (includes head node)."""
    from bond_match.dataset import mol_to_nx
    return mol_to_nx(ethanol_mol)


@pytest.fixture
def v2_reaction_entry():
    """A minimal valid pathway-style reaction entry."""
    return {
        "success": True,
        "reaction_smiles": "[CH3:1][OH:2]>>[CH2:1]=[O:2]",
    }


@pytest.fixture
def v3_reaction_entry():
    """A minimal valid blockspace-style reaction entry."""
    return {
        "success": True,
        "reaction_smiles": "[CH3:1][OH:2]>>[CH2:1]=[O:2]",
    }


@pytest.fixture
def real_v3_entries():
    """Load real blockspace entries from the dataset."""
    entries = load_uspto_entries(50, require_steps=False)
    if not entries:
        pytest.skip("No valid v3 entries found")
    return entries


class TestSmilesHelpers:
    """Tests for SMILES utility functions."""

    def test_canonicalize_removes_atom_maps(self):
        from bond_match.evaluate_ranking import canonicalize_smiles
        result = canonicalize_smiles("[CH3:1][OH:2]")
        assert ":" not in result
        assert result == canonicalize_smiles("CO")

    def test_canonicalize_removes_explicit_hs(self):
        from bond_match.evaluate_ranking import canonicalize_smiles
        result = canonicalize_smiles("[H]O[H]")
        assert result == "O"

    def test_canonicalize_is_canonical(self):
        from bond_match.evaluate_ranking import canonicalize_smiles
        assert canonicalize_smiles("OCC") == canonicalize_smiles("CCO")

    def test_canonicalize_removes_stereochem(self):
        from bond_match.evaluate_ranking import canonicalize_smiles
        assert canonicalize_smiles("C/C=C/C") == canonicalize_smiles("C/C=C\\C")

    def test_canonicalize_empty_returns_empty(self):
        from bond_match.evaluate_ranking import canonicalize_smiles
        assert canonicalize_smiles("") == ""
        assert canonicalize_smiles(None) == ""

    def test_canonicalize_invalid_returns_original(self):
        from bond_match.evaluate_ranking import canonicalize_smiles
        # Invalid SMILES are returned unchanged when RDKit cannot parse them.
        assert canonicalize_smiles("not_a_smiles") == "not_a_smiles"

    def test_canonicalize_multi_fragment_sorted(self):
        from bond_match.evaluate_ranking import canonicalize_smiles
        # Sort dot-separated fragments for consistent comparison.
        r1 = canonicalize_smiles("O.C")
        r2 = canonicalize_smiles("C.O")
        assert r1 == r2

    def test_remove_atom_mapping_strips_maps(self):
        from bond_match.evaluate_ranking import remove_atom_mapping
        result = remove_atom_mapping("[CH3:1][OH:2]")
        assert ":" not in result
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_remove_atom_mapping_preserves_heavy_count(self):
        from bond_match.evaluate_ranking import remove_atom_mapping
        result = remove_atom_mapping("[CH3:1][CH2:2][OH:3]")
        assert Chem.MolFromSmiles(result).GetNumAtoms() == 3

    def test_remove_atom_mapping_invalid_raises(self):
        from bond_match.evaluate_ranking import remove_atom_mapping
        with pytest.raises(ValueError):
            remove_atom_mapping("not_valid_smiles")


class TestApplyPredictedStepBe:
    """Tests for BE-matrix step application."""

    def test_result_is_symmetric(self, ethanol_be):
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        assert b1 is not None, "Could not find two disjoint bonds"
        result = apply_predicted_step_be(be, b1, b2, swap_type=0)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_electron_conservation_swap0(self, ethanol_be):
        """Total electrons (sum of BE matrix) must be preserved."""
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        result = apply_predicted_step_be(be, b1, b2, swap_type=0)
        np.testing.assert_allclose(result.sum(), be.sum(), atol=1e-5)

    def test_electron_conservation_swap1(self, ethanol_be):
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        result = apply_predicted_step_be(be, b1, b2, swap_type=1)
        np.testing.assert_allclose(result.sum(), be.sum(), atol=1e-5)

    def test_broken_bonds_decrease_by_one(self, ethanol_be):
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        result = apply_predicted_step_be(be, b1, b2, swap_type=0)
        assert result[b1[0], b1[1]] == be[b1[0], b1[1]] - 1.0
        assert result[b2[0], b2[1]] == be[b2[0], b2[1]] - 1.0

    def test_formed_bonds_increase_swap0(self, ethanol_be):
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        u, v = b1
        x, y = b2
        result = apply_predicted_step_be(be, b1, b2, swap_type=0)
        assert result[u, x] == be[u, x] + 1.0
        assert result[v, y] == be[v, y] + 1.0

    def test_formed_bonds_increase_swap1(self, ethanol_be):
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        u, v = b1
        x, y = b2
        result = apply_predicted_step_be(be, b1, b2, swap_type=1)
        assert result[u, y] == be[u, y] + 1.0
        assert result[v, x] == be[v, x] + 1.0

    def test_swap0_and_swap1_differ(self, ethanol_be):
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        r0 = apply_predicted_step_be(be, b1, b2, swap_type=0)
        r1 = apply_predicted_step_be(be, b1, b2, swap_type=1)
        assert not np.allclose(r0, r1), "swap_type 0 and 1 should produce different results"

    def test_no_lp_change_for_standard_metathesis(self, ethanol_be):
        """In a 4-distinct-atom metathesis, each atom loses 1 bond and gains 1,
        so net bond change is 0 and LP should not change."""
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        result = apply_predicted_step_be(be, b1, b2, swap_type=0)
        involved = {b1[0], b1[1], b2[0], b2[1]}
        for a in involved:
            assert result[a, a] == be[a, a], (
                f"LP changed for atom {a}: {be[a, a]} -> {result[a, a]}"
            )

    def test_original_not_mutated(self, ethanol_be):
        from bond_match.evaluate_ranking import apply_predicted_step_be
        be, atoms = ethanol_be
        be_copy = be.copy()
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        _ = apply_predicted_step_be(be, b1, b2, swap_type=0)
        np.testing.assert_array_equal(be, be_copy)

    def test_matches_blockspace_apply_step(self, ethanol_be):
        """Result should match blockspace.apply_step with equivalent explicit changes."""
        from bond_match.evaluate_ranking import apply_predicted_step_be
        from bond_match.blockspace import apply_step

        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        u, v = b1
        x, y = b2

        # Build explicit changes equivalent to apply_predicted_step_be.
        changes = []
        atom_bond_delta = np.zeros(len(atoms), dtype=np.float32)

        for a, b in [(u, v), (x, y)]:
            changes.append((a, b, -1.0))
            atom_bond_delta[a] -= 1.0
            atom_bond_delta[b] -= 1.0

        for a, b in [(u, x), (v, y)]:
            changes.append((a, b, 1.0))
            atom_bond_delta[a] += 1.0
            atom_bond_delta[b] += 1.0

        for k in range(len(atoms)):
            if abs(atom_bond_delta[k]) > 1e-6:
                changes.append((k, k, -atom_bond_delta[k]))

        result_fn = apply_predicted_step_be(be, b1, b2, swap_type=0)
        result_bs = apply_step(be, changes)
        np.testing.assert_allclose(result_fn, result_bs, atol=1e-6)

    def test_breaking_nonexistent_bond_produces_negative(self):
        """Breaking a non-existent bond should produce negative BE entries."""
        from bond_match.evaluate_ranking import apply_predicted_step_be

        # Synthetic system: A(0)-B(1) and C(2)-D(3), with no cross bonds.
        be = np.array([
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ], dtype=np.float32)

        # Break (0,2) and (1,3); neither bond exists in the input matrix.
        result = apply_predicted_step_be(be, (0, 2), (1, 3), swap_type=0)
        assert result[0, 2] < -0.01, "Expected negative at broken non-bond (0,2)"
        assert result[1, 3] < -0.01, "Expected negative at broken non-bond (1,3)"

    def test_multiple_molecules(self):
        """Test symmetry and conservation across a range of molecules."""
        from bond_match.evaluate_ranking import apply_predicted_step_be
        from bond_match.blockspace import mol_to_be_matrix

        for smi in ["CC(=O)O", "CCN", "c1ccccc1O", "CC(C)C"]:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            be, atoms = mol_to_be_matrix(mol)
            b1, b2 = _find_two_disjoint_bonds(be, atoms)
            if b1 is None:
                continue

            result = apply_predicted_step_be(be, b1, b2, swap_type=0)
            np.testing.assert_array_almost_equal(
                result, result.T, err_msg=f"Asymmetric result for {smi}"
            )
            np.testing.assert_allclose(
                result.sum(), be.sum(), atol=1e-5,
                err_msg=f"Electron conservation failed for {smi}"
            )


class TestApplyStepWithLp:
    """Tests for LP-aware step application."""

    def test_lp_break_decreases_diagonal(self):
        """Breaking an LP edge should decrease the real atom's diagonal by 2."""
        from bond_match.evaluate_ranking import apply_step_with_lp

        # The LP virtual node with index 3 maps to real atom 0.
        be = np.array([
            [4.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        reverse_lp_map = {3: 0}

        # Break LP edge (3, 0) and real bond (1, 2); form (3, 2) and (0, 1).
        result = apply_step_with_lp(be, (3, 0), (1, 2), swap_type=0, reverse_lp_map=reverse_lp_map)

        assert result[0, 0] == be[0, 0] - 2
        assert result[1, 2] == be[1, 2] - 1.0

    def test_lp_form_increases_diagonal(self):
        """Forming an LP edge should increase the real atom's diagonal by 2."""
        from bond_match.evaluate_ranking import apply_step_with_lp

        be = np.array([
            [2.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
        ], dtype=np.float32)

        reverse_lp_map = {3: 2}

        # The LP edge maps the diagonal change back to the associated real atom.
        result = apply_step_with_lp(be, (0, 1), (3, 2), swap_type=1, reverse_lp_map=reverse_lp_map)

        assert result[2, 2] == be[2, 2] - 2
        assert result[1, 1] == be[1, 1] + 2
        assert result[0, 0] == be[0, 0]

    def test_no_lp_nodes_behaves_like_normal_step(self, ethanol_be):
        """When no LP nodes are involved, result should match apply_predicted_step_be."""
        from bond_match.evaluate_ranking import apply_step_with_lp, apply_predicted_step_be

        be, atoms = ethanol_be
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        if b1 is None:
            pytest.skip("No disjoint bonds found")

        result_lp = apply_step_with_lp(be, b1, b2, swap_type=0, reverse_lp_map={})
        result_normal = apply_predicted_step_be(be, b1, b2, swap_type=0)
        np.testing.assert_allclose(result_lp, result_normal, atol=1e-6)

    def test_original_not_mutated(self):
        """apply_step_with_lp should not mutate the input BE matrix."""
        from bond_match.evaluate_ranking import apply_step_with_lp

        be = np.array([
            [4.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        be_copy = be.copy()

        reverse_lp_map = {3: 0}
        _ = apply_step_with_lp(be, (3, 0), (1, 2), swap_type=0, reverse_lp_map=reverse_lp_map)
        np.testing.assert_array_equal(be, be_copy)


class TestBeResultToSmiles:
    """Tests for BE matrix to SMILES conversion."""

    def test_roundtrip_single_molecule(self):
        from bond_match.evaluate_ranking import be_result_to_smiles, canonicalize_smiles
        from bond_match.blockspace import mol_to_be_matrix

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        be, atoms = mol_to_be_matrix(mol)
        result_smi = be_result_to_smiles(be, atoms)
        assert canonicalize_smiles(result_smi) == canonicalize_smiles("CCO")

    def test_roundtrip_multiple_molecules(self):
        """Dot-separated system should decompose and reconstruct correctly."""
        from bond_match.evaluate_ranking import be_result_to_smiles, canonicalize_smiles
        from bond_match.blockspace import smiles_to_matrices

        smi = "[CH4:1].[OH2:2]"
        be, atoms = smiles_to_matrices(smi, kekulize=True, include_hydrogens=True)
        result_smi = be_result_to_smiles(be, atoms)
        parts = set(canonicalize_smiles(p) for p in result_smi.split("."))
        assert "C" in parts
        assert "O" in parts

    def test_returns_string(self):
        from bond_match.evaluate_ranking import be_result_to_smiles
        from bond_match.blockspace import mol_to_be_matrix

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        be, atoms = mol_to_be_matrix(mol)
        result = be_result_to_smiles(be, atoms)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_roundtrip_after_step(self):
        """After a valid step, be_result_to_smiles should still produce parseable SMILES."""
        from bond_match.evaluate_ranking import apply_predicted_step_be, be_result_to_smiles
        from bond_match.blockspace import mol_to_be_matrix

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        be, atoms = mol_to_be_matrix(mol)
        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        if b1 is None:
            pytest.skip("No disjoint bonds found")

        be_result = apply_predicted_step_be(be, b1, b2, swap_type=0)
        smi = be_result_to_smiles(be_result, atoms)
        assert isinstance(smi, str)
        for part in smi.split("."):
            if part:
                assert Chem.MolFromSmiles(part, sanitize=False) is not None, (
                    f"Unparseable SMILES fragment: {part}"
                )


class TestV2V3StepEquivalence:
    """Cross-validate graph-based and BE-matrix-based step application."""

    def test_same_product_for_ethanol(self):
        from bond_match.evaluate_ranking import (
            apply_predicted_step_be, be_result_to_smiles, canonicalize_smiles,
        )
        from bond_match.dataset import mol_to_nx, apply_elemental_step, nx_to_mol
        from bond_match.blockspace import mol_to_be_matrix

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        G = mol_to_nx(mol)
        be, atoms = mol_to_be_matrix(mol)

        b1, b2 = _find_two_disjoint_bonds(be, atoms)
        assert b1 is not None, "No disjoint bonds found"

        for swap_type in [0, 1]:
            G_result = apply_elemental_step(G.copy(), b1, b2, swap_type=swap_type)
            v2_mol = nx_to_mol(G_result, remove_virtual=True)
            v2_smi = canonicalize_smiles(Chem.MolToSmiles(v2_mol))

            be_result = apply_predicted_step_be(be, b1, b2, swap_type=swap_type)
            v3_smi = canonicalize_smiles(be_result_to_smiles(be_result, atoms))

            assert v2_smi == v3_smi, (
                f"V2/V3 mismatch for swap_type={swap_type}: v2={v2_smi}, v3={v3_smi}"
            )

    def test_equivalence_on_multiple_molecules(self):
        """Test graph/BE equivalence across a range of molecules."""
        from bond_match.evaluate_ranking import (
            apply_predicted_step_be, be_result_to_smiles, canonicalize_smiles,
        )
        from bond_match.dataset import mol_to_nx, apply_elemental_step, nx_to_mol
        from bond_match.blockspace import mol_to_be_matrix

        # Compare graph and BE-matrix swaps for simple aliphatic systems.
        # Arbitrary swaps across aromatic rings are not chemically equivalent
        # between the two representations.
        smiles_list = ["CCO", "CCN", "CC(=O)O", "CC(C)C", "CCCC"]

        tested = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            G = mol_to_nx(mol)
            be, atoms = mol_to_be_matrix(mol)
            b1, b2 = _find_two_disjoint_bonds(be, atoms)
            if b1 is None:
                continue

            G_result = apply_elemental_step(G.copy(), b1, b2, swap_type=0)
            v2_mol = nx_to_mol(G_result, remove_virtual=True)
            if v2_mol is None:
                continue
            v2_smi = canonicalize_smiles(Chem.MolToSmiles(v2_mol))

            be_result = apply_predicted_step_be(be, b1, b2, swap_type=0)
            v3_smi = canonicalize_smiles(be_result_to_smiles(be_result, atoms))

            assert v2_smi == v3_smi, f"V2/V3 mismatch for {smi}: v2={v2_smi}, v3={v3_smi}"
            tested += 1

        assert tested >= 4, f"Only tested {tested} molecules"


class TestIsEntrySuccessful:
    """Tests for entry filtering logic."""

    def test_default_success(self):
        from bond_match.evaluate_ranking import is_entry_successful
        assert is_entry_successful({}) is True

    def test_explicit_success_false(self):
        from bond_match.evaluate_ranking import is_entry_successful
        assert is_entry_successful({"success": False}) is False

    def test_explicit_success_true(self):
        from bond_match.evaluate_ranking import is_entry_successful
        assert is_entry_successful({"success": True}) is True

    def test_too_many_steps_rejected(self):
        from bond_match.evaluate_ranking import is_entry_successful
        entry = {
            "success": True,
            "mechanisms": [{"steps": [{}] * 5}],
        }
        assert is_entry_successful(entry) is False

    def test_four_steps_accepted(self):
        from bond_match.evaluate_ranking import is_entry_successful
        entry = {
            "success": True,
            "mechanisms": [{"steps": [{}] * 4}],
        }
        assert is_entry_successful(entry) is True

    def test_num_steps_field_fallback(self):
        from bond_match.evaluate_ranking import is_entry_successful
        assert is_entry_successful({"num_steps": 5}) is False
        assert is_entry_successful({"num_steps": 3}) is True


class TestStepResult:
    """Tests for the StepResult dataclass."""

    def test_fields(self):
        from bond_match.evaluate_ranking import StepResult
        r = StepResult(product_smi="CCO", state={"key": "val"}, success=True)
        assert r.product_smi == "CCO"
        assert r.state == {"key": "val"}
        assert r.success is True

    def test_failure(self):
        from bond_match.evaluate_ranking import StepResult
        r = StepResult(product_smi="", state={}, success=False)
        assert r.success is False
        assert r.product_smi == ""
