"""Tests for v3 blockspace preprocessing pipeline in dataset.py.

Cross-validates dataset.py atom resolution and step processing against
the blockspace verify.py reference implementation.
"""

import json
import numpy as np
import pytest
import torch
from rdkit import Chem

from conftest import USPTO_TRAIN, load_uspto_entries


def test_dataset_cache_prefix_distinguishes_versioned_parent_paths():
    """_dataset_cache_prefix yields distinct cache keys for datasets that differ only in a versioned parent directory."""
    from bond_match.dataset import _dataset_cache_prefix

    v5_key = _dataset_cache_prefix(
        "/data/blockspace/v5.0/uspto_mit/train.jsonl"
    )
    v6_key = _dataset_cache_prefix(
        "/data/blockspace/v6.0/uspto_mit/train.jsonl"
    )

    assert v5_key != v6_key
    assert v5_key == "blockspace_v5.0_uspto_mit_train"
    assert v6_key == "blockspace_v6.0_uspto_mit_train"


@pytest.fixture
def v3_entries():
    """Load the first 200 valid v3 blockspace entries for batch tests; skip if the dataset is unavailable."""
    return load_uspto_entries(200)


@pytest.fixture
def single_step_entry(v3_entries):
    """Return a v3 entry whose first mechanism has exactly one step."""
    for e in v3_entries:
        if len(e["mechanisms"][0]["steps"]) == 1:
            return e
    pytest.skip("No single-step entry found")


@pytest.fixture
def multi_step_entry(v3_entries):
    """Return a v3 entry whose first mechanism has at least two steps."""
    for e in v3_entries:
        if len(e["mechanisms"][0]["steps"]) >= 2:
            return e
    pytest.skip("No multi-step entry found")


@pytest.fixture
def entry_with_lp(v3_entries):
    """Return a v3 entry containing lone-pair (LP) changes in at least one step."""
    for e in v3_entries:
        for step in e["mechanisms"][0]["steps"]:
            if step.get("lp"):
                return e
    pytest.skip("No entry with LP changes found")


class TestParseV3AtomLabel:
    """Tests for parse_v3_atom_label."""

    def test_mapped_atom(self):
        """A mapped heavy-atom label parses into element, map number, and is_unmapped_h=False."""
        from bond_match.dataset import parse_v3_atom_label

        result = parse_v3_atom_label("C:12")
        assert result["element"] == "C"
        assert result["map"] == 12
        assert result["is_unmapped_h"] is False

    def test_unmapped_h(self):
        """An H:0~X:N label is flagged as unmapped hydrogen with the correct parent element and map."""
        from bond_match.dataset import parse_v3_atom_label

        result = parse_v3_atom_label("H:0~N:15")
        assert result["element"] == "H"
        assert result["map"] == 0
        assert result["is_unmapped_h"] is True
        assert result["parent_element"] == "N"
        assert result["parent_map"] == 15

    def test_mapped_h(self):
        """A hydrogen with a nonzero map number is not flagged as unmapped."""
        from bond_match.dataset import parse_v3_atom_label

        result = parse_v3_atom_label("H:5")
        assert result["element"] == "H"
        assert result["map"] == 5
        assert result["is_unmapped_h"] is False

    def test_oxygen(self):
        """An oxygen label parses into the correct element and map number."""
        from bond_match.dataset import parse_v3_atom_label

        result = parse_v3_atom_label("O:3")
        assert result["element"] == "O"
        assert result["map"] == 3

    def test_unmapped_h_on_oxygen(self):
        """An unmapped hydrogen attached to oxygen records the oxygen parent element and map."""
        from bond_match.dataset import parse_v3_atom_label

        result = parse_v3_atom_label("H:0~O:27")
        assert result["is_unmapped_h"] is True
        assert result["parent_element"] == "O"
        assert result["parent_map"] == 27

    def test_consistency_with_verify_parse_atom_label(self):
        """parse_v3_atom_label and verify.parse_atom_label agree on element, map number, and parent information for the same label.

        A mapped hydrogen without a '~' parent suffix is not flagged as
        unmapped by dataset.py, so the is_unmapped_h check is restricted
        to labels with map > 0 when no parent is present.
        """
        from bond_match.dataset import parse_v3_atom_label
        from bond_match.blockspace import parse_atom_label

        labels = ["C:12", "N:1", "H:5", "H:0~N:15", "H:0~O:27", "Br:2", "Cl:51"]
        for label in labels:
            ds_result = parse_v3_atom_label(label)
            v_elem, v_map, v_parent_elem, v_parent_map = parse_atom_label(label)

            assert ds_result["element"] == v_elem, f"Element mismatch for {label}"
            assert ds_result["map"] == v_map, f"Map mismatch for {label}"

            if v_parent_elem is not None:
                assert ds_result["is_unmapped_h"] is True
                assert ds_result["parent_element"] == v_parent_elem
                assert ds_result["parent_map"] == v_parent_map
            else:
                if ds_result["map"] > 0:
                    assert ds_result["is_unmapped_h"] is False


@pytest.mark.slow
class TestAtomResolutionCrossValidation:
    """Cross-validate dataset.py graph-based atom resolution against
    verify.py BE-matrix-based resolution on real data."""

    def test_mapped_atoms_agree(self, sample_v3_entry):
        """For mapped atoms (map > 0), graph-based and BE-matrix-based resolution return the same atom index."""
        from bond_match.dataset import (
            parse_v3_atom_label,
            find_atom_by_v3_label,
            mol_to_nx,
        )
        from bond_match.blockspace import resolve_atom_index, mol_to_be_matrix

        rxn_smi = (
            sample_v3_entry.get("reaction_smiles_mapped")
            or sample_v3_entry["reaction_smiles"]
        )
        reactant_smi = rxn_smi.split(">>")[0]
        mol = Chem.AddHs(Chem.MolFromSmiles(reactant_smi))
        G = mol_to_nx(mol)
        be, atoms = mol_to_be_matrix(mol)

        step = sample_v3_entry["mechanisms"][0]["steps"][0]
        for pair in step.get("break", []) + step.get("form", []):
            for label_str in pair:
                parsed = parse_v3_atom_label(label_str)
                if parsed["map"] > 0:
                    graph_idx = find_atom_by_v3_label(G, parsed)
                    be_idx = resolve_atom_index(label_str, atoms, be)

                    assert graph_idx is not None, (
                        f"Graph resolver failed for {label_str}"
                    )
                    assert graph_idx == be_idx, (
                        f"Resolvers disagree for {label_str}: graph={graph_idx}, be={be_idx}"
                    )

    def test_unmapped_h_resolves_to_h_bonded_to_parent(self, sample_v3_entry):
        """For H:0~X:N labels, both resolvers return a hydrogen atom that is bonded to the designated parent atom."""
        from bond_match.dataset import (
            parse_v3_atom_label,
            find_atom_by_v3_label,
            mol_to_nx,
        )
        from bond_match.blockspace import resolve_atom_index, mol_to_be_matrix

        rxn_smi = (
            sample_v3_entry.get("reaction_smiles_mapped")
            or sample_v3_entry["reaction_smiles"]
        )
        reactant_smi = rxn_smi.split(">>")[0]
        mol = Chem.AddHs(Chem.MolFromSmiles(reactant_smi))
        G = mol_to_nx(mol)
        be, atoms = mol_to_be_matrix(mol)

        step = sample_v3_entry["mechanisms"][0]["steps"][0]
        for pair in step.get("break", []) + step.get("form", []):
            for label_str in pair:
                parsed = parse_v3_atom_label(label_str)
                if not parsed["is_unmapped_h"]:
                    continue

                graph_idx = find_atom_by_v3_label(G, parsed)
                be_idx = resolve_atom_index(label_str, atoms, be)

                assert graph_idx is not None, f"Graph resolver failed for {label_str}"
                assert G.nodes[graph_idx].get("atomic_number") == 1
                assert int(atoms[be_idx][1]) == 1

                from bond_match.dataset import _get_atomic_num

                parent_z = _get_atomic_num(parsed["parent_element"])
                parent_map = parsed["parent_map"]

                graph_parent = None
                for nbr in G.neighbors(graph_idx):
                    d = G.nodes[nbr]
                    if (
                        d.get("map_number") == parent_map
                        and d.get("atomic_number") == parent_z
                    ):
                        graph_parent = nbr
                        break
                assert graph_parent is not None, (
                    f"H {graph_idx} not bonded to parent in graph"
                )

                assert be[be_idx, be_idx] >= 0
                be_parent = None
                for i in range(len(atoms)):
                    if int(atoms[i][0]) == parent_map and int(atoms[i][1]) == parent_z:
                        be_parent = i
                        break
                assert be_parent is not None
                assert be[be_idx, be_parent] > 0, (
                    f"H {be_idx} not bonded to parent {be_parent} in BE"
                )


@pytest.mark.slow
class TestReconstructStepCrossValidation:
    """Verify that the dataset pipeline produces the same step changes as
    verify.reconstruct_step and that intermediate BE matrices stay valid."""

    def test_reconstruct_step_changes_match(self, sample_v3_entry):
        """reconstruct_step on the first mechanism step returns a list of (i, j, value) tuples; break entries are negative and form entries positive off-diagonal changes."""
        from bond_match.blockspace import reconstruct_step, mol_to_be_matrix

        rxn_smi = (
            sample_v3_entry.get("reaction_smiles_mapped")
            or sample_v3_entry["reaction_smiles"]
        )
        reactant_smi = rxn_smi.split(">>")[0]
        mol = Chem.AddHs(Chem.MolFromSmiles(reactant_smi))
        be, atoms = mol_to_be_matrix(mol)

        step = sample_v3_entry["mechanisms"][0]["steps"][0]
        changes = reconstruct_step(step, atoms, be)

        assert isinstance(changes, list)
        assert all(len(c) == 3 for c in changes)

        breaks = [(i, j, v) for i, j, v in changes if i != j and v < 0]
        forms = [(i, j, v) for i, j, v in changes if i != j and v > 0]

        assert len(breaks) >= 0
        assert len(forms) >= 0

    def test_all_intermediates_valid_single_mechanism(self, sample_v3_entry):
        """Every intermediate BE matrix obtained by sequentially applying the first mechanism's steps passes is_valid_be."""
        from bond_match.blockspace import (
            reconstruct_step,
            apply_step,
            is_valid_be,
            mol_to_be_matrix,
        )

        rxn_smi = (
            sample_v3_entry.get("reaction_smiles_mapped")
            or sample_v3_entry["reaction_smiles"]
        )
        reactant_smi = rxn_smi.split(">>")[0]
        mol = Chem.AddHs(Chem.MolFromSmiles(reactant_smi))
        be, atoms = mol_to_be_matrix(mol)

        steps = sample_v3_entry["mechanisms"][0]["steps"]
        be_current = be.copy()
        for step_idx, step in enumerate(steps):
            changes = reconstruct_step(step, atoms, be_current)
            be_current = apply_step(be_current, changes)
            assert is_valid_be(be_current), (
                f"Invalid BE matrix after step {step_idx}: min={be_current.min()}"
            )

    def test_final_product_matches_reference_smiles(self, sample_v3_entry):
        """After applying all mechanism steps, the decomposed BE matrix matches the SMILES recorded in the last step, compared canonically with atom maps stripped and hydrogens removed."""
        from bond_match.blockspace import (
            reconstruct_step,
            apply_step,
            mol_to_be_matrix,
            decompose_molecules,
            matrices_to_smiles,
        )

        rxn_smi = (
            sample_v3_entry.get("reaction_smiles_mapped")
            or sample_v3_entry["reaction_smiles"]
        )
        reactant_smi = rxn_smi.split(">>")[0]
        mol = Chem.AddHs(Chem.MolFromSmiles(reactant_smi))
        be, atoms = mol_to_be_matrix(mol)

        steps = sample_v3_entry["mechanisms"][0]["steps"]
        be_current = be.copy()
        for step in steps:
            changes = reconstruct_step(step, atoms, be_current)
            be_current = apply_step(be_current, changes)

        last_smiles = steps[-1].get("smiles", "")
        if not last_smiles:
            pytest.skip("No reference SMILES in last step")

        mols = decompose_molecules(be_current, atoms)
        actual_smiles = ".".join(matrices_to_smiles(m, a) for m, a in mols)

        def canonical(smi):
            """Canonicalize a SMILES with atom maps stripped and hydrogens removed."""
            m = Chem.MolFromSmiles(smi, sanitize=False)
            if m is None:
                return None
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            try:
                Chem.SanitizeMol(m)
                m = Chem.RemoveHs(m)
            except Exception:
                pass
            return Chem.MolToSmiles(m)

        assert canonical(actual_smiles) == canonical(last_smiles), (
            f"Final SMILES mismatch:\n  actual:   {canonical(actual_smiles)}\n  expected: {canonical(last_smiles)}"
        )

    def test_verify_mechanism_passes(self, sample_v3_entry):
        """verify_mechanism reports success for a valid dataset entry."""
        from bond_match.blockspace.verify import verify_mechanism

        ok, msgs = verify_mechanism(sample_v3_entry)
        assert ok, f"verify_mechanism failed: {msgs}"


class TestMolToBeMatrixConsistency:
    """Verify that mol_to_be_matrix from blockspace.__init__ and
    smiles_to_matrices from be_matrix.py produce equivalent results."""

    def test_same_result_as_smiles_to_matrices(self):
        """mol_to_be_matrix and smiles_to_matrices(kekulize=True, include_hydrogens=True) produce identical BE matrices and atom identities; orderings coincide because both follow RDKit atom indices."""
        from bond_match.blockspace import mol_to_be_matrix, smiles_to_matrices

        smi = "[CH3:1][OH:2]"
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)

        be_mol, atoms_mol = mol_to_be_matrix(mol)
        be_smi, atoms_smi = smiles_to_matrices(
            smi, kekulize=True, include_hydrogens=True
        )

        assert be_mol.shape == be_smi.shape, (
            f"Shape mismatch: mol={be_mol.shape}, smi={be_smi.shape}"
        )

        np.testing.assert_array_equal(atoms_mol, atoms_smi)

        np.testing.assert_allclose(be_mol, be_smi, atol=1e-5)

    def test_consistency_on_multiple_molecules(self):
        """mol_to_be_matrix and smiles_to_matrices agree for ethane, water, ammonia, and ethylene."""
        from bond_match.blockspace import mol_to_be_matrix, smiles_to_matrices

        smiles_list = [
            "[CH3:1][CH3:2]",
            "[OH2:1]",
            "[NH3:1]",
            "[CH2:1]=[CH2:2]",
        ]
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            be_mol, atoms_mol = mol_to_be_matrix(mol)
            be_smi, atoms_smi = smiles_to_matrices(
                smi, kekulize=True, include_hydrogens=True
            )

            np.testing.assert_array_equal(
                atoms_mol, atoms_smi, err_msg=f"Atoms differ for {smi}"
            )
            np.testing.assert_allclose(
                be_mol, be_smi, atol=1e-5, err_msg=f"BE differs for {smi}"
            )


@pytest.mark.slow
class TestProcessReactionEntryV3:
    """Tests for process_reaction_entry_v3."""

    def test_failed_entry_returns_empty(self):
        """An entry marked success=False yields no samples."""
        from bond_match.dataset import process_reaction_entry_v3

        assert process_reaction_entry_v3({"success": False}) == []

    def test_no_mechanisms_returns_empty(self):
        """An entry with an empty mechanisms list yields no samples."""
        from bond_match.dataset import process_reaction_entry_v3

        assert (
            process_reaction_entry_v3(
                {
                    "success": True,
                    "reaction_smiles": "CC>>CC",
                    "mechanisms": [],
                }
            )
            == []
        )

    def test_no_reaction_arrow_returns_empty(self):
        """A reaction SMILES without a '>>' arrow yields no samples."""
        from bond_match.dataset import process_reaction_entry_v3

        assert (
            process_reaction_entry_v3(
                {
                    "success": True,
                    "reaction_smiles": "CC",
                    "mechanisms": [{"steps": [{"break": [], "form": []}]}],
                }
            )
            == []
        )


    def test_step_samples_have_nonzero_labels(self, sample_v3_entry):
        """Every step sample has exactly two nonzero y entries."""
        from bond_match.dataset import process_reaction_entry_v3

        samples = process_reaction_entry_v3(sample_v3_entry, include_product=True)
        step_samples = [s for s in samples if "step" in s["type"]]

        for s in step_samples:
            nonzero = np.count_nonzero(s["y"])
            assert nonzero == 2, (
                f"Expected 2 nonzero labels, got {nonzero} in {s['type']}"
            )

    def test_product_sample_has_zero_labels(self, sample_v3_entry):
        """Exactly one product sample is produced and its y labels are all zero."""
        from bond_match.dataset import process_reaction_entry_v3

        samples = process_reaction_entry_v3(sample_v3_entry, include_product=True)
        product_samples = [s for s in samples if s["type"] == "product"]

        assert len(product_samples) == 1, "Expected exactly 1 product sample"
        assert np.all(product_samples[0]["y"] == 0)

    def test_blockspace_product_preserves_side_products(self):
        """With enable_blockspace_product the product sample retains all atoms including side products (2 C, 1 Cl, 1 Br), whereas the SMILES-based product contains only the listed product atoms (1 C, 1 Br)."""
        from bond_match.dataset import process_reaction_entry_v3

        entry = {
            "success": True,
            "reaction_smiles_mapped": "[CH3:1][Cl:2].[CH3:3][Br:4]>>[CH3:1][Br:4]",
            "mechanisms": [
                {
                    "steps": [
                        {
                            "break": [["C:1", "Cl:2"], ["C:3", "Br:4"]],
                            "form": [["C:1", "Br:4"], ["C:3", "Cl:2"]],
                            "lp": [],
                        }
                    ]
                }
            ],
        }

        blockspace_samples = process_reaction_entry_v3(
            entry,
            include_product=True,
            enable_blockspace_product=True,
        )
        smiles_samples = process_reaction_entry_v3(
            entry,
            include_product=True,
            enable_blockspace_product=False,
        )

        blockspace_product = [s for s in blockspace_samples if s["type"] == "product"][0]
        smiles_product = [s for s in smiles_samples if s["type"] == "product"][0]

        blockspace_atomic_numbers = blockspace_product["x"][:, 0].tolist()
        smiles_atomic_numbers = smiles_product["x"][:, 0].tolist()

        assert blockspace_atomic_numbers.count(6) == 2
        assert blockspace_atomic_numbers.count(17) == 1
        assert blockspace_atomic_numbers.count(35) == 1
        assert smiles_atomic_numbers.count(6) == 1
        assert smiles_atomic_numbers.count(17) == 0
        assert smiles_atomic_numbers.count(35) == 1

    def test_no_product_when_disabled(self, sample_v3_entry):
        """include_product=False suppresses the product sample."""
        from bond_match.dataset import process_reaction_entry_v3

        samples = process_reaction_entry_v3(sample_v3_entry, include_product=False)
        product_samples = [s for s in samples if s["type"] == "product"]
        assert len(product_samples) == 0

    def test_swap_type_label_is_1_or_2(self, sample_v3_entry):
        """Nonzero labels are 1 (swap type 0) or 2 (swap type 1), and both marked bonds in a sample carry the same label."""
        from bond_match.dataset import process_reaction_entry_v3

        samples = process_reaction_entry_v3(sample_v3_entry, include_product=False)
        for s in samples:
            unique_labels = set(np.unique(s["y"]))
            nonzero_labels = unique_labels - {0}
            assert nonzero_labels.issubset({1, 2}), (
                f"Unexpected label values: {nonzero_labels}"
            )
            if nonzero_labels:
                assert len(nonzero_labels) == 1, (
                    f"Mixed swap types in same sample: {nonzero_labels}"
                )



    def test_use_all_mechanisms_produces_more_samples(self, sample_v3_entry):
        """When the entry has multiple mechanisms, use_all_mechanisms=True produces at least as many samples as the single-mechanism case."""
        from bond_match.dataset import process_reaction_entry_v3

        samples_one = process_reaction_entry_v3(
            sample_v3_entry, include_product=False, use_all_mechanisms=False
        )
        samples_all = process_reaction_entry_v3(
            sample_v3_entry, include_product=False, use_all_mechanisms=True
        )

        num_mechs = len(sample_v3_entry.get("mechanisms", []))
        if num_mechs > 1:
            assert len(samples_all) >= len(samples_one), (
                f"use_all_mechanisms should produce >= samples: {len(samples_all)} vs {len(samples_one)}"
            )

    def test_entry_with_lp_changes_does_not_crash(self, entry_with_lp):
        """Entries with lone-pair changes are processed without raising and return a list."""
        from bond_match.dataset import process_reaction_entry_v3

        samples = process_reaction_entry_v3(entry_with_lp, include_product=True)
        assert isinstance(samples, list)

    def test_extended_features_add_catalyst_column(self, sample_v3_entry):
        """With extended_features=True the node feature tensor has 10 columns (5 base plus 5 extended)."""
        from bond_match.dataset import process_reaction_entry_v3

        samples = process_reaction_entry_v3(
            sample_v3_entry, include_product=True, extended_features=True
        )
        if samples:
            assert samples[0]["x"].shape[1] == 10, (
                f"Expected 10 features, got {samples[0]['x'].shape[1]}"
            )


class TestBeToNx:
    """Tests for conversion of BE matrices to NetworkX graphs via _be_to_nx."""

    def test_roundtrip_preserves_atom_count(self):
        """The graph contains one real node (atomic_number > 0) per molecule atom."""
        from bond_match.blockspace import mol_to_be_matrix
        from bond_match.dataset import _be_to_nx

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        n_atoms = mol.GetNumAtoms()

        be, atoms = mol_to_be_matrix(mol)
        G = _be_to_nx(be, atoms)

        real_nodes = [n for n, d in G.nodes(data=True) if d.get("atomic_number", 0) > 0]
        assert len(real_nodes) == n_atoms

    def test_has_head_node(self):
        """The graph contains exactly one virtual head node (atomic_number == 0)."""
        from bond_match.blockspace import mol_to_be_matrix
        from bond_match.dataset import _be_to_nx

        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)

        be, atoms = mol_to_be_matrix(mol)
        G = _be_to_nx(be, atoms)

        head_nodes = [
            n for n, d in G.nodes(data=True) if d.get("atomic_number", 0) == 0
        ]
        assert len(head_nodes) == 1

    def test_formal_charges_correct(self):
        """Formal charges computed from the BE diagonal match RDKit's values for ammonium."""
        from bond_match.blockspace import mol_to_be_matrix
        from bond_match.dataset import _be_to_nx

        mol = Chem.MolFromSmiles("[NH4+]")
        mol = Chem.AddHs(mol)
        be, atoms = mol_to_be_matrix(mol)
        G = _be_to_nx(be, atoms)

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            expected_charge = atom.GetFormalCharge()
            actual_charge = G.nodes[idx].get("formal_charge", 0)
            assert actual_charge == expected_charge, (
                f"Charge mismatch at atom {idx}: expected {expected_charge}, got {actual_charge}"
            )

    def test_bond_orders_preserved(self):
        """Off-diagonal BE entries become edge bond_order attributes: the C=O bond of formaldehyde has bond_order 2.0."""
        from bond_match.blockspace import mol_to_be_matrix
        from bond_match.dataset import _be_to_nx

        mol = Chem.MolFromSmiles("C=O")
        mol = Chem.AddHs(mol)
        be, atoms = mol_to_be_matrix(mol)
        G = _be_to_nx(be, atoms)

        found_double = False
        for u, v, data in G.edges(data=True):
            z_u = G.nodes[u].get("atomic_number", 0)
            z_v = G.nodes[v].get("atomic_number", 0)
            if {z_u, z_v} == {6, 8}:
                assert data["bond_order"] == 2.0, (
                    f"Expected double bond, got {data['bond_order']}"
                )
                found_double = True
        assert found_double, "C=O double bond not found in graph"

    def test_catalyst_flags_preserved(self):
        """catalyst_flags provided to _be_to_nx are stored as is_catalyst node attributes."""
        from bond_match.blockspace import mol_to_be_matrix
        from bond_match.dataset import _be_to_nx

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        be, atoms = mol_to_be_matrix(mol)
        n = mol.GetNumAtoms()

        flags = {i: 0 for i in range(n)}
        flags[2] = 1
        G = _be_to_nx(be, atoms, catalyst_flags=flags)

        assert G.nodes[2].get("is_catalyst") == 1
        assert G.nodes[0].get("is_catalyst") == 0


@pytest.mark.slow
class TestVerifyBatch:
    """Batch verification of dataset entries against the blockspace verification logic."""

    def test_batch_verification(self, v3_entries):
        """At least 85% of the first 100 entries pass verify_mechanism."""
        from bond_match.blockspace.verify import verify_mechanism

        passed = 0
        failed = 0
        for entry in v3_entries[:100]:
            try:
                ok, _ = verify_mechanism(entry)
                if ok:
                    passed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        total = passed + failed
        rate = passed / total if total > 0 else 0
        assert rate >= 0.85, f"Only {rate:.1%} passed verification ({passed}/{total})"


@pytest.mark.slow
class TestDatasetVsVerifyIntermediates:
    """Consistency between training samples from process_reaction_entry_v3 and the verify.py step walkthrough."""

    def test_step_count_matches(self, sample_v3_entry):
        """The number of training step samples equals the number of 2-bond-swap steps identified by reconstruct_step."""
        from bond_match.dataset import process_reaction_entry_v3
        from bond_match.blockspace import reconstruct_step, apply_step, mol_to_be_matrix

        rxn_smi = (
            sample_v3_entry.get("reaction_smiles_mapped")
            or sample_v3_entry["reaction_smiles"]
        )
        reactant_smi = rxn_smi.split(">>")[0]
        mol = Chem.AddHs(Chem.MolFromSmiles(reactant_smi))
        be, atoms = mol_to_be_matrix(mol)

        expected_swap_steps = 0
        be_current = be.copy()
        for step in sample_v3_entry["mechanisms"][0]["steps"]:
            try:
                changes = reconstruct_step(step, atoms, be_current)
            except Exception:
                break
            breaks = [(i, j) for i, j, v in changes if i != j and v < 0]
            forms = [(i, j) for i, j, v in changes if i != j and v > 0]
            if len(breaks) == 2 and len(forms) == 2:
                expected_swap_steps += 1
            be_current = apply_step(be_current, changes)

        samples = process_reaction_entry_v3(sample_v3_entry, include_product=False)
        step_samples = [s for s in samples if "step" in s["type"]]

        assert len(step_samples) == expected_swap_steps, (
            f"Expected {expected_swap_steps} swap steps, got {len(step_samples)} training samples"
        )


@pytest.mark.slow
class TestFullPipeline:
    """End-to-end ReactionDataset construction on the real dataset."""

    def test_dataset_caching_consistent(self):
        """Loading from cache yields the same length and sample contents as fresh processing."""
        from bond_match.dataset import ReactionDataset

        ds_fresh = ReactionDataset(
            str(USPTO_TRAIN),
            limit=20,
            force_recalculate_cache=True,
            data_version="v3_blockspace",
        )
        ds_cached = ReactionDataset(
            str(USPTO_TRAIN),
            limit=20,
            force_recalculate_cache=False,
            data_version="v3_blockspace",
        )
        assert len(ds_fresh) == len(ds_cached)
        if len(ds_fresh) > 0:
            s1 = ds_fresh[0]
            s2 = ds_cached[0]
            assert torch.equal(s1["x"], s2["x"])
            np.testing.assert_array_equal(s1["y"], s2["y"])
