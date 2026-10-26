"""Test that canonicalize_mol_ordering produces identical representations for equivalent SMILES."""

import numpy as np
import torch
from rdkit import Chem

from bond_match.dataset import (
    canonicalize_mol_ordering,
    prepare_v3_reactant,
    get_graph_features_nx,
    mol_to_nx,
)
from bond_match.blockspace import mol_to_be_matrix as _mol_to_be_matrix

EQUIVALENT_SMILES = [
    "O=C(O)c1ccccc1.Nc1ccccc1",
    "O=[C:1]([O:2][H:16])[c:3]1[c:4]([H:17])[c:5]([H:18])[c:6]([H:19])[c:7]([H:20])[c:8]1[H:21].[N:9]([c:10]1[c:11]([H:24])[c:12]([H:25])[c:13]([H:26])[c:14]([H:27])[c:15]1[H:28])([H:22])[H:23]",
    "Nc1ccccc1.O=C(O)c1ccccc1",
]


def _prep(smiles):
    """Run prepare_v3_reactant on a reactant-only reaction built from the given SMILES."""
    entry = {"reaction_smiles": smiles + ">>", "success": True}
    return prepare_v3_reactant(entry, extended_features=False)


def test_canonicalize_basic():
    """canonicalize_mol_ordering yields identical canonical SMILES for two component orderings of the same molecule set."""
    mol1 = Chem.AddHs(Chem.MolFromSmiles("O=C(O)c1ccccc1.Nc1ccccc1"))
    mol2 = Chem.AddHs(Chem.MolFromSmiles("Nc1ccccc1.O=C(O)c1ccccc1"))

    c1 = canonicalize_mol_ordering(mol1)
    c2 = canonicalize_mol_ordering(mol2)

    smi1 = Chem.MolToSmiles(c1)
    smi2 = Chem.MolToSmiles(c2)
    assert smi1 == smi2, f"Canonical SMILES differ: {smi1} vs {smi2}"


def test_canonicalize_preserves_atom_maps():
    """Atom-map numbers are preserved through canonicalization."""
    mol = Chem.MolFromSmiles("[C:1]([O:2])=[O:3]")
    mol = Chem.AddHs(mol)
    canon = canonicalize_mol_ordering(mol)

    maps = {a.GetAtomMapNum() for a in canon.GetAtoms() if a.GetAtomMapNum() > 0}
    assert maps == {1, 2, 3}, f"Expected maps {{1,2,3}}, got {maps}"


def test_prepare_v3_atoms_array_identical():
    """All three equivalent SMILES produce the same atomic-number sequence after canonicalization.

    atoms_array columns are [atom_map_num, atomic_num]. Atom maps differ between
    mapped and unmapped SMILES by design, so only the atomic_num column is compared.
    """
    preps = [_prep(s) for s in EQUIVALENT_SMILES]
    assert all(p is not None for p in preps), "Some SMILES failed to preprocess"

    ref_atomic_nums = preps[0]["atoms_array"][:, 1]
    for i in range(1, len(preps)):
        np.testing.assert_array_equal(
            ref_atomic_nums, preps[i]["atoms_array"][:, 1],
            err_msg=f"atomic numbers differ for SMILES[0] vs SMILES[{i}]"
        )


def test_prepare_v3_be_matrix_identical():
    """All three equivalent SMILES produce the same BE matrix after canonicalization."""
    preps = [_prep(s) for s in EQUIVALENT_SMILES]
    ref = preps[0]["be_matrix"]
    for i in range(1, len(preps)):
        np.testing.assert_array_equal(
            ref, preps[i]["be_matrix"],
            err_msg=f"be_matrix differs for SMILES[0] vs SMILES[{i}]"
        )


def test_prepare_v3_graph_features_identical():
    """All three equivalent SMILES produce identical node features, edge_index, edge_attr, and bond lists."""
    preps = [_prep(s) for s in EQUIVALENT_SMILES]

    feats = []
    for p in preps:
        x, edge_index, edge_attr, bonds, _ = get_graph_features_nx(p["G"], extended_features=False)
        feats.append((x, edge_index, edge_attr, bonds))

    ref_x, ref_ei, ref_ea, ref_bonds = feats[0]
    for i in range(1, len(feats)):
        x_i, ei_i, ea_i, bonds_i = feats[i]
        assert torch.equal(ref_x, x_i), f"x differs for SMILES[0] vs SMILES[{i}]"
        assert torch.equal(ref_ei, ei_i), f"edge_index differs for SMILES[0] vs SMILES[{i}]"
        assert torch.equal(ref_ea, ea_i), f"edge_attr differs for SMILES[0] vs SMILES[{i}]"
        assert ref_bonds == bonds_i, f"bonds differs for SMILES[0] vs SMILES[{i}]"


if __name__ == "__main__":
    test_canonicalize_basic()
    print("PASSED: test_canonicalize_basic")

    test_canonicalize_preserves_atom_maps()
    print("PASSED: test_canonicalize_preserves_atom_maps")

    test_prepare_v3_atoms_array_identical()
    print("PASSED: test_prepare_v3_atoms_array_identical")

    test_prepare_v3_be_matrix_identical()
    print("PASSED: test_prepare_v3_be_matrix_identical")

    test_prepare_v3_graph_features_identical()
    print("PASSED: test_prepare_v3_graph_features_identical")

    print("\nAll canonicalization tests passed!")
