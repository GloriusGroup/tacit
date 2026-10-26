"""Unit tests for dataset constants, metadata, mol-graph conversion, collation."""
import numpy as np
import torch
from rdkit import Chem

from bond_match.dataset.constants import (
    BOND_TYPE_MAP,
    BOND_ORDER_TO_TYPE,
    NUM_BOND_TYPES,
    ATOM_MAX_NUM,
)
from bond_match.dataset.metadata import (
    SampleMetadata,
    _coerce_sample_metadata,
    _dataset_cache_prefix,
    _extract_entry_yield,
)
from bond_match.dataset.mol_graph import (
    mol_to_nx,
    nx_to_mol,
    get_graph_features_nx,
    smiles_match,
    canonicalize_mol_ordering,
    parse_v3_atom_label,
    _get_atomic_num,
)
from bond_match.dataset.collate import collate_fn, trajectory_collate_fn


def test_bond_type_constants():
    """Bond-type maps assign the expected codes, including aromatic (order 1.5 -> type 4) and lone-pair edges (order 7.0 -> type 6)."""
    assert BOND_TYPE_MAP[Chem.BondType.SINGLE] == 1
    assert BOND_TYPE_MAP[Chem.BondType.DATIVE] == 1
    assert BOND_TYPE_MAP[Chem.BondType.DOUBLE] == 2
    assert BOND_TYPE_MAP[Chem.BondType.AROMATIC] == 4
    assert BOND_ORDER_TO_TYPE[1.0] == 1
    assert BOND_ORDER_TO_TYPE[1.5] == 4
    assert BOND_ORDER_TO_TYPE[7.0] == 6
    assert NUM_BOND_TYPES == 7
    assert ATOM_MAX_NUM == 100


def test_sample_metadata_dict_like_access():
    """SampleMetadata supports subscript access and dict-style get with defaults."""
    m = SampleMetadata(type="step", has_catalyst=True, step_index=2)
    assert m["type"] == "step"
    assert m.get("has_catalyst") is True
    assert m.get("missing", "default") == "default"


def test_coerce_sample_metadata():
    """_coerce_sample_metadata converts a dict to SampleMetadata and returns an existing SampleMetadata unchanged."""
    src = {"type": "product", "has_catalyst": False, "step_index": 1}
    coerced = _coerce_sample_metadata(src)
    assert isinstance(coerced, SampleMetadata)
    assert coerced.type == "product"
    assert _coerce_sample_metadata(coerced) is coerced


def test_dataset_cache_prefix_uses_path_tail():
    """_dataset_cache_prefix joins the path components with underscores and drops the extension."""
    prefix = _dataset_cache_prefix("/a/b/mit_v1/train.json")
    assert prefix == "a_b_mit_v1_train"


def test_extract_entry_yield():
    """_extract_entry_yield returns the numeric yield and None for non-numeric or missing values."""
    assert _extract_entry_yield({"metadata": {"Yield": 0.75}}) == 0.75
    assert _extract_entry_yield({"metadata": {"Yield": "not-a-number"}}) is None
    assert _extract_entry_yield({}) is None


def test_mol_to_nx_adds_head_node_and_roundtrips():
    """mol_to_nx adds one virtual head node (4 nodes for the 3 heavy atoms of ethanol) and nx_to_mol with remove_virtual recovers the original atom count."""
    mol = Chem.MolFromSmiles("CCO")
    G = mol_to_nx(mol)
    assert G.number_of_nodes() == 4
    back = nx_to_mol(G, remove_virtual=True)
    assert back.GetNumAtoms() == 3


def test_get_graph_features_shapes():
    """get_graph_features_nx returns one feature row per node, consistent edge tensor shapes, and a duplicate-free bond list."""
    G = mol_to_nx(Chem.MolFromSmiles("CCO"))
    x, edge_index, edge_attr, bonds, edge_map = get_graph_features_nx(G)
    assert x.shape[0] == G.number_of_nodes()
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_attr.shape[0]
    assert len(bonds) == len(set(bonds))


def test_smiles_match():
    """smiles_match identifies identical molecules regardless of SMILES ordering and rejects different or unparseable inputs."""
    assert smiles_match("CC(C)C", "CC(C)C") is True
    assert smiles_match("CCO", "OCC") is True
    assert smiles_match("CC", "CCC") is False
    assert smiles_match("not_smiles", "CC") is False


def test_canonicalize_mol_ordering_is_order_invariant():
    """canonicalize_mol_ordering yields the same canonical SMILES for two atom orderings of one molecule."""
    a = canonicalize_mol_ordering(Chem.MolFromSmiles("OCC"))
    b = canonicalize_mol_ordering(Chem.MolFromSmiles("CCO"))
    assert Chem.MolToSmiles(a) == Chem.MolToSmiles(b)


def test_parse_v3_atom_label():
    """parse_v3_atom_label extracts the element symbol and map number from a v3 atom label."""
    assert parse_v3_atom_label("C:15")["map"] == 15
    assert parse_v3_atom_label("C:15")["element"] == "C"


def test_get_atomic_num():
    """_get_atomic_num maps element symbols to atomic numbers."""
    assert _get_atomic_num("C") == 6
    assert _get_atomic_num("O") == 8
    assert _get_atomic_num("N") == 7


def _make_sample(smiles):
    """Build a single-step training sample dict from a SMILES string."""
    G = mol_to_nx(Chem.MolFromSmiles(smiles))
    x, ei, ea, bonds, _ = get_graph_features_nx(G)
    return {
        "x": x,
        "edge_index": ei,
        "edge_attr": ea,
        "y": torch.zeros(len(bonds), dtype=torch.long),
        "bonds": bonds,
        "is_over_target": False,
        "is_negative_reaction": False,
        "step_index": 0,
        "type": "step",
    }


def test_collate_fn_concatenates_and_offsets_nodes():
    """collate_fn concatenates node features across samples and assigns batch indices 0 and 1 to the two graphs."""
    s1, s2 = _make_sample("CCO"), _make_sample("CCO")
    out = collate_fn([s1, s2])
    x_batch, y_batch, batch_idx = out[0], out[1], out[2]
    assert x_batch.shape[0] == s1["x"].shape[0] + s2["x"].shape[0]
    assert set(batch_idx.tolist()) == {0, 1}


def test_collate_matrix_output_is_symmetric():
    """With enable_matrix_output, collate_fn returns a 3-D (batch, max_bonds, max_bonds) target matrix that is symmetric; the first bond is marked reactive so the matrix is nonzero."""
    s = _make_sample("CCO")
    s["y"][0] = 1
    out = collate_fn([s], enable_matrix_output=True)
    y_matrix = out[1]
    assert y_matrix.dim() == 3
    assert torch.equal(y_matrix[0], y_matrix[0].transpose(0, 1))


def test_trajectory_collate_appends_trajectory_ids():
    """trajectory_collate_fn returns 12 elements, one more than collate_fn, with trajectory ids appended."""
    traj = {"states": [_make_sample("CCO"), _make_sample("CCO")], "type": "trajectory"}
    out = trajectory_collate_fn([traj])
    assert len(out) == 12
