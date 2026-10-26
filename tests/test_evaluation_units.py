"""Unit tests for pure helpers in the evaluation package (utils, be_ops, results)."""
import numpy as np

from bond_match.evaluation.utils import (
    canonicalize_smiles,
    check_smiles_match,
    remove_atom_mapping,
    _normalize_smiles,
    _rewrite_product_as_reactant,
    _maybe_rewrite_yield0_entry,
    is_entry_successful,
    _is_no_reaction_target,
)
from bond_match.evaluation.constants import _normalize_cache
from bond_match.evaluation.be_ops import apply_predicted_step_be, be_result_to_smiles
from bond_match.evaluation.results import StopReason, BEState, PairResult
from bond_match.blockspace import smiles_to_matrices


def test_canonicalize_smiles():
    """canonicalize_smiles strips atom mapping and maps empty input to an empty string."""
    assert canonicalize_smiles("[CH3:1][CH2:2][OH:3]") == "CCO"
    assert canonicalize_smiles("") == ""


def test_check_smiles_match_is_fragment_subset():
    """check_smiles_match is True only when the target fragments are a subset
    of the predicted fragments, and False for empty predictions."""
    assert check_smiles_match("CCO", "CCO") is True
    assert check_smiles_match("CCO.O", "O") is True
    assert check_smiles_match("O", "CCO.O") is False
    assert check_smiles_match("", "O") is False


def test_remove_atom_mapping():
    """remove_atom_mapping strips atom-map numbers and leaves unmapped SMILES unchanged."""
    assert remove_atom_mapping("[CH3:1][OH:2]") == "CO"
    assert remove_atom_mapping("CCO") == "CCO"


def test_normalize_smiles_is_cached():
    """_normalize_smiles stores its result in _normalize_cache and a repeated
    call returns the cached value."""
    _normalize_cache.clear()
    out = _normalize_smiles("CCO")
    assert "CCO" in _normalize_cache
    assert _normalize_smiles("CCO") == out


def test_rewrite_product_as_reactant():
    """_rewrite_product_as_reactant replaces the product with the reactant;
    strings without '>>' are returned unchanged."""
    assert _rewrite_product_as_reactant("CCO>>CCC") == "CCO>>CCO"
    assert _rewrite_product_as_reactant("CCO") == "CCO"


def test_maybe_rewrite_yield0_entry():
    """_maybe_rewrite_yield0_entry rewrites the product of a yield-zero entry
    when enabled and leaves the entry untouched when disabled."""
    entry = {"metadata": {"Yield": 0.0}, "reaction_smiles": "CCO>>CCC"}
    rewritten = _maybe_rewrite_yield0_entry(entry, enable_yield0_negatives=True)
    assert rewritten["reaction_smiles"] == "CCO>>CCO"
    kept = _maybe_rewrite_yield0_entry(entry, enable_yield0_negatives=False)
    assert kept["reaction_smiles"] == "CCO>>CCC"


def test_is_entry_successful():
    """is_entry_successful requires success=True and rejects entries with
    more than 4 steps; missing keys default to successful."""
    assert is_entry_successful({"success": True, "num_steps": 2}) is True
    assert is_entry_successful({"success": False, "num_steps": 1}) is False
    assert is_entry_successful({"num_steps": 9}) is False
    assert is_entry_successful({}) is True


def test_is_no_reaction_target():
    """_is_no_reaction_target is True only when the target equals the
    reactant; a None target is False."""
    assert _is_no_reaction_target("CCO", "CCO") is True
    assert _is_no_reaction_target("CCO", "CCC") is False
    assert _is_no_reaction_target("CCO", None) is False


def test_apply_predicted_step_conserves_electrons_and_symmetry():
    """apply_predicted_step_be on a 4-atom chain (bonds 0-1, 1-2, 2-3)
    preserves matrix shape, symmetry, and total electron count, and zeroes
    the two broken bonds (0,1) and (2,3)."""
    be = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float
    )
    new = apply_predicted_step_be(be, (0, 1), (2, 3), swap_type=0)
    assert new.shape == be.shape
    assert np.allclose(new, new.T)
    assert np.isclose(be.sum(), new.sum())
    assert new[0, 1] == 0 and new[2, 3] == 0


def test_be_result_to_smiles_roundtrip_multimolecule():
    """be_result_to_smiles recovers both fragments of a multi-molecule input.
    Canonicalization for comparison goes through Chem.MolFromSmiles followed
    by Chem.MolToSmiles because Chem.CanonSmiles is unreliable under the
    MolToSmiles patch."""
    from rdkit import Chem

    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    be, atoms = smiles_to_matrices("CCO.O", include_hydrogens=True)
    smi = be_result_to_smiles(be, atoms)
    frags = {canon(s) for s in smi.split(".")}
    assert canon("CCO") in frags
    assert canon("O") in frags


def test_stop_reason_members():
    """StopReason defines PREDICTED_IS_OVER and MAX_STEPS_REACHED members."""
    names = {s.name for s in StopReason}
    assert "PREDICTED_IS_OVER" in names
    assert "MAX_STEPS_REACHED" in names


def test_bestate_defaults():
    """BEState defaults: has_catalyst False, empty reverse_lp_map, and
    n_reactant_heavy 0."""
    bs = BEState(be_matrix=np.eye(2), atoms_identity=np.array([[1, 6], [2, 8]]))
    assert bs.has_catalyst is False
    assert bs.reverse_lp_map == {}
    assert bs.n_reactant_heavy == 0


def test_pair_result_defaults():
    """PairResult defaults the local bond indices to -1."""
    pr = PairResult(is_over=False, score=1.0, action_prob=0.5)
    assert pr.local_b1 == -1 and pr.local_b2 == -1
    assert pr.is_over is False


def test_first_step_stop_matches_only_no_reaction_targets():
    """A first-step stop is correct for an unchanged reactant set, not when the product is already a reactant."""
    from bond_match.evaluation.utils import is_no_reaction_match

    assert is_no_reaction_match("CCO.CO", "CCO.CO")
    assert not is_no_reaction_match("CCO.CO", "CCO")
    assert not is_no_reaction_match("CCO", "")


def test_load_all_eval_entries_keeps_every_entry(tmp_path):
    """Entries failing the blockspace filter are kept with their reaction SMILES and metadata only."""
    import json
    from bond_match.evaluation.data import _load_all_eval_entries

    ok = {"reaction_smiles": "CCO>>CC=O", "success": True, "num_steps": 1,
          "mechanisms": [{"steps": [{"break": [], "form": []}]}], "metadata": {"i": 0}}
    failed = {"reaction_smiles": "CCO>>CCCl", "success": False, "num_steps": 0, "mechanisms": [], "metadata": {"i": 1}}
    long = {**ok, "mechanisms": [{"steps": [{}] * 5}], "metadata": {"i": 2}}
    well = {"reaction_smiles": "CCO>>CC=O", "is_negative": True, "metadata": {"i": 3}}
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in (ok, failed, long, well)))

    data = _load_all_eval_entries(str(path))
    assert [e["metadata"]["i"] for e in data] == [0, 1, 2, 3]
    assert data[0] == ok and data[3] == well
    assert data[1] == {"reaction_smiles": "CCO>>CCCl", "metadata": {"i": 1}}
    assert data[2] == {"reaction_smiles": "CCO>>CC=O", "metadata": {"i": 2}}
