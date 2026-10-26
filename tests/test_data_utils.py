"""Unit tests for dataset_util, catalyst_utils, and sampler."""
import pytest
import torch
from rdkit import Chem

from bond_match.dataset_util import (
    remove_amap_not_in_product,
    remap_rxn_smi,
    canonicalize,
)
from bond_match.catalyst_utils import fix_catalyst_smiles
from bond_match.sampler import weighted_shard_aware_sample, WeightedDistributedSampler


def _canon(smi):
    """Canonicalize a SMILES string via an explicit Mol round-trip; Chem.CanonSmiles is unusable because bond_match monkeypatches Chem.MolToSmiles with a single-positional-argument variant."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def test_remove_amap_not_in_product_keeps_product_and_maps_all_reactant_atoms():
    """remove_amap_not_in_product leaves the product side unchanged and assigns a nonzero atom map to every reactant atom."""
    rxn = "[CH3:1][CH2:2]>>[CH4:1]"
    out = remove_amap_not_in_product(rxn)
    react, prod = out.split(">>")
    assert prod == "[CH4:1]"
    rmol = Chem.MolFromSmiles(react)
    assert all(a.GetAtomMapNum() > 0 for a in rmol.GetAtoms())


def test_remove_amap_is_idempotent():
    """Applying remove_amap_not_in_product twice gives the same result as applying it once."""
    rxn = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    once = remove_amap_not_in_product(rxn)
    twice = remove_amap_not_in_product(once)
    assert once == twice


def test_remap_rxn_smi_returns_smiles_and_correspondence():
    """remap_rxn_smi returns a parseable reaction SMILES and a correspondence dict, with both reaction halves valid."""
    new_rxn, corr = remap_rxn_smi("[CH3:1][OH:2]>>[CH3:1][OH:2]")
    assert isinstance(new_rxn, str) and new_rxn.count(">>") == 1
    assert isinstance(corr, dict)
    react, prod = new_rxn.split(">>")
    assert Chem.MolFromSmiles(react) is not None
    assert Chem.MolFromSmiles(prod) is not None


def test_canonicalize_strips_atom_maps():
    """canonicalize removes all atom-map numbers and preserves the underlying structure."""
    out = canonicalize("[CH3:1][OH:2]")
    mol = Chem.MolFromSmiles(out)
    assert mol is not None
    assert all(a.GetAtomMapNum() == 0 for a in mol.GetAtoms())
    assert _canon(out) == _canon("CO")


def test_fix_catalyst_passthrough_valid():
    """fix_catalyst_smiles passes a valid SMILES through structurally unchanged."""
    out = fix_catalyst_smiles("CCO")
    assert out is not None and Chem.MolFromSmiles(out) is not None
    assert _canon(out) == _canon("CCO")


def test_fix_catalyst_is_lru_cached():
    """Repeated calls with identical input return the same cached object."""
    assert fix_catalyst_smiles("CCO") is fix_catalyst_smiles("CCO")


def test_fix_catalyst_never_returns_invalid_smiles():
    """fix_catalyst_smiles returns either None or a SMILES that RDKit can parse."""
    for s in ("CCO", "c1ccccc1", "totally-not-smiles", "@@@"):
        out = fix_catalyst_smiles(s)
        assert out is None or Chem.MolFromSmiles(out) is not None


def test_weighted_sample_count_and_bias():
    """weighted_shard_aware_sample returns the requested number of in-range indices, with the heavily weighted index sampled most often."""
    g = torch.Generator().manual_seed(0)
    idx = weighted_shard_aware_sample(list(range(3)), [10.0, 1.0, 1.0], 500, g, 0)
    assert isinstance(idx, list) and len(idx) == 500
    assert all(0 <= i < 3 for i in idx)
    assert idx.count(0) > idx.count(1)
    assert idx.count(0) > idx.count(2)


def test_weighted_sample_shard_mode_in_range():
    """Shard-aware sampling with shard_size=2 returns the requested count of in-range indices."""
    g = torch.Generator().manual_seed(1)
    idx = weighted_shard_aware_sample(list(range(4)), [1.0] * 4, 200, g, shard_size=2)
    assert len(idx) == 200
    assert all(0 <= i < 4 for i in idx)


def test_weighted_sample_zero_weights_rejected():
    """Sampling with all-zero weights raises RuntimeError or ValueError."""
    g = torch.Generator().manual_seed(2)
    with pytest.raises((RuntimeError, ValueError)):
        weighted_shard_aware_sample(list(range(3)), [0.0, 0.0, 0.0], 10, g, 0)


def test_weighted_distributed_sampler_partitions_indices():
    """WeightedDistributedSampler yields non-empty, in-range index sets for each of two replicas."""
    n = 20
    weights = [1.0] * n
    s0 = WeightedDistributedSampler(
        list(range(n)), weights, num_replicas=2, rank=0, seed=7
    )
    s1 = WeightedDistributedSampler(
        list(range(n)), weights, num_replicas=2, rank=1, seed=7
    )
    i0, i1 = list(s0), list(s1)
    assert len(i0) > 0 and len(i1) > 0
    assert all(0 <= i < n for i in i0 + i1)
