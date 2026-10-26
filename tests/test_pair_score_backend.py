"""Tests that the exact_attention_chunked pair-score backend matches the baseline backend."""
import copy

import torch

from bond_match.model import BondGAT


def _make_model() -> BondGAT:
    """Construct a small BondGAT configured with the baseline pair-score backend."""
    return BondGAT(
        d_model=32,
        heads=4,
        num_layers=1,
        transformer_layers=1,
        atom_max_num=100,
        num_bond_types=7,
        dropout=0.0,
        return_type_prediction=False,
        use_attention_pair_scorer=True,
        use_symmetric_pair_scorer=True,
        pair_score_backend="legacy",
        pair_score_chunk_size=3,
    ).eval()


def _legacy_scores(model: BondGAT, z: torch.Tensor, binary_logits: torch.Tensor, mode: str):
    """Compute upper-triangle pair scores through compute_pair_scores."""
    triu = torch.triu_indices(z.size(0), z.size(0), offset=1)
    scores = model.compute_pair_scores(
        z[triu[0]],
        z[triu[1]],
        binary_logits[triu[0]],
        binary_logits[triu[1]],
        mode=mode,
    )
    return scores, triu


@torch.no_grad()
def test_exact_attention_chunked_backend_matches_legacy():
    """The exact_attention_chunked backend reproduces baseline pair scores and
    upper-triangle indices for the pair_only, fused, and learned_fusion modes."""
    torch.manual_seed(0)
    legacy = _make_model()
    exact = copy.deepcopy(legacy)
    exact._pair_score_backend = "exact_attention_chunked"
    exact._pair_score_chunk_size = 3

    z = torch.randn(11, 32)
    binary_logits = torch.randn(11, 1)

    for mode in ("pair_only", "fused", "learned_fusion"):
        legacy_scores, legacy_triu = _legacy_scores(legacy, z, binary_logits, mode)
        exact_scores, exact_triu = exact.compute_pair_scores_upper_triangle(
            z, binary_logits=binary_logits, mode=mode
        )

        assert torch.equal(legacy_triu, exact_triu)
        assert torch.allclose(legacy_scores, exact_scores, atol=1e-6, rtol=1e-6)
