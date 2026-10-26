"""Tests for token-mask handling in the sparse MoE layer."""
import torch

from bond_match.modules.sparse_moe import MoE


def test_moe_token_mask_keeps_sequence_outputs_batch_invariant():
    """With a token_mask, a sequence's MoE outputs are identical whether the
    sequence is processed alone or padded inside a larger batch."""
    torch.manual_seed(0)

    moe = MoE(
        dim=8,
        num_experts=4,
        expert_hidden_mult=2,
        gating_top_n=2,
        capacity_factor_eval=1.0,
    )
    moe.eval()

    seq_a = torch.randn(1, 4, 8)
    seq_b = torch.randn(1, 6, 8)

    single_mask = torch.ones(1, 4, dtype=torch.bool)
    batch_x = torch.zeros(2, 6, 8)
    batch_x[0, :4] = seq_a[0]
    batch_x[1] = seq_b[0]
    batch_mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]]
    )

    with torch.no_grad():
        out_single = moe(seq_a, token_mask=single_mask).outputs
        out_batch = moe(batch_x, token_mask=batch_mask).outputs

    assert torch.allclose(out_single[0], out_batch[0, :4], atol=1e-6, rtol=1e-5)
