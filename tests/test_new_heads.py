"""Tests for NodeAttentionSwapHead and AttentionPoolVirtualHead, including
their integration with BondGAT."""
import torch
import pytest
from bond_match.model.heads import NodeAttentionSwapHead, AttentionPoolVirtualHead
from bond_match.model import BondGAT


class TestNodeAttentionSwapHead:
    """Unit tests for NodeAttentionSwapHead."""

    @pytest.fixture
    def head(self):
        """NodeAttentionSwapHead with dropout disabled for determinism."""
        return NodeAttentionSwapHead(d_model=64, num_heads=4, dropout=0.0)

    def test_output_shape_1d(self, head):
        """A single unbatched atom embedding yields output shape (1, 2)."""
        h = torch.randn(64)
        out = head(h, h, h, h)
        assert out.shape == (1, 2)

    def test_output_shape_batch(self, head):
        """Batched atom embeddings yield output shape (B, 2)."""
        B = 4
        h = torch.randn(B, 64)
        out = head(h, h, h, h)
        assert out.shape == (B, 2)

    def test_role_awareness(self, head):
        """Exchanging the two bonds' atom embeddings changes the output,
        reflecting the role embeddings."""
        torch.manual_seed(42)
        head.eval()
        h_u1 = torch.randn(1, 64)
        h_u2 = torch.randn(1, 64)
        h_v1 = torch.randn(1, 64)
        h_v2 = torch.randn(1, 64)

        with torch.no_grad():
            out_original = head(h_u1, h_u2, h_v1, h_v2)
            out_swapped = head(h_v1, h_v2, h_u1, h_u2)

        assert not torch.allclose(out_original, out_swapped, atol=1e-5), (
            "Swapping bond atoms should produce different outputs due to role embeddings"
        )

    def test_gradient_flow(self):
        """All four atom-embedding inputs receive non-zero gradients."""
        head = NodeAttentionSwapHead(d_model=32, num_heads=4, dropout=0.0)
        atoms = [torch.randn(2, 32, requires_grad=True) for _ in range(4)]
        out = head(*atoms)
        loss = out.sum()
        loss.backward()
        for i, a in enumerate(atoms):
            assert a.grad is not None, f"No gradient for atom {i}"
            assert a.grad.abs().sum() > 0, f"Zero gradient for atom {i}"

    def test_multi_layer(self):
        """A two-layer head preserves the (B, 2) output shape."""
        head = NodeAttentionSwapHead(d_model=64, num_heads=4, num_layers=2, dropout=0.0)
        h = torch.randn(3, 64)
        out = head(h, h, h, h)
        assert out.shape == (3, 2)


class TestAttentionPoolVirtualHead:
    """Unit tests for AttentionPoolVirtualHead."""

    @pytest.fixture
    def head(self):
        """AttentionPoolVirtualHead with dropout disabled for determinism."""
        return AttentionPoolVirtualHead(d_model=64, num_heads=4, dropout=0.0)

    def test_output_shape(self, head):
        """Pooled output has one logit per molecule."""
        bond_ctx = torch.randn(10, 64)
        batch_vec = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        step_idx = torch.tensor([0, 1])
        out = head(bond_ctx, batch_vec, step_idx)
        assert out.shape == (2, 1)

    def test_variable_lengths(self, head):
        """Molecules with different bond counts (3 and 5) pool correctly in
        one batch."""
        bond_ctx = torch.randn(8, 64)
        batch_vec = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        step_idx = torch.tensor([0, 2])
        out = head(bond_ctx, batch_vec, step_idx)
        assert out.shape == (2, 1)

    def test_step_awareness(self, head):
        """Different step indices produce different outputs."""
        head.eval()
        bond_ctx = torch.randn(5, 64)
        batch_vec = torch.zeros(5, dtype=torch.long)

        with torch.no_grad():
            out0 = head(bond_ctx, batch_vec, torch.tensor([0]))
            out3 = head(bond_ctx, batch_vec, torch.tensor([3]))

        assert not torch.allclose(out0, out3, atol=1e-5), (
            "Different step indices should produce different outputs"
        )

    def test_sparse_vs_dense(self, head):
        """The sparse path (bond_context + batch_vec) matches the path with
        precomputed dense bonds and mask."""
        from torch_geometric.utils import to_dense_batch

        head.eval()
        bond_ctx = torch.randn(8, 64)
        batch_vec = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        step_idx = torch.tensor([0, 1])

        dense_bonds, mask = to_dense_batch(bond_ctx, batch_vec)

        with torch.no_grad():
            out_sparse = head(bond_ctx, batch_vec, step_idx)
            out_dense = head(bond_ctx, batch_vec, step_idx,
                             dense_bonds=dense_bonds, mask=mask)

        assert torch.allclose(out_sparse, out_dense, atol=1e-5), (
            f"Max diff: {(out_sparse - out_dense).abs().max().item():.8f}"
        )

    def test_single_bond(self, head):
        """A molecule with a single bond yields output shape (1, 1)."""
        bond_ctx = torch.randn(1, 64)
        batch_vec = torch.zeros(1, dtype=torch.long)
        step_idx = torch.tensor([0])
        out = head(bond_ctx, batch_vec, step_idx)
        assert out.shape == (1, 1)

    def test_deep_head(self):
        """use_deep_head=True preserves the (num_molecules, 1) output shape."""
        head = AttentionPoolVirtualHead(d_model=64, num_heads=4, use_deep_head=True)
        bond_ctx = torch.randn(6, 64)
        batch_vec = torch.tensor([0, 0, 0, 1, 1, 1])
        step_idx = torch.tensor([0, 1])
        out = head(bond_ctx, batch_vec, step_idx)
        assert out.shape == (2, 1)


def _make_graph(n_atoms=8, n_bonds=6, batch_size=1, device="cpu"):
    """Create a minimal valid chain-graph input for BondGAT."""
    x = torch.randint(0, 30, (n_atoms * batch_size, 7), device=device)
    batch_idx = torch.arange(batch_size, device=device).repeat_interleave(n_atoms)
    src, dst = [], []
    for b in range(batch_size):
        off = b * n_atoms
        for i in range(n_atoms - 1):
            src.extend([off + i, off + i + 1])
            dst.extend([off + i + 1, off + i])
    edge_index = torch.tensor([src, dst], device=device)
    edge_attr = torch.randint(0, 5, (edge_index.size(1),), device=device)
    bonds = []
    for b in range(batch_size):
        off = b * n_atoms
        for i in range(n_bonds):
            bonds.append([off + i, off + i + 1])
    bond_indices = torch.tensor(bonds, device=device)
    is_virtual = torch.zeros(n_atoms * batch_size, dtype=torch.long, device=device)
    step_index = torch.zeros(batch_size, dtype=torch.long, device=device)
    return x, batch_idx, edge_index, edge_attr, bond_indices, is_virtual, step_index


class TestBondGATIntegration:
    """Integration tests for the new heads within BondGAT."""

    def test_node_attn_swap_forward(self):
        """BondGAT with swap_type_head='node_attn' runs forward, returns the
        expected output keys, and installs NodeAttentionSwapHead."""
        torch.manual_seed(42)
        model = BondGAT(
            d_model=64, heads=4, num_layers=2, transformer_layers=2,
            use_pair_bond_type=True, return_type_prediction=True,
            swap_type_head="node_attn",
        ).eval()
        x, bi, ei, ea, bonds, iv, si = _make_graph()
        with torch.no_grad():
            out = model(x, bi, ei, ea, bonds, iv, step_index=si)

        assert "logits_virtual" in out
        assert "logits_binary" in out
        assert "embeddings" in out
        assert isinstance(model.pair_bond_type_classifier, NodeAttentionSwapHead)

    def test_attn_pool_virtual_forward(self):
        """BondGAT with use_attention_pool_virtual=True produces (1, 1)
        virtual logits via AttentionPoolVirtualHead."""
        torch.manual_seed(42)
        model = BondGAT(
            d_model=64, heads=4, num_layers=2, transformer_layers=2,
            return_type_prediction=True,
            use_attention_pool_virtual=True,
        ).eval()
        x, bi, ei, ea, bonds, iv, si = _make_graph()
        with torch.no_grad():
            out = model(x, bi, ei, ea, bonds, iv, step_index=si)

        assert out["logits_virtual"].shape == (1, 1)
        assert isinstance(model.attention_pool_virtual, AttentionPoolVirtualHead)

    def test_both_features_combined(self):
        """Both head options can be enabled together in a single forward pass."""
        torch.manual_seed(42)
        model = BondGAT(
            d_model=64, heads=4, num_layers=2, transformer_layers=2,
            use_pair_bond_type=True, return_type_prediction=True,
            swap_type_head="node_attn",
            use_attention_pool_virtual=True,
        ).eval()
        x, bi, ei, ea, bonds, iv, si = _make_graph()
        with torch.no_grad():
            out = model(x, bi, ei, ea, bonds, iv, step_index=si)

        assert out["logits_virtual"].shape == (1, 1)
        assert isinstance(model.pair_bond_type_classifier, NodeAttentionSwapHead)
        assert isinstance(model.attention_pool_virtual, AttentionPoolVirtualHead)
