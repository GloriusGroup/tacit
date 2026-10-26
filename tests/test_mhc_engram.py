"""Tests for the mHC (Hyper-Connections) module and its use in BondTransformerBlock."""

import torch
import pytest

from bond_match.modules.mhc import HyperConnections, StreamManager, _doubly_stochastic_project
from bond_match.model import BondGAT, BondTransformerBlock


def _make_graph(n_atoms=10, batch_size=2, d_model=64):
    """Construct a minimal two-molecule batched graph input dictionary."""
    x = torch.zeros(n_atoms, 5, dtype=torch.long)
    x[:, 0] = torch.randint(0, 50, (n_atoms,))
    x[:, 2] = torch.randint(0, 5, (n_atoms,))
    x[:, 3] = torch.randint(0, 2, (n_atoms,))

    batch_idx = torch.zeros(n_atoms, dtype=torch.long)
    half = n_atoms // 2
    batch_idx[half:] = 1

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, half, half + 1, half + 1, half + 2, half + 2, half + 3],
         [1, 0, 2, 1, 3, 2, half + 1, half, half + 2, half + 1, half + 3, half + 2]]
    )
    edge_attr = torch.randint(1, 5, (edge_index.size(1), 1))

    bond_indices_0 = torch.tensor([[0, 1], [1, 2], [2, 3]])
    bond_indices_1 = torch.tensor([[half, half + 1], [half + 1, half + 2], [half + 2, half + 3]])
    bond_indices = torch.cat([bond_indices_0, bond_indices_1], dim=0)

    is_virtual = torch.zeros(batch_size, dtype=torch.long)
    step_index = torch.zeros(batch_size, dtype=torch.long)

    return {
        "x": x, "batch_idx": batch_idx, "edge_index": edge_index,
        "edge_attr": edge_attr, "bond_indices": bond_indices,
        "is_virtual": is_virtual, "step_index": step_index,
        "batch_size": batch_size, "d_model": d_model, "n_atoms": n_atoms,
    }


class TestDoublyStochastic:
    """Tests for _doubly_stochastic_project."""

    def test_doubly_stochastic(self):
        """Projection yields a non-negative matrix whose rows and columns sum
        to approximately one."""
        logits = torch.randn(4, 4) * 0.5
        ds = _doubly_stochastic_project(logits, n_iters=10)
        assert ds.shape == (4, 4)
        assert torch.allclose(ds.sum(dim=0), torch.ones(4), atol=0.05)
        assert torch.allclose(ds.sum(dim=1), torch.ones(4), atol=0.05)
        assert (ds >= -1e-6).all()

    def test_identity_init(self):
        """Strongly diagonal logits project to approximately the identity matrix."""
        logits = torch.full((3, 3), -8.0)
        logits.fill_diagonal_(0.0)
        ds = _doubly_stochastic_project(logits, n_iters=10)
        assert torch.allclose(ds, torch.eye(3), atol=0.15)


class TestHyperConnections:
    """Tests for HyperConnections width and depth connections."""

    def test_shapes(self):
        """width_conn and depth_conn produce the expected shapes for
        stream-expanded inputs."""
        S, B, T, D = 2, 4, 10, 64
        hc = HyperConnections(num_streams=S, d_model=D)
        x = torch.randn(B * S, T, D)

        branch_in, residuals_mixed = hc.width_conn(x)
        assert branch_in.shape == (B, T, D)
        assert residuals_mixed.shape == (B, S, T, D)

        branch_out = torch.randn(B, T, D)
        out = hc.depth_conn(branch_out, residuals_mixed)
        assert out.shape == (B * S, T, D)

    def test_gradient_flow(self):
        """Gradients reach the input and the mixing-matrix logits. With
        num_streams=2 the scalar H_res_logit parameter is used instead of the
        matrix H_res_logits."""
        S, B, T, D = 2, 3, 8, 32
        hc = HyperConnections(num_streams=S, d_model=D)
        x = torch.randn(B * S, T, D, requires_grad=True)

        branch_in, residuals_mixed = hc.width_conn(x)
        branch_out = branch_in * 2
        out = hc.depth_conn(branch_out, residuals_mixed)
        out.sum().backward()

        assert x.grad is not None
        assert hc.H_res_logit.grad is not None
        assert hc.H_pre_logits.grad is not None
        assert hc.H_post_logits.grad is not None


class TestStreamManager:
    """Tests for StreamManager expand and reduce operations."""

    def test_expand_reduce(self):
        """reduce(expand(x)) sums the S identical stream copies, yielding x * S."""
        S = 3
        sm = StreamManager(S)
        x = torch.randn(4, 10, 64)
        expanded = sm.expand(x)
        assert expanded.shape == (12, 10, 64)

        reduced = sm.reduce(expanded)
        assert reduced.shape == (4, 10, 64)
        assert torch.allclose(reduced, x * S)

    def test_mask_expand(self):
        """expand interleaves stream copies per sample; for S=2 the layout is
        [mask[0], mask[0], mask[1], mask[1]]."""
        S = 2
        sm = StreamManager(S)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        expanded = sm.expand(mask)
        assert expanded.shape == (4, 3)
        assert (expanded[0] == mask[0]).all()
        assert (expanded[1] == mask[0]).all()
        assert (expanded[2] == mask[1]).all()
        assert (expanded[3] == mask[1]).all()


class TestBondTransformerBlockMHC:
    """Tests for BondTransformerBlock with mHC and additive residuals."""

    def test_forward_shape(self):
        """An mHC residual block preserves the (B*S, T, D) input shape."""
        S, B, T, D = 2, 4, 10, 64
        block = BondTransformerBlock(
            D, nhead=2, dim_feedforward=128, dropout=0.0,
            enable_cross_attention=False, residual_type="mhc",
            num_streams=S, block_idx=0, total_blocks=1,
        )
        x = torch.randn(B * S, T, D)
        mask = torch.zeros(B * S, T, dtype=torch.bool)
        out = block(x, src_key_padding_mask=mask)
        assert out.shape == (B * S, T, D)

    def test_add_unchanged(self):
        """An additive residual block preserves the (B, T, D) input shape."""
        B, T, D = 4, 10, 64
        block = BondTransformerBlock(
            D, nhead=2, dim_feedforward=128, dropout=0.0,
            residual_type="add",
        )
        x = torch.randn(B, T, D)
        out = block(x)
        assert out.shape == (B, T, D)
