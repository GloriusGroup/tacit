"""CPU forward-pass tests for a minimal BondGAT model and gradient checks for
the pair ranking losses computed on its output."""
import torch

from bond_match.model import BondGAT
from bond_match.losses.ranking_loss import PairRankingLoss, EBMPairRankingLoss


def _tiny_model():
    """Construct a minimal BondGAT configuration for CPU testing."""
    return BondGAT(
        d_model=16,
        heads=2,
        num_layers=1,
        atom_max_num=20,
        num_bond_types=7,
        transformer_layers=1,
        dropout=0.0,
    )


def _tiny_inputs():
    """Build inputs for a batch of two molecules, each with two atoms and one bond."""
    x = torch.tensor(
        [[6, 0, 0, 0, 0.0], [6, 0, 0, 0, 0.0], [8, 0, 1, 0, 0.0], [7, 0, 1, 0, 0.0]],
        dtype=torch.float32,
    )
    batch_idx = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    edge_attr = torch.tensor([[1], [1], [1], [1]])
    bond_indices = torch.tensor([[0, 1], [2, 3]])
    step_index = torch.tensor([0, 0])
    return x, batch_idx, edge_index, edge_attr, bond_indices, step_index


def test_forward_output_keys_and_shapes():
    """Forward pass returns embeddings, binary logits, and stop logits with one stop logit per molecule and one embedding per scored bond."""
    model = _tiny_model().eval()
    x, bidx, ei, ea, bi, si = _tiny_inputs()
    with torch.no_grad():
        out = model(x, bidx, ei, ea, bi, step_index=si)
    for key in ("embeddings", "logits_binary", "logits_stop"):
        assert key in out
    assert out["logits_binary"].shape == (2, 1)
    assert out["logits_stop"].shape == (2, 1)
    assert out["embeddings"].shape[0] == 2


def test_forward_requires_step_index():
    """Forward pass raises an AssertionError when step_index is None."""
    model = _tiny_model().eval()
    x, bidx, ei, ea, bi, _ = _tiny_inputs()
    with torch.no_grad():
        try:
            model(x, bidx, ei, ea, bi, step_index=None)
            raised = False
        except AssertionError:
            raised = True
    assert raised


def test_forward_is_deterministic_in_eval():
    """Two forward passes in eval mode produce identical binary logits."""
    model = _tiny_model().eval()
    args = _tiny_inputs()
    with torch.no_grad():
        a = model(*args[:5], step_index=args[5])["logits_binary"]
        b = model(*args[:5], step_index=args[5])["logits_binary"]
    assert torch.allclose(a, b)


def _loss_inputs(model):
    """Run a forward pass and assemble loss inputs in which the single bond in each molecule is labeled reactive."""
    x, bidx, ei, ea, bi, si = _tiny_inputs()
    out = model(x, bidx, ei, ea, bi, step_index=si)
    out["model_ref"] = model
    bond_batch_vec = bidx[bi[:, 0]]
    targets = torch.tensor([1, 1])
    is_over = torch.tensor([False, False])
    return out, targets, bond_batch_vec, is_over


def test_pair_ranking_loss_is_scalar_with_gradients():
    """PairRankingLoss returns a scalar that backpropagates gradients to model parameters."""
    model = _tiny_model().train()
    loss = PairRankingLoss(temperature=0.1)(*_loss_inputs(model))
    assert loss.dim() == 0 and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_ebm_ranking_loss_is_scalar_with_gradients():
    """EBMPairRankingLoss returns a scalar that backpropagates gradients to model parameters."""
    model = _tiny_model().train()
    loss = EBMPairRankingLoss(lambda_anchor=0.05)(*_loss_inputs(model))
    assert loss.dim() == 0 and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
