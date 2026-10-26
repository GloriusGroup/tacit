"""Unit tests for reusable NN building blocks: LoRA, mHC, scorers, heads,
and the transformer block. All run on CPU with tiny tensors."""
import torch
import torch.nn as nn

from bond_match.modules.lora import (
    LoRALinear,
    apply_lora_to_model,
    freeze_non_lora_parameters,
    collect_lora_stats,
    checkpoint_has_lora_weights,
)
from bond_match.modules.mhc import (
    _doubly_stochastic_project,
    StreamManager,
    HyperConnections,
)
from bond_match.model.scorers import AttentionPairScorer, BondPairScorer
from bond_match.model.heads import (
    AtomSwapTypeHead,
    MLPSwapTypeHead,
    NodeAttentionSwapHead,
    AttentionPoolVirtualHead,
)
from bond_match.model.transformer import BondTransformerBlock


def test_lora_is_identity_at_init_and_freezes_base():
    """LoRALinear matches the base layer at initialization, since lora_B is
    zero-initialized and the delta vanishes; base weights are frozen while
    lora_A and lora_B remain trainable."""
    base = nn.Linear(8, 4)
    lora = LoRALinear.from_linear(base, rank=4, alpha=8.0, dropout=0.0)
    x = torch.randn(3, 8)
    assert torch.allclose(base(x), lora(x), atol=1e-6)
    assert not lora.weight.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_apply_lora_and_freeze():
    """apply_lora_to_model replaces only modules matching the name patterns;
    freeze_non_lora_parameters leaves only LoRA parameters trainable; and
    collect_lora_stats reports the replacement and parameter counts."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_a = nn.Linear(8, 8)
            self.proj_b = nn.Linear(8, 4)
            self.classifier = nn.Linear(4, 2)

    net = Net()
    replaced = apply_lora_to_model(net, ["proj"], rank=4, alpha=8.0, dropout=0.0)
    assert len(replaced) == 2 and not any("classifier" in r for r in replaced)
    freeze_non_lora_parameters(net)
    for name, p in net.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            assert p.requires_grad
        else:
            assert not p.requires_grad
    stats = collect_lora_stats(net)
    assert stats.replaced_modules == 2
    assert stats.trainable_params < stats.total_params


def test_checkpoint_has_lora_weights():
    """checkpoint_has_lora_weights detects LoRA keys in a state dict."""
    assert checkpoint_has_lora_weights({"x.lora_A": torch.zeros(1)}) is True
    assert checkpoint_has_lora_weights({"x.weight": torch.zeros(1)}) is False


def test_doubly_stochastic_projection_sums_to_one():
    """The iterative Sinkhorn-style projection yields a non-negative matrix
    whose rows and columns sum to approximately one."""
    torch.manual_seed(0)
    logits = torch.randn(4, 4)
    M = _doubly_stochastic_project(logits, n_iters=50)
    assert torch.allclose(M.sum(-1), torch.ones(4), atol=1e-2)
    assert torch.allclose(M.sum(-2), torch.ones(4), atol=1e-2)
    assert (M >= 0).all()


def test_stream_manager_expand_reduce_roundtrip():
    """expand multiplies the batch dimension by the stream count and reduce
    restores it."""
    sm = StreamManager(num_streams=2)
    x = torch.randn(3, 5)
    expanded = sm.expand(x)
    assert expanded.shape == (6, 5)
    assert sm.reduce(expanded).shape == (3, 5)


def test_hyper_connections_shapes():
    """width_conn and depth_conn produce the expected shapes for
    stream-expanded inputs."""
    B, S, T, D = 2, 2, 4, 16
    mhc = HyperConnections(num_streams=S, d_model=D)
    x = torch.randn(B * S, T, D)
    branch_input, residuals = mhc.width_conn(x)
    assert branch_input.shape == (B, T, D)
    assert residuals.shape == (B, S, T, D)
    out = mhc.depth_conn(torch.randn(B, T, D), residuals)
    assert out.shape == (B * S, T, D)


def test_attention_pair_scorer_symmetry_and_upper_triangle():
    """A symmetric AttentionPairScorer is invariant to argument order, and
    score_upper_triangle returns n*(n-1)/2 scores."""
    scorer = AttentionPairScorer(
        d_model=16, num_heads=2, use_symmetric_pair_scorer=True
    ).eval()
    zi, zj = torch.randn(5, 16), torch.randn(5, 16)
    with torch.no_grad():
        s = scorer(zi, zj)
        assert s.shape == (5, 1)
        assert torch.allclose(s, scorer(zj, zi), atol=1e-5)
        ut = scorer.score_upper_triangle(torch.randn(6, 16))
        assert ut.shape == (6 * 5 // 2,)


def test_bond_pair_scorer_symmetry():
    """A symmetric BondPairScorer is invariant to argument order."""
    scorer = BondPairScorer(input_dim=16, use_symmetric_pair_scorer=True).eval()
    zi, zj = torch.randn(4, 16), torch.randn(4, 16)
    with torch.no_grad():
        assert scorer(zi, zj).shape == (4, 1)
        assert torch.allclose(scorer(zi, zj), scorer(zj, zi), atol=1e-5)


def test_swap_type_heads_output_shapes():
    """AtomSwapTypeHead and MLPSwapTypeHead output 3 classes
    (stable/swap0/swap1); NodeAttentionSwapHead outputs 2 classes
    (swap0/swap1)."""
    a = AtomSwapTypeHead(d_model=16, dropout=0.0).eval()
    atoms = [torch.randn(2, 16) for _ in range(4)]
    assert a(*atoms).shape == (2, 3)

    m = MLPSwapTypeHead(d_model=16).eval()
    assert m(torch.randn(2, 16), torch.randn(2, 16)).shape == (2, 3)

    n = NodeAttentionSwapHead(d_model=16, num_heads=2, dropout=0.0).eval()
    assert n(*atoms).shape == (2, 2)


def test_attention_pool_virtual_head_output_shape():
    """AttentionPoolVirtualHead outputs one logit per molecule."""
    head = AttentionPoolVirtualHead(d_model=16, num_heads=2).eval()
    bond_context = torch.randn(4, 16)
    bond_batch_vec = torch.tensor([0, 0, 1, 1])
    step_index = torch.tensor([0, 1])
    out = head(bond_context, bond_batch_vec, step_index)
    assert out.shape == (2, 1)


def test_bond_transformer_block_preserves_shape():
    """BondTransformerBlock with additive residuals preserves the input shape."""
    blk = BondTransformerBlock(
        d_model=16, nhead=2, dim_feedforward=32, dropout=0.0, residual_type="add"
    ).eval()
    x = torch.randn(2, 5, 16)
    assert blk(x).shape == (2, 5, 16)
