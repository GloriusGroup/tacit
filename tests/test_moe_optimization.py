"""Equivalence and benchmark tests for the batched MoE expert forward against
the sequential reference implementation."""

import time
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bond_match.modules.sparse_moe import MoE, Expert


def _sequential_expert_forward(moe, x):
    """Reference sequential expert forward used for equivalence checks."""
    experts = moe.experts.experts
    B, E, C, D = x.shape
    outputs = torch.zeros_like(x)
    for e_idx in range(E):
        outputs[:, e_idx] = experts[e_idx](x[:, e_idx])
    return outputs


def test_batched_expert_equivalence():
    """Batched expert output must match sequential expert output in fp32."""
    torch.manual_seed(42)
    dim = 384
    num_experts = 8

    moe = MoE(dim=dim, num_experts=num_experts, gating_top_n=2,
              expert_hidden_mult=1)
    moe.eval()

    expert_inputs = torch.randn(4, num_experts, 20, dim)

    with torch.no_grad():
        seq_out = _sequential_expert_forward(moe, expert_inputs)
        bat_out = moe._batched_expert_forward(expert_inputs)

    max_diff = (seq_out - bat_out).abs().max().item()
    assert torch.allclose(seq_out, bat_out, atol=1e-5, rtol=1e-4), \
        f"Max diff: {max_diff}"
    print(f"  PASS: batched matches sequential (max diff={max_diff:.2e})")


def test_batched_expert_equivalence_bf16():
    """Batched expert output must match sequential output under bfloat16
    autocast within an absolute tolerance of 0.01; bf16 carries roughly three
    significant digits and the batched and sequential matmul kernels
    accumulate in different orders. Skipped without CUDA."""
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA for bf16 test")
        return

    torch.manual_seed(42)
    dim = 384
    num_experts = 8
    device = torch.device('cuda')

    moe = MoE(dim=dim, num_experts=num_experts, gating_top_n=2,
              expert_hidden_mult=1).to(device)
    moe.eval()

    expert_inputs = torch.randn(4, num_experts, 20, dim, device=device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        seq_out = _sequential_expert_forward(moe, expert_inputs)
        bat_out = moe._batched_expert_forward(expert_inputs)

    max_diff = (seq_out - bat_out).abs().max().item()
    assert max_diff < 0.01, f"bf16 max diff too large: {max_diff}"
    print(f"  PASS: batched matches sequential in bf16 (max diff={max_diff:.2e})")


def test_gradient_flow():
    """All expert parameters and the input must receive non-zero gradients
    through the full MoE forward."""
    torch.manual_seed(42)
    dim = 128
    num_experts = 4

    moe = MoE(dim=dim, num_experts=num_experts, gating_top_n=2,
              expert_hidden_mult=1)
    moe.train()

    x = torch.randn(2, 10, dim, requires_grad=True)
    result = moe(x)
    loss = result.outputs.sum() + result.total_aux_loss
    loss.backward()

    for i, expert in enumerate(moe.experts.experts):
        for name, param in expert.named_parameters():
            assert param.grad is not None, f"Expert {i} param {name}: no gradient"
            assert param.grad.abs().sum() > 0, f"Expert {i} param {name}: zero gradient"

    assert x.grad is not None, "Input gradient is None"
    print("  PASS: all expert parameters receive non-zero gradients")


def test_full_moe_shapes():
    """Full MoE forward preserves input shape and produces finite outputs and
    auxiliary loss."""
    torch.manual_seed(42)
    dim = 256
    num_experts = 8

    moe = MoE(dim=dim, num_experts=num_experts, gating_top_n=4,
              expert_hidden_mult=1)
    moe.eval()

    x = torch.randn(8, 30, dim)
    result = moe(x)

    assert result.outputs.shape == x.shape, \
        f"Shape mismatch: {result.outputs.shape} vs {x.shape}"
    assert torch.isfinite(result.outputs).all(), "Non-finite values in output"
    assert torch.isfinite(result.total_aux_loss), "Non-finite aux loss"
    print("  PASS: shapes and finiteness OK")


def test_gradient_equivalence():
    """Input gradients from the batched expert path must match the sequential
    path."""
    torch.manual_seed(42)
    dim = 128
    num_experts = 4

    moe = MoE(dim=dim, num_experts=num_experts, gating_top_n=2,
              expert_hidden_mult=1)
    moe.eval()

    expert_inputs = torch.randn(2, num_experts, 10, dim, requires_grad=True)

    expert_inputs_seq = expert_inputs.detach().clone().requires_grad_(True)
    seq_out = _sequential_expert_forward(moe, expert_inputs_seq)
    seq_out.sum().backward()

    expert_inputs_bat = expert_inputs.detach().clone().requires_grad_(True)
    bat_out = moe._batched_expert_forward(expert_inputs_bat)
    bat_out.sum().backward()

    max_grad_diff = (expert_inputs_seq.grad - expert_inputs_bat.grad).abs().max().item()
    assert torch.allclose(expert_inputs_seq.grad, expert_inputs_bat.grad, atol=1e-5, rtol=1e-4), \
        f"Gradient max diff: {max_grad_diff}"
    print(f"  PASS: input gradients match (max diff={max_grad_diff:.2e})")


def benchmark_batched_vs_sequential():
    """Benchmark batched versus sequential expert forward, plus the full MoE
    forward including gating. Reports timings; makes no assertions."""
    torch.manual_seed(42)
    dim = 384
    num_experts = 8
    capacity = 30
    batch = 16

    moe = MoE(dim=dim, num_experts=num_experts, gating_top_n=4,
              expert_hidden_mult=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe = moe.to(device).eval()

    expert_inputs = torch.randn(batch, num_experts, capacity, dim, device=device)

    for _ in range(20):
        _sequential_expert_forward(moe, expert_inputs)
        moe._batched_expert_forward(expert_inputs)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    n_iters = 200

    start = time.perf_counter()
    for _ in range(n_iters):
        _sequential_expert_forward(moe, expert_inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seq_time = (time.perf_counter() - start) / n_iters * 1000

    start = time.perf_counter()
    for _ in range(n_iters):
        moe._batched_expert_forward(expert_inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    bat_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"  Sequential: {seq_time:.3f} ms/iter")
    print(f"  Batched:    {bat_time:.3f} ms/iter")
    print(f"  Speedup:    {seq_time/bat_time:.2f}x")

    x = torch.randn(batch, 50, dim, device=device)
    for _ in range(20):
        moe(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        moe(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    full_time = (time.perf_counter() - start) / n_iters * 1000
    print(f"  Full MoE forward: {full_time:.3f} ms/iter")


if __name__ == "__main__":
    print("=== MoE Optimization Tests ===\n")

    print("[1] Batched vs sequential equivalence (fp32):")
    test_batched_expert_equivalence()

    print("[2] Batched vs sequential equivalence (bf16):")
    test_batched_expert_equivalence_bf16()

    print("[3] Gradient flow through batched path:")
    test_gradient_flow()

    print("[4] Full MoE forward shapes:")
    test_full_moe_shapes()

    print("[5] Gradient equivalence (batched vs sequential):")
    test_gradient_equivalence()

    print("\n[6] Speed benchmark:")
    benchmark_batched_vs_sequential()

    print("\nAll tests passed.")
