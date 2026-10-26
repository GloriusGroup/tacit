"""Verify single-GPU vs DDP (2 ranks) produce equivalent:
  1. Weighted sampling distributions (oversampling ratios)
  2. Global weighted sample sequences
  3. Gradient magnitudes (with and without gradient accumulation)
  4. Seed collision avoidance
  5. Epoch variation across samplers

Run: cd ~/bond_match && .venv/bin/python -m pytest tests/test_ddp_weighting.py -v
"""

import math
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from bond_match.sampler import WeightedDistributedSampler
from bond_match.dataset import ShardAwareSampler


class _MockShardedDataset:
    """Minimal mock satisfying ShardAwareSampler's interface."""

    def __init__(self, n: int, shard_size: int = 500):
        self._n = n
        self.shard_size = shard_size
        self.n_shards = math.ceil(n / shard_size)

    def __len__(self):
        return self._n

    def storage_index(self, idx: int) -> int:
        return idx


def _build_weights(
    n, photo_frac=0.01, multi_frac=0.10, photo_factor=5.0, multi_factor=3.0
):
    n_photo = int(photo_frac * n)
    n_multi = int(multi_frac * n)
    weights, labels = [], []
    for i in range(n):
        if i < n_photo:
            weights.append(photo_factor)
            labels.append("photo")
        elif i < n_photo + n_multi:
            weights.append(multi_factor)
            labels.append("multi")
        else:
            weights.append(1.0)
            labels.append("normal")
    return weights, labels


def _class_fractions(selected_indices, labels):
    counts = Counter(labels[i] for i in selected_indices)
    total = len(selected_indices)
    return {c: counts.get(c, 0) / total for c in ("photo", "multi", "normal")}


def _interleave_rank_sequences(rank_sequences):
    out = []
    max_len = max(len(seq) for seq in rank_sequences)
    for pos in range(max_len):
        for seq in rank_sequences:
            if pos < len(seq):
                out.append(seq[pos])
    return out


def test_sampling_distribution():
    """ShardAwareSampler (1 GPU) vs WeightedDistributedSampler (2 ranks)
    should produce the same per-class oversampling ratios."""
    N = 10_000
    weights, labels = _build_weights(N)
    ds = _MockShardedDataset(N, shard_size=500)
    n_trials = 40

    single_fracs = {"photo": [], "multi": [], "normal": []}
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(trial)
        sampler = ShardAwareSampler(
            ds, weights=weights, num_samples=len(weights), generator=g
        )
        fracs = _class_fractions(list(sampler), labels)
        for c in single_fracs:
            single_fracs[c].append(fracs[c])

    ddp_fracs = {"photo": [], "multi": [], "normal": []}
    for trial in range(n_trials):
        all_indices = []
        for rank in range(2):
            sampler = WeightedDistributedSampler(
                ds,
                weights,
                num_replicas=2,
                rank=rank,
                seed=trial,
                shard_size=ds.shard_size,
            )
            sampler.set_epoch(0)
            all_indices.extend(list(sampler))
        fracs = _class_fractions(all_indices, labels)
        for c in ddp_fracs:
            ddp_fracs[c].append(fracs[c])

    for c in ("photo", "multi", "normal"):
        s_mean = np.mean(single_fracs[c])
        d_mean = np.mean(ddp_fracs[c])
        delta = abs(s_mean - d_mean)
        assert delta < 0.015, (
            f"{c}: single-GPU={s_mean:.4f} vs DDP={d_mean:.4f} differ by {delta:.4f}"
        )


def test_ddp_reconstructs_single_gpu_global_sequence():
    """Interleaving all DDP ranks should recover the single-GPU sample sequence.

    DDP draws ``total_size`` samples to make the epoch evenly divisible across ranks,
    so we compare against a single-GPU sampler configured with the same number of draws.
    """
    N = 2_001
    weights, _ = _build_weights(N)
    ds = _MockShardedDataset(N, shard_size=500)
    seed = 42
    epoch = 3

    ddp_rank0 = WeightedDistributedSampler(
        ds,
        weights,
        num_replicas=2,
        rank=0,
        seed=seed,
        shard_size=ds.shard_size,
    )
    ddp_rank0.set_epoch(epoch)
    total_size = ddp_rank0.total_size

    single = ShardAwareSampler(ds, weights=weights, num_samples=total_size, seed=seed)
    single.set_epoch(epoch)
    single_seq = list(single)

    rank_sequences = []
    for rank in range(2):
        sampler = WeightedDistributedSampler(
            ds,
            weights,
            num_replicas=2,
            rank=rank,
            seed=seed,
            shard_size=ds.shard_size,
        )
        sampler.set_epoch(epoch)
        rank_sequences.append(list(sampler))

    reconstructed = _interleave_rank_sequences(rank_sequences)
    assert reconstructed == single_seq


def test_epoch_variation():
    """Both samplers should produce different samples across epochs."""
    N = 2_000
    weights, _ = _build_weights(N)
    ds = _MockShardedDataset(N, shard_size=500)

    g = torch.Generator().manual_seed(42)
    sampler = ShardAwareSampler(ds, weights=weights, num_samples=N, generator=g)
    epoch0 = list(sampler)
    epoch1 = list(sampler)
    assert epoch0 != epoch1, (
        "ShardAwareSampler returned identical sequences across epochs"
    )

    sampler = WeightedDistributedSampler(
        ds, weights, num_replicas=2, rank=0, seed=42, shard_size=ds.shard_size
    )
    sampler.set_epoch(0)
    ep0 = list(sampler)
    sampler.set_epoch(1)
    ep1 = list(sampler)
    assert ep0 != ep1, (
        "WeightedDistributedSampler returned identical sequences across epochs"
    )


def test_gradient_single_vs_ddp():
    """Single-GPU (batch=2B) should match simulated DDP (2 ranks, batch=B each)."""
    B = 8
    torch.manual_seed(0)
    model_s = nn.Linear(16, 1, bias=False)
    x_all = torch.randn(2 * B, 16)
    loss_s = nn.functional.mse_loss(model_s(x_all), torch.zeros(2 * B, 1))
    loss_s.backward()
    grad_single = model_s.weight.grad.clone()

    torch.manual_seed(0)
    model_d = nn.Linear(16, 1, bias=False)
    x_all = torch.randn(2 * B, 16)

    model_d.zero_grad()
    nn.functional.mse_loss(model_d(x_all[:B]), torch.zeros(B, 1)).backward()
    grad_r0 = model_d.weight.grad.clone()

    model_d.zero_grad()
    nn.functional.mse_loss(model_d(x_all[B:]), torch.zeros(B, 1)).backward()
    grad_r1 = model_d.weight.grad.clone()

    grad_ddp = (grad_r0 + grad_r1) / 2
    assert (grad_single - grad_ddp).abs().max().item() < 1e-6


def test_gradient_accum():
    """batch=2B should match batch=B with accum_steps=2."""
    B = 8
    torch.manual_seed(0)
    model_s = nn.Linear(16, 1, bias=False)
    x_all = torch.randn(2 * B, 16)
    nn.functional.mse_loss(model_s(x_all), torch.zeros(2 * B, 1)).backward()
    grad_single = model_s.weight.grad.clone()

    torch.manual_seed(0)
    model_a = nn.Linear(16, 1, bias=False)
    x_all = torch.randn(2 * B, 16)
    model_a.zero_grad()
    for step in range(2):
        x_step = x_all[step * B : (step + 1) * B]
        (nn.functional.mse_loss(model_a(x_step), torch.zeros(B, 1)) / 2).backward()
    grad_accum = model_a.weight.grad.clone()

    assert (grad_single - grad_accum).abs().max().item() < 1e-6


def test_gradient_ddp_plus_accum():
    """DDP (2 ranks) + accum_steps=2 should match single-GPU batch=4B."""
    B, accum = 4, 2
    total = 2 * accum * B

    torch.manual_seed(0)
    model_s = nn.Linear(16, 1, bias=False)
    x_all = torch.randn(total, 16)
    nn.functional.mse_loss(model_s(x_all), torch.zeros(total, 1)).backward()
    grad_single = model_s.weight.grad.clone()

    torch.manual_seed(0)
    model_d = nn.Linear(16, 1, bias=False)
    x_all = torch.randn(total, 16)
    rank_grads = []
    for rank in range(2):
        model_d.zero_grad()
        for step in range(accum):
            idx = rank * accum * B + step * B
            x_step = x_all[idx : idx + B]
            (
                nn.functional.mse_loss(model_d(x_step), torch.zeros(B, 1)) / accum
            ).backward()
        rank_grads.append(model_d.weight.grad.clone())

    grad_ddp_accum = (rank_grads[0] + rank_grads[1]) / 2
    assert (grad_single - grad_ddp_accum).abs().max().item() < 1e-6


def test_seed_no_collision():
    """New seeding formula avoids cross-(epoch, rank) collisions."""
    num_replicas = 4
    seed = 0

    seen = set()
    for epoch in range(100):
        for rank in range(num_replicas):
            s = seed + epoch * num_replicas + rank
            assert s not in seen, f"Collision: epoch={epoch}, rank={rank}"
            seen.add(s)

    old_seen = set()
    collisions = 0
    for epoch in range(100):
        for rank in range(num_replicas):
            s = seed + epoch + rank
            if s in old_seen:
                collisions += 1
            old_seen.add(s)
    assert collisions > 0, "Old formula should have collisions"


def test_ddp_rank_lengths_match_total_size():
    """DDP ranks should partition the global sequence without dropping draws."""
    N = 1_001
    weights = [1.0] * N
    ds = _MockShardedDataset(N)

    samplers = [
        WeightedDistributedSampler(
            ds,
            weights,
            num_replicas=2,
            rank=rank,
            seed=42,
            shard_size=ds.shard_size,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(0)

    rank_sequences = [list(sampler) for sampler in samplers]
    reconstructed = _interleave_rank_sequences(rank_sequences)
    assert len(reconstructed) == samplers[0].total_size
