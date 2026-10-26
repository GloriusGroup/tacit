"""Distributed samplers with weighted sampling support."""

import torch
from torch.utils.data.distributed import DistributedSampler


def _storage_index(dataset, idx: int) -> int:
    if hasattr(dataset, "storage_index"):
        return dataset.storage_index(idx)
    return idx


def weighted_shard_aware_sample(
    dataset,
    weights,
    num_samples: int,
    generator: torch.Generator | None,
    shard_size: int = 0,
) -> list[int]:
    """Sample indices with replacement using global shard-aware weighted sampling.

    This mirrors the single-GPU weighted sampler logic: allocate draws across shards
    proportional to total shard weight, shuffle shard order, then sample within each
    shard according to per-sample weights.
    """
    if num_samples <= 0:
        return []

    weights = torch.as_tensor(weights, dtype=torch.float64)
    n_total = len(weights)
    if n_total == 0:
        return []

    if shard_size <= 0:
        chosen = torch.multinomial(
            weights, num_samples, replacement=True, generator=generator
        )
        return chosen.tolist()

    n_shards = max(getattr(dataset, "n_shards", 0), 1)
    shard_indices = [[] for _ in range(n_shards)]
    for idx in range(n_total):
        shard_id = _storage_index(dataset, idx) // shard_size
        if shard_id >= len(shard_indices):
            shard_indices.extend([[] for _ in range(shard_id + 1 - len(shard_indices))])
        shard_indices[shard_id].append(idx)

    shard_weights = []
    shard_totals = []
    for indices in shard_indices:
        if indices:
            w = weights[indices]
            shard_weights.append(w)
            shard_totals.append(w.sum())
        else:
            shard_weights.append(torch.empty(0, dtype=torch.float64))
            shard_totals.append(torch.tensor(0.0, dtype=torch.float64))

    shard_totals_t = torch.stack(shard_totals)
    total_weight = shard_totals_t.sum()
    if total_weight <= 0:
        raise ValueError("All sampling weights are zero")
    shard_probs = shard_totals_t / total_weight

    samples_per_shard = (shard_probs * num_samples).long()
    remainder = num_samples - samples_per_shard.sum().item()
    if remainder > 0:
        _, top_idx = torch.topk(shard_probs, min(remainder, len(shard_indices)))
        for idx in top_idx.tolist():
            samples_per_shard[idx] += 1

    shard_order = torch.randperm(len(shard_indices), generator=generator).tolist()
    sampled = []
    for shard_id in shard_order:
        n_draws = samples_per_shard[shard_id].item()
        if n_draws <= 0:
            continue
        indices = shard_indices[shard_id]
        local_weights = shard_weights[shard_id]
        chosen = torch.multinomial(
            local_weights, n_draws, replacement=True, generator=generator
        )
        sampled.extend(indices[c.item()] for c in chosen)

    return sampled


class WeightedDistributedSampler(DistributedSampler):
    """DDP sampler that mirrors the single-GPU shard-aware weighted sampler.

    A global weighted shard-aware sample sequence is drawn first, then the
    sequence is partitioned across ranks in the same strided fashion as
    ``DistributedSampler``. This keeps DDP sampling aligned with the single-GPU
    weighted training distribution instead of re-sampling independently per rank.
    """

    def __init__(
        self, dataset, weights, num_replicas=None, rank=None, seed=0, shard_size=0
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed
        )
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.shard_size = shard_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        global_indices = weighted_shard_aware_sample(
            self.dataset,
            self.weights,
            self.total_size,
            generator=g,
            shard_size=self.shard_size,
        )
        assert len(global_indices) == self.total_size

        rank_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        assert len(rank_indices) == self.num_samples
        return iter(rank_indices)
