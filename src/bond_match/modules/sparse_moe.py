from __future__ import annotations

from functools import partial
from collections import namedtuple
from typing import Tuple

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F

from beartype import beartype

import einx
from einops import rearrange, repeat, reduce, pack, unpack

from colt5_attention import topk as maybe_differentiable_topk

import torch.distributed as dist

from st_moe_pytorch.distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    pad_dim_to,
    has_only_one_value
)

MIN_EXPERT_CAPACITY = 4

MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
    'total_aux_loss',
    'balance_loss',
    'router_z_loss'
])

def exists(val):
    return val is not None

def default(val, default):
    if exists(val):
        return val

    return default() if callable(default) else default

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cast_tuple(el, len = 1):
    return el if isinstance(el, tuple) else ((el,) * len)

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# ``torch.nn.functional.one_hot`` requires all indices to be within range.

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias = True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x * self.mult_bias

class Expert(Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False,
        dropout = 0.0
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)

        self.net = Sequential(
            RMSNorm(dim) if prenorm else None,
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden, mult_bias = mult_bias),
            nn.Dropout(dropout) if dropout > 0 else None,
            nn.Linear(dim_hidden, dim)
        )

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        return self.net(x)

class Experts(Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        allow_var_seq_len = False # whether to handle variable sequence length
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = ModuleList(experts)

        # Distributed execution settings.

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        self.all_gather = AllGather()

        self.allow_var_seq_len = allow_var_seq_len

        # Device tracker for moving inactive distributed experts to CPU.

        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts
        seq_len = shape[-2]

        world_size = 1
        rank = 0

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            var_seq_len = not has_only_one_value(seq_sizes)

            assert self.allow_var_seq_len or not var_seq_len, 'number of tokens per expert must be the same - if you want the framework to handle it, set `allow_var_seq_len = True` on `Experts`'

            # Pad variable-length sequences before collective communication.

            if var_seq_len:
                max_seq_size = seq_sizes.amax().item()
                x = pad_dim_to(x, max_seq_size, dim = -2)

            # Gather and concatenate batches while preserving variable batch sizes.

            x, batch_sizes = self.all_gather(x)
            total_batch_size = batch_sizes.sum().item()

            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # Experts assigned to this rank.

        num_experts_per_rank = num_experts
        expert_slice = slice(0, num_experts)

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim = -1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # Ranks without an assigned expert receive empty batch slices.

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)

        # In distributed mode, each rank handles a subset of experts and batches.

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')

            x = x.split(num_experts_batches_across_ranks, dim = 0)
            x, experts_per_rank_sizes = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # Route tokens to assigned experts and skip empty inputs during evaluation.

        outs = []

        for expert, expert_input in zip(experts, x):
            # Skip experts that received no tokens (all-zero dispatch)
            if expert_input.abs().sum() == 0 and not self.training:
                outs.append(torch.zeros_like(expert_input))
            else:
                out = expert(expert_input)
                outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x, requires_grad = self.training)

        # Gather across merged expert-batch dimensions, then restore batch layout.

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs, sizes = experts_per_rank_sizes)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs, _ = split_by_rank(outs)

            outs = outs[..., :seq_len, :]

        assert outs.shape == shape
        return outs

# Routing implementation adapted from Tensor2Tensor MoE.
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

class TopNGating(Module):

    @beartype
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        top_n = 2,
        threshold_train: float | Tuple[float, ...] = 0.2,
        threshold_eval: float | Tuple[float, ...] = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.to_gates = nn.Linear(dim, num_gates, bias = False)

        self.differentiable_topk = differentiable_topk

        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable = not differentiable_topk,
            fused = differentiable_topk_fused # use triton fused coordinate descent if possible by default
        )

        assert top_n >= 2, 'must be 2 or more experts'
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval        

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    def forward(
        self,
        x,
        token_mask = None,
        noise_gates = False,
        noise_mult = 1.
    ):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        c - capacity
        """

        *_, b, group_size, dim, dtype, top_n, num_gates, eps = *x.shape, x.dtype, self.top_n, self.num_gates, self.eps

        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Expert capacity bounds the number of positions assigned per sequence.
        # A static capacity dimension is required for expert batch sizes.

        if token_mask is None:
            valid_counts = torch.full(
                (b,), group_size, device=x.device, dtype=torch.long
            )
        else:
            valid_counts = token_mask.to(x.device).sum(dim=-1)

        expert_capacity_per_seq = torch.div(
            valid_counts * capacity_factor,
            num_gates,
            rounding_mode='floor',
        )
        expert_capacity_per_seq = torch.minimum(expert_capacity_per_seq, valid_counts)
        expert_capacity_per_seq = torch.clamp(
            expert_capacity_per_seq, min=MIN_EXPERT_CAPACITY
        )
        expert_capacity = int(expert_capacity_per_seq.max().item())
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_limit = expert_capacity_per_seq.to(x.dtype).view(b, 1, 1)

        # gate logits and gates

        gate_logits = self.to_gates(x)

        maybe_noised_gate_logits = gate_logits

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim = -1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k = top_n)

        gate_indices = topk_return.indices

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        # move top-n dimension first: (..., k) -> (k, ...)
        gates = gates.movedim(-1, 0)
        gate_indices = gate_indices.movedim(-1, 0)

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)
        mask = one_hot_gate_indices.float()

        if token_mask is not None:
            token_mask = token_mask.to(mask.device).bool()
            token_mask_f = token_mask.unsqueeze(0).unsqueeze(-1).to(mask.dtype)
            mask = mask * token_mask_f

        mask_1 = mask[0]

        denom = gates.sum(dim=0, keepdim=True).clamp(min=eps)
        gates = gates / denom

        if self.training:
            probs = torch.zeros_like(gates).uniform_(0., 1.)
            should_route = probs < (gates / threshold.clamp(min=eps).view(-1, 1, 1))
            should_route[0, ...] = True
            mask *= should_route.float().unsqueeze(-1)

        mask_cumsum = cumsum_exclusive(mask, dim=-2)

        positions = []
        prev_expert_count = 0.

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]
            mask[n] *= (position_in_expert < expert_capacity_limit).float()
            prev_expert_count = mask[n].sum(dim=-2, keepdim=True) + prev_expert_count
            position_in_expert = position_in_expert.sum(dim=-1)
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        mask_flat = mask.sum(dim=-1)    # (k, b, n)
        gates = gates * mask_flat

        # combine tensor via native broadcasting (avoids einx 5D intermediate)
        pos_one_hot = F.one_hot(
            positions.long().clamp(min=0, max=expert_capacity - 1), expert_capacity
        )                                                        # (k, b, n, c)

        combine_tensor = (
            (gates * mask_flat)[..., None, None] *               # (k, b, n, 1, 1)
            one_hot_gate_indices.unsqueeze(-1) *                 # (k, b, n, e, 1)
            pos_one_hot.unsqueeze(-2)                            # (k, b, n, 1, c)
        ).sum(dim=0)                                             # (b, n, e, c)

        dispatch_tensor = combine_tensor.bool().to(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses
        if self.training:
            density_1 = mask_1.mean(dim=-2)
            density_1_proxy = raw_gates.mean(dim=-2)
            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # router z-loss
        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim=-1)
            router_z_loss = torch.square(router_z_loss)
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss

class MoE(Module):

    @beartype
    def __init__(self,
        dim,
        num_experts = 16,
        expert_hidden_mult = 4,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        gating_top_n = 2,
        balance_loss_coef = 1e-2,
        router_z_loss_coef = 1e-3,
        experts: Module | None = None,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        is_distributed = None,
        allow_var_seq_len = False,
        expert_dim: int | None = None,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # DeepSeek-style: experts operate at a smaller dimension with projections
        expert_dim = default(expert_dim, dim)
        self.expert_dim = expert_dim
        self.has_proj = expert_dim != dim
        if self.has_proj:
            self.down_proj = nn.Linear(dim, expert_dim, bias=False)
            self.up_proj = nn.Linear(expert_dim, dim, bias=False)

        self.gate = TopNGating(
            dim,
            top_n = gating_top_n,
            num_gates = num_experts,
            straight_through_dispatch_tensor = straight_through_dispatch_tensor,
            differentiable_topk = differentiable_topk,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        experts = default(experts, lambda: [Expert(dim = expert_dim, hidden_mult = expert_hidden_mult, dropout = expert_dropout) for _ in range(num_experts)])

        self.experts = Experts(
            experts,
            is_distributed = is_distributed,
            allow_var_seq_len = allow_var_seq_len
        )

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def _get_expert_linears(self):
        """Find the two Linear layers and GEGLU in expert nets (handles optional dropout)."""
        e0 = self.experts.experts[0]
        linears = [m for m in e0.net if isinstance(m, nn.Linear)]
        geglus = [m for m in e0.net if isinstance(m, GEGLU)]
        dropouts = [m for m in e0.net if isinstance(m, nn.Dropout)]
        # Index of each layer type in the Sequential
        lin_indices = [i for i, m in enumerate(e0.net) if isinstance(m, nn.Linear)]
        geglu_indices = [i for i, m in enumerate(e0.net) if isinstance(m, GEGLU)]
        return lin_indices[0], geglu_indices[0], lin_indices[1], len(dropouts) > 0

    def _batched_expert_forward(self, x):
        """All experts in parallel via batched GEMM (replaces sequential for-loop).

        Reshapes to (E, B*C, D) so each expert gets a contiguous (B*C, D) block,
        then uses torch.bmm for E simultaneous GEMMs. This is ~3x faster than
        the sequential loop and avoids the poor cache locality of matmul broadcast.
        Gradients flow back through torch.stack to the Expert parameters.
        """
        experts = self.experts.experts
        B, E, C, D = x.shape

        lin1_idx, geglu_idx, lin2_idx, has_dropout = self._get_expert_linears()

        w1 = torch.stack([e.net[lin1_idx].weight for e in experts])   # (E, 2H, D)
        b1 = torch.stack([e.net[lin1_idx].bias for e in experts])     # (E, 2H)
        w2 = torch.stack([e.net[lin2_idx].weight for e in experts])   # (E, D_out, H)
        b2 = torch.stack([e.net[lin2_idx].bias for e in experts])     # (E, D_out)
        geglu_mult = torch.stack([
            e.net[geglu_idx].mult_bias if isinstance(e.net[geglu_idx].mult_bias, torch.Tensor)
            else torch.ones(w2.shape[-1], device=x.device, dtype=x.dtype)
            for e in experts
        ])                                                      # (E, H)

        # Reshape to expert-major contiguous layout for efficient batched GEMM
        x_e = x.permute(1, 0, 2, 3).contiguous().view(E, B * C, D)

        # Linear 1: (E, B*C, D) @ (E, D, 2H) -> (E, B*C, 2H)
        h = torch.bmm(x_e, w1.mT) + b1[:, None, :]
        x_half, gate = h.chunk(2, dim=-1)
        h = F.gelu(gate) * x_half * geglu_mult[:, None, :]
        if has_dropout and self.training:
            dropout_mod = experts[0].net[lin2_idx - 1]  # Dropout is right before final Linear
            h = dropout_mod(h)
        # Linear 2: (E, B*C, H) @ (E, H, D) -> (E, B*C, D)
        out = torch.bmm(h, w2.mT) + b2[:, None, :]

        # Reshape back to (B, E, C, D)
        return out.view(E, B, C, D).permute(1, 0, 2, 3)

    def forward(
        self,
        x,
        token_mask = None,
        noise_gates = False,
        noise_mult = 1.,
        group_ids = None,
    ):
        # When tokens from multiple states are packed into one row (trajectory
        # attention), per-row capacity in the gate would couple MoE drops to
        # trajectory length and cause train/eval drift. Reshape to per-state
        # batch (one state = one row) so capacity is per-state. Output is
        # scattered back to the input (B, N, D) layout, so block residuals see
        # no shape change.
        if group_ids is not None and token_mask is not None:
            B_traj, T_max, D = x.shape
            valid_flat = token_mask.reshape(-1)
            x_flat = x.reshape(-1, D)
            groups_flat = group_ids.reshape(-1)

            x_valid = x_flat[valid_flat]
            groups_valid = groups_flat[valid_flat]

            from torch_geometric.utils import to_dense_batch

            # Product-history dropout can alter step ordering, so do not assume
            # packed trajectory tokens are sorted by state id here.
            sort_idx = torch.argsort(groups_valid, stable=True)
            unsort_idx = torch.empty_like(sort_idx)
            unsort_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
            x_sorted = x_valid[sort_idx]
            groups_sorted = groups_valid[sort_idx]

            # Remap to dense 0..N_states-1 (handles non-contiguous state IDs).
            _, inv = torch.unique(groups_sorted, sorted=True, return_inverse=True)
            x_state, mask_state = to_dense_batch(x_sorted, inv)

            inner = self.forward(
                x_state,
                token_mask=mask_state,
                noise_gates=noise_gates,
                noise_mult=noise_mult,
                group_ids=None,
            )
            out_valid = inner.outputs[mask_state][unsort_idx]

            # Scatter back into the input layout using x's dtype to avoid
            # autocast bf16/fp32 assignment mismatches.
            out_flat = torch.zeros_like(x_flat)
            out_flat[valid_flat] = out_valid.to(x_flat.dtype)
            out = out_flat.view(B_traj, T_max, D)
            return MixtureOfExpertsReturn(
                out, inner.total_aux_loss, inner.balance_loss, inner.router_z_loss
            )

        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(
            x,
            token_mask=token_mask,
            noise_gates=noise_gates,
            noise_mult=noise_mult,
        )

        # dispatch (at full dim for routing)
        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)

        # project down to expert_dim if needed
        if self.has_proj:
            B, E, C, D = expert_inputs.shape
            expert_inputs = self.down_proj(expert_inputs.reshape(B * E * C, D)).reshape(B, E, C, self.expert_dim)

        # batched expert forward — all experts in one shot via batched matmul
        expert_outputs = self._batched_expert_forward(expert_inputs)

        # project back up to full dim if needed
        if self.has_proj:
            B, E, C, _ = expert_outputs.shape
            expert_outputs = self.up_proj(expert_outputs.reshape(B * E * C, self.expert_dim)).reshape(B, E, C, self.dim)

        # combine
        output = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)

        # losses
        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef
        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        return MixtureOfExpertsReturn(output, total_aux_loss, balance_loss, router_z_loss)


class SparseMoEBlock(Module):

    @beartype
    def __init__(
        self,
        moe: MoE,
        *,
        add_ff_before = False,
        add_ff_after = True,
        residual_type: str = "add",
        num_streams: int = 2,
        block_idx: int = 0,
        total_blocks: int = 1,
    ):
        super().__init__()
        dim = moe.dim

        self.moe = moe
        self.moe_prenorm = RMSNorm(dim)

        # ff_before/ff_after operate at full dim; derive hidden mult from experts
        expert_dim = moe.expert_dim
        expert_hidden_mult = moe.experts.experts[0].net[0].weight.shape[0] * 3 / (expert_dim * 4)

        self.ff_before = Expert(dim, hidden_mult=expert_hidden_mult, prenorm = True) if add_ff_before else None
        self.ff_after = Expert(dim, hidden_mult=expert_hidden_mult, prenorm = True) if add_ff_after else None

        # mHC support: wrap the MoE sublayer with hyper-connections
        self._residual_type = residual_type
        if residual_type == "mhc":
            from bond_match.modules.mhc import HyperConnections
            # MoE gets its own HC, ff_after gets its own HC
            total_sublayers = total_blocks * 2  # moe + ff_after per block
            base = block_idx * 2
            self.hc_moe = HyperConnections(num_streams, dim, layer_idx=base, total_layers=total_sublayers)
            if add_ff_after:
                self.hc_ff_after = HyperConnections(num_streams, dim, layer_idx=base + 1, total_layers=total_sublayers)

        # AttnRes support: per-sublayer attention residual connections
        if residual_type in ("full_attnres", "block_attnres"):
            from bond_match.modules.attention_residuals import AttnResConnection
            self.ar_moe = AttnResConnection(dim)
            if add_ff_after:
                self.ar_ff_after = AttnResConnection(dim)

    def forward(
        self,
        x,
        token_mask = None,
        noise_gates = False,
        noise_mult = 1.,
        group_ids = None,
    ):

        # feedforward before (no mHC — this is a simple stabilization FF)
        if exists(self.ff_before):
            x = self.ff_before(x) + x

        if self._residual_type == "mhc":
            return self._forward_mhc(x, token_mask, noise_gates, noise_mult, group_ids)
        return self._forward_add(x, token_mask, noise_gates, noise_mult, group_ids)

    def _forward_add(self, x, token_mask, noise_gates, noise_mult, group_ids=None):
        # mixture of experts layer
        residual = x
        moe_out, total_aux_loss, balance_loss, router_z_loss = self.moe(
            self.moe_prenorm(x),
            token_mask=token_mask,
            noise_gates=noise_gates,
            noise_mult=noise_mult,
            group_ids=group_ids,
        )
        x = moe_out + residual

        # feedforward after
        if exists(self.ff_after):
            x = self.ff_after(x) + x

        return MixtureOfExpertsReturn(x, total_aux_loss, balance_loss, router_z_loss)

    def _forward_mhc(self, x, token_mask, noise_gates, noise_mult, group_ids=None):
        # mHC MoE: width_conn -> prenorm -> moe -> depth_conn
        branch_in, residuals_mixed = self.hc_moe.width_conn(x)
        moe_out, total_aux_loss, balance_loss, router_z_loss = self.moe(
            self.moe_prenorm(branch_in),
            token_mask=token_mask,
            noise_gates=noise_gates,
            noise_mult=noise_mult,
            group_ids=group_ids,
        )
        x = self.hc_moe.depth_conn(moe_out, residuals_mixed)

        # mHC FF after
        if exists(self.ff_after):
            branch_in, residuals_mixed = self.hc_ff_after.width_conn(x)
            ff_out = self.ff_after(branch_in)
            x = self.hc_ff_after.depth_conn(ff_out, residuals_mixed)

        return MixtureOfExpertsReturn(x, total_aux_loss, balance_loss, router_z_loss)

    def _forward_attnres(self, state, token_mask=None, noise_gates=False, noise_mult=1., group_ids=None):
        """AttnRes path: aggregate from state, run MoE + ff_after, accumulate."""
        h = self.ar_moe.pre_sublayer(state)
        moe_out, total_aux_loss, balance_loss, router_z_loss = self.moe(
            self.moe_prenorm(h),
            token_mask=token_mask,
            noise_gates=noise_gates,
            noise_mult=noise_mult,
            group_ids=group_ids,
        )
        self.ar_moe.post_sublayer(state, moe_out)

        if exists(self.ff_after):
            h = self.ar_ff_after.pre_sublayer(state)
            ff_out = self.ff_after(h)
            self.ar_ff_after.post_sublayer(state, ff_out)

        return MixtureOfExpertsReturn(None, total_aux_loss, balance_loss, router_z_loss)
