"""Attention-residual connections as a replacement for additive residuals.

Replaces fixed-weight residual accumulation (h_l = h_{l-1} + f(h_{l-1})) with
softmax attention over preceding sublayer outputs, so each sublayer aggregates
earlier representations with learned, input-dependent weights.

Two variants are provided:
  Full:  attention over all previous sublayer outputs (memory grows with depth).
  Block: sublayers grouped into blocks; attention is over block-level
         representations.

The interface mirrors the hyper-connection modules so the residual strategy can
be selected by configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class AttnResVariant(Enum):
    """Attention-residual aggregation variant (full or block)."""

    FULL = "full_attnres"
    BLOCK = "block_attnres"


def _rms_norm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm over the last dimension via the fused F.rms_norm kernel.

    Args:
        x: Input tensor.
        weight: Per-feature scale, shape (x.size(-1),).
        eps: Numerical-stability epsilon.

    Returns:
        Normalized tensor of the same shape as x.
    """
    return F.rms_norm(x, [x.size(-1)], weight=weight, eps=eps)


def _attnres_aggregate(sources: list[torch.Tensor], query: torch.Tensor,
                       norm_weight: torch.Tensor, norm_eps: float) -> torch.Tensor:
    """Aggregate sources by softmax attention against a learned query.

    Sources are stacked along a leading dimension, RMS-normalized to form keys,
    and weighted by query-key dot products. The two-source case uses an
    equivalent sigmoid interpolation.

    Args:
        sources: List of N tensors, each shape (B, T, d).
        query: Query vector, shape (d,).
        norm_weight: RMSNorm scale, shape (d,).
        norm_eps: RMSNorm epsilon.

    Returns:
        Aggregated tensor, shape (B, T, d).
    """
    n = len(sources)
    if n == 1:
        return sources[0]

    V = torch.stack(sources, dim=0)
    K = F.rms_norm(V, [V.size(-1)], weight=norm_weight, eps=norm_eps)

    if n == 2:
        diff = (query * (K[1] - K[0])).sum(-1, keepdim=True)
        alpha1 = torch.sigmoid(diff)
        s0, s1 = sources[0], sources[1]
        return s0 + alpha1 * (s1 - s0)

    logits = (query * K).sum(-1)
    alpha = logits.softmax(dim=0).unsqueeze(-1)
    return (alpha * V).sum(dim=0)


class _RMSNorm(nn.Module):
    """RMSNorm applied to attention-residual keys.

    Normalizing keys removes magnitude bias between sources accumulated over
    different depths.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        """Initialize the per-feature scale.

        Args:
            d: Feature dimension.
            eps: Numerical-stability epsilon.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm over the last dimension of x."""
        return _rms_norm_fused(x, self.weight, self.eps)


class AttnResState:
    """Mutable state passed between transformer blocks for Attention Residuals.

    Full AttnRes:  maintains ``sources`` — list of all sublayer outputs.
    Block AttnRes: maintains ``completed_blocks`` (finalized block representations)
                   and ``partial_block`` (running intra-block sum of sublayer outputs).
    """

    def __init__(self, embedding: torch.Tensor, variant: AttnResVariant, block_size: int = 6):
        """Initialize the source state from the input embedding.

        Args:
            embedding: Initial representation, shape (B, T, d), used as the
                first source / completed block.
            variant: Full or block aggregation variant.
            block_size: Number of sublayers per block (block variant only).
        """
        self.variant = variant
        self.block_size = block_size

        if variant == AttnResVariant.FULL:
            self.sources: list[torch.Tensor] = [embedding]
        else:
            self.completed_blocks: list[torch.Tensor] = [embedding]
            self.partial_block: torch.Tensor | None = None
            self.layer_in_block: int = 0


class AttnResConnection(nn.Module):
    """Per-sublayer attention residual connection.

    Before each sublayer (``pre_sublayer``):
      1. [Block only] If block boundary reached, commit partial_block.
      2. Compute softmax attention over all previous representations
         to produce the sublayer input.

    After each sublayer (``post_sublayer``):
      - Full:  append raw sublayer output to sources.
      - Block: accumulate sublayer output into partial_block (intra-block sum).
    """

    def __init__(self, d_model: int):
        """Initialize the learned query and key normalization.

        Args:
            d_model: Token embedding dimension.
        """
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        self.norm = _RMSNorm(d_model)

    def _aggregate(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """Attention-aggregate sources using this connection's query and norm."""
        return _attnres_aggregate(sources, self.query, self.norm.weight, self.norm.eps)

    def pre_sublayer(self, state: AttnResState) -> torch.Tensor:
        """Check block boundary then aggregate to get sublayer input."""
        if state.variant == AttnResVariant.BLOCK:
            if state.layer_in_block == state.block_size:
                state.completed_blocks.append(state.partial_block)
                state.partial_block = None
                state.layer_in_block = 0

            if state.partial_block is not None:
                sources = state.completed_blocks + [state.partial_block]
            else:
                sources = state.completed_blocks
        else:
            sources = state.sources

        return self._aggregate(sources)

    def post_sublayer(self, state: AttnResState, output: torch.Tensor):
        """Accumulate sublayer output into state."""
        if state.variant == AttnResVariant.FULL:
            state.sources.append(output)
        else:
            if state.partial_block is None:
                state.partial_block = output
            else:
                state.partial_block = state.partial_block + output
            state.layer_in_block += 1


class AttnResFinalizer(nn.Module):
    """Final aggregation after all transformer blocks.

    Full AttnRes:  returns last source (already information-complete via
                   the preceding sublayer's attention aggregation).
    Block AttnRes: commits remaining partial block, then attention-aggregates
                   over all completed block representations.
    """

    def __init__(self, d_model: int, variant: AttnResVariant):
        """Initialize finalizer parameters (block variant only).

        Args:
            d_model: Token embedding dimension.
            variant: Full or block aggregation variant.
        """
        super().__init__()
        self.variant = variant
        if variant == AttnResVariant.BLOCK:
            self.query = nn.Parameter(torch.zeros(d_model))
            self.norm = _RMSNorm(d_model)

    def forward(self, state: AttnResState) -> torch.Tensor:
        """Produce the final representation from accumulated state.

        Args:
            state: Accumulated AttnResState after all blocks.

        Returns:
            Final token representation, shape (B, T, d_model).
        """
        if self.variant == AttnResVariant.FULL:
            return state.sources[-1]

        if state.partial_block is not None:
            state.completed_blocks.append(state.partial_block)
            state.partial_block = None

        return _attnres_aggregate(
            state.completed_blocks, self.query, self.norm.weight, self.norm.eps
        )
