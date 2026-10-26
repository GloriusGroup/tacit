"""Manifold-constrained hyper-connections (mHC).

Replaces standard residual connections (x = x + f(x)) with multi-stream
residuals connected through learnable doubly-stochastic mixing matrices.

Streams use an interleaved layout: expanding [b0, b1] with S=2 yields
[b0, b0, b1, b1], so view(B, S, T, D) decomposes the leading dimension into
batch and stream dimensions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _doubly_stochastic_project(logits: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Project logits to a doubly-stochastic matrix by alternating normalization.

    The projection is computed without autograd through the iterations; a
    straight-through estimator returns the projected matrix in the forward pass
    while routing gradients to the logits via a row-softmax surrogate.

    Args:
        logits: Square logit matrix, shape (..., S, S).
        n_iters: Number of row/column normalization iterations.

    Returns:
        Approximately doubly-stochastic matrix, same shape as logits.
    """
    with torch.no_grad():
        M = torch.exp(logits)
        for _ in range(n_iters):
            M = M / M.sum(dim=-1, keepdim=True)
            M = M / M.sum(dim=-2, keepdim=True)

    row_softmax = F.softmax(logits, dim=-1)
    return (M - row_softmax).detach() + row_softmax


class HyperConnections(nn.Module):
    """Single-sublayer hyper-connection wrapper.

    For S=2, uses closed-form sigmoid parameterization (no iterations).
    For S>2, uses doubly-stochastic projection with straight-through gradient.
    All ops use broadcast multiply (autocast-friendly, no matmul on tiny matrices).
    """

    def __init__(self, num_streams: int, d_model: int, layer_idx: int = 0, total_layers: int = 1):
        """Initialize the mixing and pre/post connection logits.

        The residual-mixing parameterization depends on the stream count: S=2
        uses a single sigmoid logit, while S>2 uses a full logit matrix
        projected to doubly-stochastic form. The pre-connection logits are
        initialized to select the stream layer_idx % num_streams.

        Args:
            num_streams: Number of residual streams S.
            d_model: Token embedding dimension.
            layer_idx: Index of this connection within the stack.
            total_layers: Total number of connections in the stack.
        """
        super().__init__()
        self.num_streams = num_streams
        self.d_model = d_model

        if num_streams == 2:
            self.H_res_logit = nn.Parameter(torch.tensor(4.0))
        else:
            H_res_init = torch.full((num_streams, num_streams), -8.0)
            H_res_init.fill_diagonal_(0.0)
            self.H_res_logits = nn.Parameter(H_res_init)

        init_idx = layer_idx % num_streams
        H_pre_init = torch.full((num_streams,), -8.0)
        H_pre_init[init_idx] = 0.0
        self.H_pre_logits = nn.Parameter(H_pre_init)
        self.H_post_logits = nn.Parameter(torch.zeros(num_streams))

    def _get_H_res(self) -> torch.Tensor:
        """Return the doubly-stochastic stream-mixing matrix, shape (S, S)."""
        if self.num_streams == 2:
            a = torch.sigmoid(self.H_res_logit)
            return torch.stack([torch.stack([a, 1 - a]), torch.stack([1 - a, a])])
        return _doubly_stochastic_project(self.H_res_logits)

    def width_conn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix streams and select the sublayer branch input.

        Args:
            x: Stream-expanded tokens, shape (B*S, T, D).

        Returns:
            Tuple of (branch_input, residuals_mixed) with shapes (B, T, D) and
            (B, S, T, D).
        """
        S = self.num_streams
        B = x.shape[0] // S
        streams = x.view(B, S, *x.shape[1:])

        H_res = self._get_H_res()
        H_pre = F.softmax(self.H_pre_logits, dim=-1)

        residuals_mixed = (H_res.view(1, S, S, 1, 1) * streams.unsqueeze(1)).sum(dim=2)

        branch_input = (H_pre.view(1, S, 1, 1) * streams).sum(dim=1)

        return branch_input, residuals_mixed

    def depth_conn(self, branch_output: torch.Tensor, residuals_mixed: torch.Tensor) -> torch.Tensor:
        """Distribute the branch output across streams and add the residuals.

        Args:
            branch_output: Sublayer output, shape (B, T, D).
            residuals_mixed: Mixed residual streams, shape (B, S, T, D).

        Returns:
            Updated stream-expanded tokens, shape (B*S, T, D).
        """
        S = self.num_streams
        H_post = F.softmax(self.H_post_logits, dim=-1)

        branch_to_streams = branch_output.unsqueeze(1) * H_post.view(1, S, 1, 1)

        output = residuals_mixed + branch_to_streams
        return output.reshape(-1, *output.shape[2:])


class StreamManager:
    """Expand and reduce the batch dimension for the interleaved stream layout."""

    def __init__(self, num_streams: int):
        """Store the stream count.

        Args:
            num_streams: Number of residual streams S.
        """
        self.num_streams = num_streams

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat-interleave the batch dimension: (B, ...) -> (B*S, ...)."""
        return x.repeat_interleave(self.num_streams, dim=0)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Sum over streams: (B*S, ...) -> (B, ...)."""
        S = self.num_streams
        B = x.shape[0] // S
        return x.view(B, S, *x.shape[1:]).sum(dim=1)
