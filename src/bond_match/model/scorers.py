"""Bond pair scoring modules."""

import torch
import torch.nn as nn


class AttentionPairScorer(nn.Module):
    """Score bond pairs via multi-head query-key dot-product attention.

    Each bond embedding is projected to query and key spaces. The per-head
    score is (W_q z_i) . (W_k z_j) / sqrt(d_head), and a learned linear layer
    aggregates the H head scores into a scalar. Optionally symmetrized by
    averaging the forward and reversed scores.
    """

    def __init__(self, d_model, num_heads=4, use_symmetric_pair_scorer=False):
        """Initialize projections and head-aggregation weights.

        Args:
            d_model: Bond embedding dimension; must be divisible by num_heads.
            num_heads: Number of attention heads.
            use_symmetric_pair_scorer: If True, average forward and reversed
                scores to enforce symmetry in the bond pair ordering.
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_symmetric_pair_scorer = use_symmetric_pair_scorer
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5
        self.head_weight = nn.Linear(num_heads, 1, bias=False)

    def _compute_attn(self, z_i, z_j):
        """Compute per-head scaled dot-product scores.

        Args:
            z_i: Bond embeddings, shape (N, d_model).
            z_j: Bond embeddings, shape (N, d_model).

        Returns:
            Per-head attention scores, shape (N, num_heads).
        """
        q = self.W_q(z_i).view(-1, self.num_heads, self.d_head)
        k = self.W_k(z_j).view(-1, self.num_heads, self.d_head)
        attn = (q * k).sum(-1) * self.scale
        return attn

    def _combine(self, attn_fwd, attn_rev=None):
        """Aggregate per-head scores into a scalar, optionally symmetrized.

        Args:
            attn_fwd: Forward per-head scores, shape (N, num_heads).
            attn_rev: Optional reversed per-head scores for symmetrization.

        Returns:
            Pair score, shape (N, 1).
        """
        score = self.head_weight(attn_fwd)
        if attn_rev is not None:
            score = (score + self.head_weight(attn_rev)) / 2.0
        return score

    def forward(self, z_i, z_j):
        """Score bond pairs (z_i, z_j).

        Args:
            z_i: Bond embeddings, shape (N, d_model).
            z_j: Bond embeddings, shape (N, d_model).

        Returns:
            Pair scores, shape (N, 1).
        """
        attn_fwd = self._compute_attn(z_i, z_j)
        if self.use_symmetric_pair_scorer:
            attn_rev = self._compute_attn(z_j, z_i)
            return self._combine(attn_fwd, attn_rev)
        return self._combine(attn_fwd)

    def score_upper_triangle(self, z: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        """Compute upper-triangular pair scores without materializing all pairs.

        Processes rows in chunks to bound memory. The output is flattened in
        row-major upper-triangular order (matching torch.triu_indices).

        Args:
            z: Bond embeddings, shape (n_bonds, d_model).
            chunk_size: Number of rows processed per chunk.

        Returns:
            Flattened upper-triangular pair scores, shape
            (n_bonds * (n_bonds - 1) / 2,).
        """
        n_bonds = z.size(0)
        if n_bonds < 2:
            return z.new_empty(0)

        q_all = self.W_q(z).view(n_bonds, self.num_heads, self.d_head)
        k_all = self.W_k(z).view(n_bonds, self.num_heads, self.d_head)
        head_weight = self.head_weight.weight.view(self.num_heads)

        out = z.new_empty(n_bonds * (n_bonds - 1) // 2)
        write_pos = 0

        for row_start in range(0, n_bonds - 1, chunk_size):
            row_end = min(row_start + chunk_size, n_bonds - 1)
            q_chunk = q_all[row_start:row_end]
            base = torch.einsum("rhd,nhd->rnh", q_chunk, k_all) * self.scale
            scores = torch.einsum("rnh,h->rn", base, head_weight)

            if self.use_symmetric_pair_scorer:
                k_chunk = k_all[row_start:row_end]
                rev = torch.einsum("nhd,rhd->rnh", q_all, k_chunk) * self.scale
                rev_scores = torch.einsum("rnh,h->rn", rev, head_weight)
                scores = 0.5 * (scores + rev_scores)

            for local_row, global_row in enumerate(range(row_start, row_end)):
                row_scores = scores[local_row, global_row + 1 :]
                row_len = row_scores.numel()
                out[write_pos : write_pos + row_len] = row_scores
                write_pos += row_len

        return out

    def score_full_batched(self, z_padded: torch.Tensor) -> torch.Tensor:
        """Compute the full pairwise score matrix per batch element.

        Args:
            z_padded: Padded bond embeddings, shape (B, M, d_model).

        Returns:
            Pair scores, shape (B, M, M), where entry (b, i, j) is the
            head-aggregated score between bonds i and j. Only meaningful when
            both i and j index valid bonds for batch element b.
        """
        B, M, _ = z_padded.shape
        q = self.W_q(z_padded).view(B, M, self.num_heads, self.d_head)
        k = self.W_k(z_padded).view(B, M, self.num_heads, self.d_head)
        attn = torch.einsum("bmhd,bnhd->bmnh", q, k) * self.scale
        score = self.head_weight(attn).squeeze(-1)
        if self.use_symmetric_pair_scorer:
            score = 0.5 * (score + score.transpose(1, 2))
        return score


class BondPairScorer(nn.Module):
    """MLP compatibility score between two bonds.

    Concatenates the two bond embeddings and passes them through an MLP,
    allowing non-linear and asymmetric pairwise interactions. Optionally
    symmetrized by averaging the score over both input orderings.
    """

    def __init__(self, input_dim, hidden_dim=128, use_symmetric_pair_scorer=False, dropout=0.1):
        """Initialize the scoring MLP.

        Args:
            input_dim: Dimension of a single bond embedding.
            hidden_dim: Hidden width of the first MLP layer.
            use_symmetric_pair_scorer: If True, average scores over both
                input orderings to enforce symmetry.
            dropout: Dropout probability inside the MLP.
        """
        super().__init__()
        self.use_symmetric_pair_scorer = use_symmetric_pair_scorer
        self.net = nn.Sequential(nn.Linear(input_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, z_i, z_j):
        """Score bond pairs (z_i, z_j).

        Args:
            z_i: Bond embeddings, shape (N, input_dim).
            z_j: Bond embeddings, shape (N, input_dim).

        Returns:
            Pair scores, shape (N, 1).
        """
        combined = torch.cat([z_i, z_j], dim=-1)
        if self.use_symmetric_pair_scorer:
            combined_sym = torch.cat([z_j, z_i], dim=-1)
            score = self.net(combined)
            score_sym = self.net(combined_sym)
            return (score + score_sym) / 2.0
        return self.net(combined)
