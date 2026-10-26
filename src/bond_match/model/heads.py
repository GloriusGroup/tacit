"""Swap-type prediction heads and virtual node prediction heads."""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class MLPSwapTypeHead(nn.Sequential):
    """Original MLP swap type classifier from concatenated bond embeddings.

    Takes [z_i || z_j] (2*d_model) and predicts 3 classes: stable, swap0, swap1.
    This is the format used by v6 checkpoints. Inherits nn.Sequential directly
    so state_dict keys match the old checkpoint format (0.weight, 1.weight, etc.).
    """

    def __init__(self, d_model: int):
        """Build the MLP classifier over concatenated bond embeddings.

        Args:
            d_model: Bond embedding dimension.
        """
        super().__init__(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, z_b1, z_b2=None):
        """Predict swap-type logits from one or two bond embeddings.

        Args:
            z_b1: Bond embedding(s); either the pre-concatenated input or the
                first bond when z_b2 is given. Shape (..., 2*d_model) or
                (..., d_model).
            z_b2: Optional second bond embedding, shape (..., d_model).

        Returns:
            Swap-type logits, shape (..., 3).
        """
        if z_b2 is not None:
            if z_b1.dim() == 1:
                z_b1, z_b2 = z_b1.unsqueeze(0), z_b2.unsqueeze(0)
            x = torch.cat([z_b1, z_b2], dim=-1)
        else:
            x = z_b1
        return super().forward(x)


class AtomSwapTypeHead(nn.Module):
    """Predict bond swap type from the 4 individual atom embeddings.

    Given break bonds (u1,u2) and (v1,v2), swap type determines which atoms
    reconnect:
        swap0: form (u1,v1) and (u2,v2) -- parallel
        swap1: form (u1,v2) and (u2,v1) -- cross

    Uses atom-level compatibility scores + a small MLP for context.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """Build atom projection and the swap-type classifier.

        Args:
            d_model: Atom embedding dimension.
            dropout: Dropout probability in the classifier.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(2 + d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )
        self._scale = d_model ** -0.5

    def forward(self, h_u1, h_u2, h_v1, h_v2):
        """Predict swap-type logits from the four atom embeddings.

        The parallel (swap0) and cross (swap1) reconnection compatibility
        scores are computed from scaled dot products of projected atom
        embeddings and combined with bond asymmetry features.

        Args:
            h_u1, h_u2: Atom embeddings of the first break bond, shape
                (d_model,) or (B, d_model).
            h_v1, h_v2: Atom embeddings of the second break bond.

        Returns:
            Swap-type logits, shape (B, 3) or (1, 3).
        """
        if h_u1.dim() == 1:
            h_u1, h_u2, h_v1, h_v2 = [x.unsqueeze(0) for x in [h_u1, h_u2, h_v1, h_v2]]

        p_u1 = self.proj(h_u1)
        p_u2 = self.proj(h_u2)
        p_v1 = self.proj(h_v1)
        p_v2 = self.proj(h_v2)

        score0 = (p_u1 * p_v1).sum(-1, keepdim=True) * self._scale + \
                 (p_u2 * p_v2).sum(-1, keepdim=True) * self._scale

        score1 = (p_u1 * p_v2).sum(-1, keepdim=True) * self._scale + \
                 (p_u2 * p_v1).sum(-1, keepdim=True) * self._scale

        bond_a_asym = h_u1 - h_u2
        bond_b_asym = h_v1 - h_v2

        features = torch.cat([score0, score1, bond_a_asym, bond_b_asym], dim=-1)
        return self.classifier(features)


class BondSplitSwapHead(nn.Module):
    """Predict swap type from post-transformer bond embeddings.

    Learns to decompose atom signals from bond tokens. Each bond token
    was built as proj([h_u, h_v, h_u*h_v]) -- the projection preserves
    asymmetry between the two atoms. This head learns two projections
    to extract "atom 1" and "atom 2" signals from the bond embedding,
    then computes compatibility scores like AtomSwapTypeHead.

    Optionally uses pre-transformer GAT node embeddings (detached) as
    a guide for atom identity, without gradient interference.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, use_node_guide: bool = True):
        """Build atom-signal projections, compatibility projection, and classifier.

        Args:
            d_model: Bond embedding dimension.
            dropout: Dropout probability in the classifier.
            use_node_guide: If True, add a gated projection of detached GAT
                node embeddings to the extracted atom signals.
        """
        super().__init__()
        half = d_model // 2
        self.atom1_proj = nn.Linear(d_model, half)
        self.atom2_proj = nn.Linear(d_model, half)
        self.compat_proj = nn.Linear(half, half, bias=False)
        self._scale = half ** -0.5
        self._use_node_guide = use_node_guide
        if use_node_guide:
            self.node_proj = nn.Linear(d_model, half)
            self.node_gate = nn.Parameter(torch.tensor(0.1))
        cls_in = 2 + d_model * 2
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, z_b1, z_b2,
                h_u1=None, h_u2=None, h_v1=None, h_v2=None):
        """Predict swap-type logits from post-transformer bond embeddings.

        Args:
            z_b1: Bond 1 embedding, shape (d_model,) or (1, d_model).
            z_b2: Bond 2 embedding, same shape as z_b1.
            h_u1, h_u2: Optional GAT node embeddings for bond 1 atoms; used
                (detached) only when use_node_guide is True and provided.
            h_v1, h_v2: Optional GAT node embeddings for bond 2 atoms.

        Returns:
            Swap-type logits, shape (B, 3) or (1, 3).
        """
        if z_b1.dim() == 1:
            z_b1, z_b2 = z_b1.unsqueeze(0), z_b2.unsqueeze(0)

        a1_b1 = self.atom1_proj(z_b1)
        a2_b1 = self.atom2_proj(z_b1)
        a1_b2 = self.atom1_proj(z_b2)
        a2_b2 = self.atom2_proj(z_b2)

        if self._use_node_guide and h_u1 is not None:
            if h_u1.dim() == 1:
                h_u1, h_u2, h_v1, h_v2 = [x.unsqueeze(0) for x in [h_u1, h_u2, h_v1, h_v2]]
            g = self.node_gate
            a1_b1 = a1_b1 + g * self.node_proj(h_u1.detach())
            a2_b1 = a2_b1 + g * self.node_proj(h_u2.detach())
            a1_b2 = a1_b2 + g * self.node_proj(h_v1.detach())
            a2_b2 = a2_b2 + g * self.node_proj(h_v2.detach())

        p1_b1 = self.compat_proj(a1_b1)
        p2_b1 = self.compat_proj(a2_b1)
        p1_b2 = self.compat_proj(a1_b2)
        p2_b2 = self.compat_proj(a2_b2)

        score0 = (p1_b1 * p1_b2).sum(-1, keepdim=True) * self._scale + \
                 (p2_b1 * p2_b2).sum(-1, keepdim=True) * self._scale
        score1 = (p1_b1 * p2_b2).sum(-1, keepdim=True) * self._scale + \
                 (p2_b1 * p1_b2).sum(-1, keepdim=True) * self._scale

        bond_diff = z_b1 - z_b2
        bond_prod = z_b1 * z_b2

        features = torch.cat([score0, score1, bond_diff, bond_prod], dim=-1)
        return self.classifier(features)


class NodeAttentionSwapHead(nn.Module):
    """Predict swap type via self-attention over the 4 atoms of a bond pair.

    Given break bonds (u1,u2) and (v1,v2), stacks the 4 atom embeddings as
    a sequence with learned role embeddings, runs self-attention so each atom
    sees the other 3's chemical context, then computes compatibility scores.

    Outputs 2 classes (swap0=parallel, swap1=cross) -- no wasted stable class.
    """

    def __init__(self, d_model: int, num_heads: int = 4, num_layers: int = 1,
                 dropout: float = 0.1):
        """Build role embeddings, self-attention layers, and the classifier.

        Args:
            d_model: Atom embedding dimension.
            num_heads: Number of attention heads per self-attention layer.
            num_layers: Number of self-attention layers over the 4 atoms.
            dropout: Dropout probability in attention and the classifier.
        """
        super().__init__()
        self.role_emb = nn.Embedding(4, d_model)

        self.sa_layers = nn.ModuleList()
        self.sa_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.sa_layers.append(
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True),
            )
            self.sa_norms.append(nn.LayerNorm(d_model))
        self.sa_dropout = nn.Dropout(dropout)

        self.compat_proj = nn.Linear(d_model, d_model, bias=False)
        self._scale = d_model ** -0.5

        self.classifier = nn.Sequential(
            nn.Linear(2 + d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, h_u1, h_u2, h_v1, h_v2):
        """Predict swap-type logits via self-attention over the 4 atoms.

        The four atom embeddings are stacked as a length-4 sequence with
        learned role embeddings, processed by self-attention, then combined
        into parallel (swap0) and cross (swap1) compatibility scores.

        Args:
            h_u1, h_u2: Atom embeddings of the first break bond, shape
                (d_model,) or (B, d_model).
            h_v1, h_v2: Atom embeddings of the second break bond.

        Returns:
            Swap-type logits [swap0, swap1], shape (B, 2).
        """
        if h_u1.dim() == 1:
            h_u1, h_u2, h_v1, h_v2 = [x.unsqueeze(0) for x in [h_u1, h_u2, h_v1, h_v2]]

        B = h_u1.size(0)
        device = h_u1.device
        roles = self.role_emb(torch.arange(4, device=device))

        seq = torch.stack([h_u1, h_u2, h_v1, h_v2], dim=1)
        seq = seq + roles.unsqueeze(0)

        for norm, sa in zip(self.sa_norms, self.sa_layers):
            x_norm = norm(seq)
            attn_out, _ = sa(x_norm, x_norm, x_norm)
            seq = seq + self.sa_dropout(attn_out)

        u1, u2, v1, v2 = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3]

        p_u1 = self.compat_proj(u1)
        p_u2 = self.compat_proj(u2)
        p_v1 = self.compat_proj(v1)
        p_v2 = self.compat_proj(v2)

        score0 = (p_u1 * p_v1).sum(-1, keepdim=True) * self._scale + \
                 (p_u2 * p_v2).sum(-1, keepdim=True) * self._scale
        score1 = (p_u1 * p_v2).sum(-1, keepdim=True) * self._scale + \
                 (p_u2 * p_v1).sum(-1, keepdim=True) * self._scale

        bond_a_asym = u1 - u2
        bond_b_asym = v1 - v2

        features = torch.cat([score0, score1, bond_a_asym, bond_b_asym], dim=-1)
        return self.classifier(features)


class AttentionPoolVirtualHead(nn.Module):
    """Learned attention pooling for virtual node (continue/stop) prediction.

    A learnable query token, conditioned on step index, attends over all
    per-molecule bond embeddings. Replaces mean-pool aggregation with
    importance-weighted pooling.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1,
                 max_steps: int = 8, use_deep_head: bool = False):
        """Build the learnable query, step embeddings, attention, and output head.

        Args:
            d_model: Bond embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability in attention.
            max_steps: Number of distinct step indices with learned embeddings.
            use_deep_head: If True, use a deeper output MLP.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.step_emb = nn.Embedding(max_steps, d_model)
        self._max_steps = max_steps

        self.query_proj = nn.Linear(d_model, d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        if use_deep_head:
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, bond_context, bond_batch_vec, step_index,
                dense_bonds=None, mask=None):
        """
        Args:
            bond_context: [total_bonds, d_model] post-transformer bond embeddings
            bond_batch_vec: [total_bonds] molecule assignment per bond
            step_index: [B] current step indices
            dense_bonds: optional [B, max_bonds, d_model] pre-computed dense batch
            mask: optional [B, max_bonds] True=valid, False=padding
        Returns:
            logits: [B, 1] virtual node logits
        """
        if dense_bonds is not None and mask is not None:
            kv = dense_bonds
            pad_mask = ~mask
        else:
            kv, mask_computed = to_dense_batch(bond_context, bond_batch_vec)
            pad_mask = ~mask_computed

        B = kv.size(0)
        step_emb = self.step_emb(step_index.clamp(0, self._max_steps - 1))
        query = self.query.expand(B, -1, -1) + step_emb.unsqueeze(1)

        query_norm = self.norm_q(self.query_proj(query))
        kv_norm = self.norm_kv(kv)

        pooled, _ = self.attn(query_norm, kv_norm, kv_norm, key_padding_mask=pad_mask)
        return self.output_head(pooled.squeeze(1))
