"""Bond transformer block with self-attention, optional cross-attention, and FFN.

Supports three residual connection strategies (add, mhc, attnres) dispatched
internally so that callers always use the same ``forward()`` signature.
"""

import torch
import torch.nn as nn


def _fix_all_masked(history_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Unmask rows where every position is masked to prevent NaN in cross-attn."""
    if history_mask is None:
        return None
    all_masked = history_mask.all(dim=-1)
    if all_masked.any():
        history_mask = history_mask.clone()
        history_mask[all_masked] = False
    return history_mask


class BondTransformerBlock(nn.Module):
    """Self-attention + optional cross-attention + FFN.

    The residual connection strategy (``residual_type``) is handled internally:
    callers always use ``forward(x, src_key_padding_mask, history_context,
    history_mask, attnres_state)`` and never inspect ``_residual_type``.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, enable_cross_attention=False,
                 layer_scale_init=0.0, residual_type="add", num_streams=2,
                 block_idx=0, total_blocks=1, use_moe=False, moe_num_experts=8,
                 moe_top_k=2, moe_loss_coef=0.01, attnres_block_size=6,
                 moe_expert_dim=None, moe_expert_dropout=0.0):
        """Build attention, optional cross-attention, and FFN/MoE sublayers.

        The residual strategy (add, mhc, full_attnres, block_attnres) determines
        which auxiliary submodules (hyper-connections or attention-residual
        connections) are instantiated. LayerScale is enabled when
        layer_scale_init > 0.

        Args:
            d_model: Token embedding dimension.
            nhead: Number of attention heads.
            dim_feedforward: Hidden width of the FFN (also sets the expert
                hidden multiplier when use_moe is True).
            dropout: Dropout probability.
            enable_cross_attention: If True, add a cross-attention sublayer
                over history context.
            layer_scale_init: Initial LayerScale value; LayerScale is applied
                when this is positive.
            residual_type: One of "add", "mhc", "full_attnres", "block_attnres".
            num_streams: Number of mHC streams (mhc only).
            block_idx: Index of this block within the stack.
            total_blocks: Total number of blocks in the stack.
            use_moe: If True, replace the FFN with a sparse Mixture-of-Experts.
            moe_num_experts: Number of MoE experts.
            moe_top_k: Number of experts selected per token.
            moe_loss_coef: Coefficient for the MoE load-balancing loss.
            attnres_block_size: Block size for block-attnres aggregation.
            moe_expert_dim: Optional reduced expert dimension.
            moe_expert_dropout: Dropout probability inside MoE experts.
        """
        super().__init__()
        self._residual_type = residual_type
        self._use_moe = use_moe

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.enable_cross_attention = enable_cross_attention
        if enable_cross_attention:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm_cross = nn.LayerNorm(d_model)
            self.dropout_cross = nn.Dropout(dropout)

        self.moe_aux_loss = torch.tensor(0.0)
        if use_moe:
            from bond_match.modules.sparse_moe import MoE as LocalMoE, SparseMoEBlock
            moe = LocalMoE(
                dim=d_model,
                num_experts=moe_num_experts,
                expert_hidden_mult=dim_feedforward // d_model,
                gating_top_n=moe_top_k,
                balance_loss_coef=moe_loss_coef,
                router_z_loss_coef=moe_loss_coef * 0.1,
                expert_dim=moe_expert_dim,
                expert_dropout=moe_expert_dropout,
            )
            self.sparse_moe_block = SparseMoEBlock(
                moe, add_ff_before=False, add_ff_after=True,
                residual_type=residual_type, num_streams=num_streams,
                block_idx=block_idx, total_blocks=total_blocks,
            )
            self.norm2 = None
            self.dropout2 = None
            self.ffn = None
        else:
            self.sparse_moe_block = None
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        self.use_layer_scale = layer_scale_init > 0
        if self.use_layer_scale:
            self.gamma_sa = nn.Parameter(torch.full((d_model,), layer_scale_init))
            self.gamma_ffn = nn.Parameter(torch.full((d_model,), layer_scale_init))
            if enable_cross_attention:
                self.gamma_ca = nn.Parameter(torch.full((d_model,), layer_scale_init))

        if residual_type == "mhc":
            from bond_match.modules.mhc import HyperConnections
            total_sublayers = total_blocks * 3
            base = block_idx * 3
            self.hc_sa = HyperConnections(num_streams, d_model, layer_idx=base, total_layers=total_sublayers)
            self.hc_ffn = HyperConnections(num_streams, d_model, layer_idx=base + 2, total_layers=total_sublayers)
            if enable_cross_attention:
                self.hc_ca = HyperConnections(num_streams, d_model, layer_idx=base + 1, total_layers=total_sublayers)

        if residual_type in ("full_attnres", "block_attnres"):
            from bond_match.modules.attention_residuals import AttnResConnection
            self.ar_sa = AttnResConnection(d_model)
            if not use_moe:
                self.ar_ffn = AttnResConnection(d_model)
            if enable_cross_attention:
                self.ar_ca = AttnResConnection(d_model)
            self._attnres_block_size = attnres_block_size

    def forward(self, x, src_key_padding_mask=None, history_context=None, history_mask=None,
                attnres_state=None, attn_bias=None, group_ids=None):
        """Apply the block using the configured residual strategy.

        Dispatches to the additive, mHC, or attention-residual implementation.
        For "add" and "mhc", returns the updated tensor of the same shape as x.
        For the attention-residual variants, the result is accumulated into
        attnres_state in place and None is returned.

        Args:
            x: Input tokens, shape (B, T, d_model) (or (B*num_streams, T,
                d_model) for mHC); unused by the attention-residual path.
            src_key_padding_mask: Boolean mask, shape (B, T), True at padding.
            history_context: Optional cross-attention context.
            history_mask: Optional boolean mask for history_context.
            attnres_state: Mutable AttnResState (attention-residual variants).
            attn_bias: Optional additive attention bias / mask.
            group_ids: Optional per-token group ids for MoE capacity.

        Returns:
            Updated tokens for the "add" and "mhc" paths; None for the
            attention-residual paths.
        """
        if self._residual_type in ("full_attnres", "block_attnres"):
            return self._forward_attnres(src_key_padding_mask, history_context, history_mask, attnres_state, attn_bias=attn_bias, group_ids=group_ids)
        if self._residual_type == "mhc":
            return self._forward_mhc(x, src_key_padding_mask, history_context, history_mask, attn_bias=attn_bias, group_ids=group_ids)
        return self._forward_add(x, src_key_padding_mask, history_context, history_mask, attn_bias=attn_bias, group_ids=group_ids)

    def _forward_add(self, x, src_key_padding_mask=None, history_context=None, history_mask=None, attn_bias=None, group_ids=None):
        """Pre-norm additive-residual forward pass.

        Applies pre-norm self-attention, optional pre-norm cross-attention, and
        FFN or MoE, each added back as a residual (optionally LayerScale-scaled).

        Args:
            x: Input tokens, shape (B, T, d_model).
            src_key_padding_mask: Boolean padding mask, shape (B, T).
            history_context: Optional cross-attention context.
            history_mask: Optional boolean mask for history_context.
            attn_bias: Optional additive attention bias / mask.
            group_ids: Optional per-token group ids for MoE capacity.

        Returns:
            Updated tokens, shape (B, T, d_model).
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask, attn_mask=attn_bias, need_weights=False)
        sa_out = self.dropout1(attn_out)
        x = x + (self.gamma_sa * sa_out if self.use_layer_scale else sa_out)

        if self.enable_cross_attention and history_context is not None and history_context.size(1) > 0:
            history_mask = _fix_all_masked(history_mask)
            x_norm = self.norm_cross(x)
            cross_out, _ = self.cross_attn(x_norm, history_context, history_context, key_padding_mask=history_mask, need_weights=False)
            ca_out = self.dropout_cross(cross_out)
            x = x + (self.gamma_ca * ca_out if self.use_layer_scale else ca_out)

        if self._use_moe:
            token_mask = None if src_key_padding_mask is None else ~src_key_padding_mask
            moe_result = self.sparse_moe_block._forward_add(
                x, token_mask, False, 1., group_ids=group_ids
            )
            self.moe_aux_loss = moe_result.total_aux_loss
            x = moe_result.outputs
        else:
            ffn_out = self.dropout2(self.ffn(self.norm2(x)))
            x = x + (self.gamma_ffn * ffn_out if self.use_layer_scale else ffn_out)
        return x

    def _forward_mhc(self, x, src_key_padding_mask=None, history_context=None, history_mask=None, attn_bias=None, group_ids=None):
        """Manifold-constrained hyper-connection (mHC) forward pass.

        Each sublayer is wrapped by a width connection (mixing streams to form
        the branch input) and a depth connection (distributing the branch
        output back into the streams). Stream-aligned masks/contexts are
        subsampled with stride S = num_streams.

        Args:
            x: Stream-expanded tokens, shape (B*num_streams, T, d_model).
            src_key_padding_mask: Boolean padding mask over expanded streams.
            history_context: Optional cross-attention context.
            history_mask: Optional boolean mask for history_context.
            attn_bias: Optional additive attention bias / mask.
            group_ids: Optional per-token group ids for MoE capacity.

        Returns:
            Updated stream-expanded tokens, shape (B*num_streams, T, d_model).
        """
        S = self.hc_sa.num_streams

        branch_in, residuals_mixed = self.hc_sa.width_conn(x)
        x_norm = self.norm1(branch_in)
        sa_mask = src_key_padding_mask[::S] if src_key_padding_mask is not None else None
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=sa_mask, attn_mask=attn_bias, need_weights=False)
        sa_out = self.dropout1(attn_out)
        if self.use_layer_scale:
            sa_out = self.gamma_sa * sa_out
        x = self.hc_sa.depth_conn(sa_out, residuals_mixed)

        if self.enable_cross_attention and history_context is not None and history_context.size(1) > 0:
            history_mask = _fix_all_masked(history_mask)
            branch_in, residuals_mixed = self.hc_ca.width_conn(x)
            x_norm = self.norm_cross(branch_in)
            ca_history = history_context[::S] if history_context is not None else None
            ca_mask = history_mask[::S] if history_mask is not None else None
            cross_out, _ = self.cross_attn(x_norm, ca_history, ca_history, key_padding_mask=ca_mask, need_weights=False)
            ca_out = self.dropout_cross(cross_out)
            if self.use_layer_scale:
                ca_out = self.gamma_ca * ca_out
            x = self.hc_ca.depth_conn(ca_out, residuals_mixed)

        if self._use_moe:
            token_mask = None if sa_mask is None else ~sa_mask
            moe_result = self.sparse_moe_block(x, token_mask=token_mask, group_ids=group_ids)
            self.moe_aux_loss = moe_result.total_aux_loss
            x = moe_result.outputs
        else:
            branch_in, residuals_mixed = self.hc_ffn.width_conn(x)
            ffn_out = self.dropout2(self.ffn(self.norm2(branch_in)))
            if self.use_layer_scale:
                ffn_out = self.gamma_ffn * ffn_out
            x = self.hc_ffn.depth_conn(ffn_out, residuals_mixed)
        return x

    def _forward_attnres(self, src_key_padding_mask=None, history_context=None, history_mask=None, state=None, attn_bias=None, group_ids=None):
        """Attention-residual forward pass.

        Each sublayer reads its input by aggregating prior representations from
        state (pre_sublayer) and writes its output back into state
        (post_sublayer). The state is mutated in place; nothing is returned.

        Args:
            src_key_padding_mask: Boolean padding mask, shape (B, T).
            history_context: Optional cross-attention context.
            history_mask: Optional boolean mask for history_context.
            state: Mutable AttnResState accumulating sublayer outputs.
            attn_bias: Optional additive attention bias / mask.
            group_ids: Optional per-token group ids for MoE capacity.
        """
        h = self.ar_sa.pre_sublayer(state)
        x_norm = self.norm1(h)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask, attn_mask=attn_bias, need_weights=False)
        sa_out = self.dropout1(attn_out)
        if self.use_layer_scale:
            sa_out = self.gamma_sa * sa_out
        self.ar_sa.post_sublayer(state, sa_out)

        if self.enable_cross_attention and history_context is not None and history_context.size(1) > 0:
            history_mask = _fix_all_masked(history_mask)
            h = self.ar_ca.pre_sublayer(state)
            x_norm = self.norm_cross(h)
            cross_out, _ = self.cross_attn(x_norm, history_context, history_context, key_padding_mask=history_mask, need_weights=False)
            ca_out = self.dropout_cross(cross_out)
            if self.use_layer_scale:
                ca_out = self.gamma_ca * ca_out
            self.ar_ca.post_sublayer(state, ca_out)

        if self._use_moe:
            token_mask = None if src_key_padding_mask is None else ~src_key_padding_mask
            moe_result = self.sparse_moe_block._forward_attnres(
                state, token_mask=token_mask, group_ids=group_ids
            )
            self.moe_aux_loss = moe_result.total_aux_loss
        else:
            h = self.ar_ffn.pre_sublayer(state)
            ffn_out = self.dropout2(self.ffn(self.norm2(h)))
            if self.use_layer_scale:
                ffn_out = self.gamma_ffn * ffn_out
            self.ar_ffn.post_sublayer(state, ffn_out)
