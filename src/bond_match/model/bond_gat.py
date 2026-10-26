"""BondGAT -- main ranking-based bond prediction network.

Graph Attention Network backbone with bond-level transformer and
multi-step reaction history context via cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch

from bond_match.model.heads import (
    AtomSwapTypeHead,
    AttentionPoolVirtualHead,
    BondSplitSwapHead,
    MLPSwapTypeHead,
    NodeAttentionSwapHead,
)
from bond_match.model.scorers import AttentionPairScorer, BondPairScorer
from bond_match.model.transformer import BondTransformerBlock


class _ResidualManager:
    """Encapsulates residual-type-specific expand / reduce / state creation.

    BondGAT delegates to this so that ``forward()`` and
    ``encode_history_tokens()`` stay free of residual-type conditionals.
    """

    def __init__(
        self,
        residual_type: str,
        num_streams: int,
        attnres_block_size: int,
        stream_manager=None,
        attnres_variant=None,
        attnres_finalizer=None,
    ):
        self._type = residual_type
        self._num_streams = num_streams
        self._attnres_block_size = attnres_block_size
        self._stream_manager = stream_manager
        self._attnres_variant = attnres_variant
        self._attnres_finalizer = attnres_finalizer

    @property
    def is_mhc(self) -> bool:
        return self._type == "mhc"

    @property
    def is_attnres(self) -> bool:
        return self._type in ("full_attnres", "block_attnres")

    def build_pad_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert boolean bond mask to padding mask, duplicating for mHC streams."""
        if self.is_mhc:
            return (~mask).repeat_interleave(self._num_streams, dim=0)
        return ~mask

    def expand_context(
        self, history_context: torch.Tensor | None, history_mask: torch.Tensor | None
    ):
        """Duplicate history tensors for mHC streams."""
        if not self.is_mhc:
            return history_context, history_mask
        if history_context is not None:
            history_context = history_context.repeat_interleave(
                self._num_streams, dim=0
            )
        if history_mask is not None:
            history_mask = history_mask.repeat_interleave(self._num_streams, dim=0)
        return history_context, history_mask

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Expand input tensor into mHC streams."""
        if self.is_mhc:
            return self._stream_manager.expand(x)
        return x

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce mHC streams back to single tensor."""
        if self.is_mhc:
            return self._stream_manager.reduce(x)
        return x

    def run_transformer(
        self,
        blocks,
        attn_output,
        dense_bonds,
        pad_mask,
        history_context=None,
        history_mask=None,
        attn_bias=None,
        group_ids=None,
    ):
        """Run bond transformer blocks with the appropriate residual strategy."""
        if self.is_attnres:
            from bond_match.modules.attention_residuals import AttnResState

            state = AttnResState(
                dense_bonds, self._attnres_variant, self._attnres_block_size
            )
            for block in blocks:
                block(
                    None,
                    src_key_padding_mask=pad_mask,
                    history_context=history_context,
                    history_mask=history_mask,
                    attnres_state=state,
                    attn_bias=attn_bias,
                    group_ids=group_ids,
                )
            return self._attnres_finalizer(state)

        # Standard additive or mHC pass
        for block in blocks:
            attn_output = block(
                attn_output,
                src_key_padding_mask=pad_mask,
                history_context=history_context,
                history_mask=history_mask,
                attn_bias=attn_bias,
                group_ids=group_ids,
            )
        return attn_output


class BondGAT(nn.Module):
    def __init__(
        self,
        d_model=256,
        heads=4,
        num_layers=4,
        atom_max_num=100,
        num_bond_types=20,
        dropout=0.1,
        transformer_layers=2,
        return_type_prediction=False,
        extended_atom_features=False,
        enable_history_context: bool = False,
        use_pair_bond_type: bool = False,
        use_symmetric_pair_scorer: bool = False,
        use_max_pool_virtual: bool = False,
        use_edge_bond_tokens: bool = False,
        use_history_step_pos: bool = False,
        pair_scorer_dropout: float = 0.0,
        layer_scale_init: float = 0.0,
        bond_proj_mode: str = "default",
        use_pair_score_virtual: bool = False,
        use_attention_pair_scorer: bool = False,
        use_deep_virtual_head: bool = False,
        use_step_aware_virtual: bool = False,
        gat_drop_path: float = 0.0,
        step_embed_bond_proj: bool = False,
        residual_connection_type: str = "add",
        mhc_num_streams: int = 2,
        use_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_loss_coef: float = 0.01,
        moe_expert_dim: int | None = None,
        moe_expert_dropout: float = 0.0,
        swap_type_head: str = "atom",
        attnres_block_size: int = 6,
        use_attention_pool_virtual: bool = False,
        use_canonical_rank: bool = False,
        skip_gat: bool = False,
        graph_attention_bias: bool = False,
        graph_attention_bias_hops: int = 1,
        pair_score_backend: str = "legacy",
        pair_score_chunk_size: int = 512,
        drop_head_bond_tokens_after_gnn: bool = False,
        skip_empty_history_cross_attn_for_history_encoder: bool = False,
        trajectory_attention: bool = False,
    ):
        super(BondGAT, self).__init__()
        self.return_type_prediction = return_type_prediction
        self.use_max_pool_virtual = use_max_pool_virtual
        self.use_edge_bond_tokens = use_edge_bond_tokens
        self.d_model = d_model
        self.num_heads = heads
        self.extended_atom_features = extended_atom_features
        self.enable_history_context = enable_history_context
        self.use_pair_bond_type = use_pair_bond_type
        self.use_symmetric_pair_scorer = use_symmetric_pair_scorer
        self._step_embed_bond_proj = step_embed_bond_proj
        self._residual_connection_type = residual_connection_type
        self._mhc_num_streams = mhc_num_streams
        self._attnres_block_size = attnres_block_size
        self._skip_gat = skip_gat
        self._graph_attention_bias = graph_attention_bias
        self._graph_attention_bias_hops = graph_attention_bias_hops
        self._pair_score_backend = pair_score_backend
        self._pair_score_chunk_size = pair_score_chunk_size
        self._drop_head_bond_tokens_after_gnn = drop_head_bond_tokens_after_gnn
        self._skip_empty_history_cross_attn_for_history_encoder = (
            skip_empty_history_cross_attn_for_history_encoder
        )
        self._trajectory_attention = trajectory_attention

        # mHC stream manager
        _stream_manager = None
        if residual_connection_type == "mhc":
            from bond_match.modules.mhc import StreamManager

            self.stream_manager = StreamManager(mhc_num_streams)
            _stream_manager = self.stream_manager

        self.atom_emb = nn.Embedding(atom_max_num + 1, d_model)
        self.charge_emb = nn.Embedding(20, d_model, padding_idx=0)
        self.h_count_emb = nn.Embedding(10, d_model)
        self.aromatic_emb = nn.Embedding(2, d_model)
        self.symmetry_emb = nn.Linear(1, d_model)
        self.is_virtual_emb = nn.Embedding(2, d_model)
        self.step_idx_node_emb = nn.Embedding(8, d_model)
        self.edge_emb = nn.Embedding(num_bond_types + 1, d_model)

        self._use_canonical_rank = use_canonical_rank
        n_emb = 7
        if use_canonical_rank:
            self.canonical_rank_emb = nn.Linear(1, d_model)
            n_emb += 1
        if extended_atom_features:
            self.ring_emb = nn.Embedding(2, d_model)
            self.ring_size_emb = nn.Embedding(10, d_model)
            self.degree_emb = nn.Embedding(8, d_model)
            self.hybridization_emb = nn.Embedding(8, d_model)
            self.catalyst_emb = nn.Embedding(2, d_model)
            n_emb += 5

        self.node_proj = nn.Sequential(
            nn.Linear(d_model * n_emb, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        self.layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self._gat_layer_scale = layer_scale_init > 0
        if self._gat_layer_scale:
            self.gat_gammas = nn.ParameterList()
        self._gat_drop_path = gat_drop_path
        if gat_drop_path > 0:
            self._drop_probs = [
                i / max(num_layers - 1, 1) * gat_drop_path for i in range(num_layers)
            ]
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    d_model,
                    d_model // heads,
                    heads=heads,
                    edge_dim=d_model,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )
            self.gat_norms.append(nn.LayerNorm(d_model))
            if self._gat_layer_scale:
                self.gat_gammas.append(
                    nn.Parameter(torch.full((d_model,), layer_scale_init))
                )

        bond_proj_dim = d_model * 4 if use_edge_bond_tokens else d_model * 3
        if step_embed_bond_proj:
            bond_proj_dim += d_model
        self._bond_proj_mode = bond_proj_mode
        if bond_proj_mode == "deep":
            self.bond_proj = nn.Sequential(
                nn.Linear(bond_proj_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        else:
            self.bond_proj = nn.Sequential(
                nn.Linear(bond_proj_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
            )
        if bond_proj_mode == "residual":
            self.bond_proj_skip = nn.LayerNorm(d_model)

        _residual_kwargs = dict(
            residual_type=residual_connection_type,
            num_streams=mhc_num_streams,
            use_moe=use_moe,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_loss_coef=moe_loss_coef,
            moe_expert_dim=moe_expert_dim,
            moe_expert_dropout=moe_expert_dropout,
            attnres_block_size=attnres_block_size,
        )
        self.bond_transformer_blocks = nn.ModuleList(
            [
                BondTransformerBlock(
                    d_model,
                    heads,
                    d_model * 2,
                    dropout,
                    enable_cross_attention=enable_history_context,
                    layer_scale_init=layer_scale_init,
                    block_idx=i,
                    total_blocks=transformer_layers,
                    **_residual_kwargs,
                )
                for i in range(transformer_layers)
            ]
        )

        # AttnRes: finalizer for output aggregation
        _attnres_variant = None
        _attnres_finalizer = None
        if residual_connection_type in ("full_attnres", "block_attnres"):
            from bond_match.modules.attention_residuals import (
                AttnResFinalizer,
                AttnResVariant,
            )

            _attnres_variant = AttnResVariant(residual_connection_type)
            self._attnres_variant = _attnres_variant
            self.attnres_finalizer = AttnResFinalizer(d_model, _attnres_variant)
            _attnres_finalizer = self.attnres_finalizer

        # Residual manager: encapsulates expand/reduce/state logic
        self._residual_mgr = _ResidualManager(
            residual_type=residual_connection_type,
            num_streams=mhc_num_streams,
            attnres_block_size=attnres_block_size,
            stream_manager=_stream_manager,
            attnres_variant=_attnres_variant,
            attnres_finalizer=_attnres_finalizer,
        )

        # Graph attention bias: per-head learned scalars for each hop distance
        if graph_attention_bias:
            self._graph_bias_weights = nn.Parameter(
                torch.zeros(graph_attention_bias_hops, heads)
            )

        self._use_attention_pool_virtual = use_attention_pool_virtual
        if use_attention_pool_virtual:
            self.attention_pool_virtual = AttentionPoolVirtualHead(
                d_model,
                num_heads=heads,
                dropout=dropout,
                max_steps=8,
                use_deep_head=use_deep_virtual_head,
            )

        self._use_step_aware_virtual = use_step_aware_virtual
        if use_deep_virtual_head:
            self.virtual_output_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.virtual_output_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
            )
        if use_step_aware_virtual:
            if use_deep_virtual_head:
                self.virtual_output_head_multi = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
            else:
                self.virtual_output_head_multi = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
                )

        self.step_index_emb = nn.Embedding(8, d_model)
        self.use_pair_score_virtual = use_pair_score_virtual
        virtual_gate_dim = d_model * 3 if use_max_pool_virtual else d_model * 2
        if use_pair_score_virtual:
            self.pair_score_proj = nn.Linear(3, d_model)
            virtual_gate_dim += d_model
        self.virtual_gate = nn.Linear(virtual_gate_dim, d_model)

        self.step_count_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )

        if use_history_step_pos and enable_history_context:
            self.history_step_pos_emb = nn.Embedding(8, d_model)

        self.binary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )

        ps_dropout = pair_scorer_dropout if pair_scorer_dropout > 0 else dropout
        if use_attention_pair_scorer:
            self.pair_scorer = AttentionPairScorer(
                d_model,
                num_heads=heads,
                use_symmetric_pair_scorer=self.use_symmetric_pair_scorer,
            )
        else:
            self.pair_scorer = BondPairScorer(
                d_model,
                hidden_dim=d_model,
                use_symmetric_pair_scorer=self.use_symmetric_pair_scorer,
                dropout=ps_dropout,
            )

        self.fusion_weights = nn.Parameter(torch.ones(3))

        self.pair_scorer_with_reactivity = BondPairScorer(
            d_model + 1,
            hidden_dim=d_model,
            use_symmetric_pair_scorer=self.use_symmetric_pair_scorer,
            dropout=ps_dropout,
        )

        if self.return_type_prediction:
            self.bond_type_classifier = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
            )
            if use_pair_bond_type:
                self._swap_type_head_type = swap_type_head
                if swap_type_head == "bond_split":
                    self.pair_bond_type_classifier = BondSplitSwapHead(
                        d_model,
                        dropout=dropout,
                        use_node_guide=True,
                    )
                elif swap_type_head == "mlp":
                    self.pair_bond_type_classifier = MLPSwapTypeHead(d_model)
                elif swap_type_head == "node_attn":
                    self.pair_bond_type_classifier = NodeAttentionSwapHead(
                        d_model,
                        num_heads=heads,
                        dropout=dropout,
                    )
                else:
                    self.pair_bond_type_classifier = AtomSwapTypeHead(
                        d_model,
                        dropout=dropout,
                    )

        self._init_weights(transformer_layers)

    @classmethod
    def from_config(cls, config, dropout: float | None = None) -> "BondGAT":
        """Construct BondGAT from a RankingConfig.

        Args:
            config: RankingConfig instance.
            dropout: Override dropout (e.g. 0.0 for eval). If None uses default.
        """
        kwargs = dict(
            d_model=config.embed_dim,
            heads=config.num_heads,
            num_layers=config.num_layers,
            atom_max_num=config.atom_max_num,
            num_bond_types=config.num_bond_types,
            transformer_layers=config.transformer_layers,
            return_type_prediction=True,
            extended_atom_features=config.extended_atom_features,
            enable_history_context=config.enable_history_context,
            use_pair_bond_type=config.use_pair_bond_type,
            use_symmetric_pair_scorer=config.use_symmetric_pair_scorer,
            use_max_pool_virtual=config.use_max_pool_virtual,
            use_edge_bond_tokens=config.use_edge_bond_tokens,
            use_history_step_pos=config.use_history_step_pos,
            pair_scorer_dropout=config.pair_scorer_dropout,
            layer_scale_init=config.use_layer_scale,
            bond_proj_mode=config.bond_proj_mode,
            use_pair_score_virtual=config.use_pair_score_virtual,
            use_attention_pair_scorer=config.use_attention_pair_scorer,
            use_deep_virtual_head=config.use_deep_virtual_head,
            use_step_aware_virtual=config.use_step_aware_virtual,
            gat_drop_path=config.gat_drop_path,
            step_embed_bond_proj=config.step_embed_bond_proj,
            residual_connection_type=config.residual_connection_type,
            mhc_num_streams=config.mhc_num_streams,
            use_moe=config.use_moe,
            moe_num_experts=config.moe_num_experts,
            moe_top_k=config.moe_top_k,
            moe_loss_coef=config.moe_loss_coef,
            moe_expert_dim=config.moe_expert_dim,
            moe_expert_dropout=getattr(config, "moe_expert_dropout", 0.0),
            swap_type_head=getattr(config, "swap_type_head", "mlp"),
            attnres_block_size=getattr(config, "attnres_block_size", 6),
            use_attention_pool_virtual=getattr(
                config, "use_attention_pool_virtual", False
            ),
            use_canonical_rank=getattr(config, "use_canonical_rank", False),
            skip_gat=getattr(config, "skip_gat", False),
            graph_attention_bias=getattr(config, "graph_attention_bias", False),
            graph_attention_bias_hops=getattr(config, "graph_attention_bias_hops", 1),
            pair_score_backend=getattr(config, "pair_score_backend", "legacy"),
            pair_score_chunk_size=getattr(config, "pair_score_chunk_size", 512),
            drop_head_bond_tokens_after_gnn=getattr(
                config, "drop_head_bond_tokens_after_gnn", False
            ),
            skip_empty_history_cross_attn_for_history_encoder=getattr(
                config, "skip_empty_history_cross_attn_for_history_encoder", False
            ),
            trajectory_attention=getattr(config, "trajectory_attention", False),
        )
        if dropout is not None:
            kwargs["dropout"] = dropout
        model = cls(**kwargs)
        if getattr(config, "lora_enabled", False):
            from bond_match.modules.lora import apply_lora_to_model

            apply_lora_to_model(
                model,
                target_modules=list(config.lora_target_modules),
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )
        return model

    def _init_weights(self, n_transformer_layers: int):
        """Xavier uniform init for Linear/MHA, normal for Embeddings, scaled residual outputs."""
        gat_params = {id(p) for layer in self.layers for p in layer.parameters()}

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if id(module.weight) not in gat_params:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

        # Scale residual output projections by 1/sqrt(2*n_layers)
        if n_transformer_layers > 0:
            scale = (2 * n_transformer_layers) ** -0.5
            with torch.no_grad():
                for block in self.bond_transformer_blocks:
                    block.self_attn.out_proj.weight.mul_(scale)
                    if block.enable_cross_attention:
                        block.cross_attn.out_proj.weight.mul_(scale)
                    if block.ffn is not None:
                        block.ffn[-1].weight.mul_(scale)

    def _compute_bond_attn_bias(
        self, bond_indices: torch.Tensor, bond_batch_vec: torch.Tensor, max_bonds: int
    ) -> torch.Tensor:
        """Compute per-head adjacency attention bias for the bond transformer.

        Two bonds are adjacent (hop=1) if they share an atom. Higher hops use
        powers of the adjacency matrix. Returns (B*H, N, N) bias tensor
        suitable for nn.MultiheadAttention's attn_mask.
        """
        H = self._graph_bias_weights.size(1)
        n_hops = self._graph_attention_bias_hops
        device = bond_indices.device
        B = bond_batch_vec.max().item() + 1

        bias = torch.zeros(B, H, max_bonds, max_bonds, device=device)
        counts = torch.bincount(bond_batch_vec, minlength=B)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = counts.cumsum(0)

        for mol_idx in range(B):
            n = counts[mol_idx].item()
            s = offsets[mol_idx].item()
            a1, a2 = bond_indices[s : s + n, 0], bond_indices[s : s + n, 1]

            adj = (
                (a1[:, None] == a1[None, :])
                | (a1[:, None] == a2[None, :])
                | (a2[:, None] == a1[None, :])
                | (a2[:, None] == a2[None, :])
            )
            adj.fill_diagonal_(False)
            hop_mat = adj.float()

            for hop in range(n_hops):
                # self._graph_bias_weights[hop] shape: (H,) — per-head scalar
                bias[mol_idx, :, :n, :n] += (
                    self._graph_bias_weights[hop, :, None, None] * hop_mat
                )
                if hop < n_hops - 1:
                    hop_mat = (hop_mat @ adj.float()).clamp(max=1.0)

        return bias.reshape(B * H, max_bonds, max_bonds)

    def _score_pair(self, z_i, z_j):
        return self.pair_scorer(z_i, z_j).squeeze(-1)

    def _legacy_pair_scores_upper_triangle(
        self, z: torch.Tensor, binary_logits: torch.Tensor | None = None, mode: str = "pair_only"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_bonds = z.size(0)
        triu_idx = torch.triu_indices(n_bonds, n_bonds, offset=1, device=z.device)
        if triu_idx.size(1) == 0:
            return torch.empty(0, device=z.device), triu_idx

        pair_scores = self.compute_pair_scores(
            z[triu_idx[0]],
            z[triu_idx[1]],
            binary_logits[triu_idx[0]] if binary_logits is not None else None,
            binary_logits[triu_idx[1]] if binary_logits is not None else None,
            mode=mode,
        )
        return pair_scores, triu_idx

    def compute_pair_scores_upper_triangle(
        self, z: torch.Tensor, binary_logits: torch.Tensor | None = None, mode: str = "pair_only"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute upper-triangle pair scores in the same order as torch.triu_indices."""
        if (
            self._pair_score_backend != "exact_attention_chunked"
            or not isinstance(self.pair_scorer, AttentionPairScorer)
            or mode == "pair_absorbs_binary"
        ):
            return self._legacy_pair_scores_upper_triangle(z, binary_logits, mode)

        triu_idx = torch.triu_indices(z.size(0), z.size(0), offset=1, device=z.device)
        if triu_idx.size(1) == 0:
            return torch.empty(0, device=z.device), triu_idx

        pair_raw = self.pair_scorer.score_upper_triangle(
            z, chunk_size=self._pair_score_chunk_size
        )
        if mode == "pair_only":
            return pair_raw, triu_idx

        react = (
            torch.sigmoid(binary_logits.squeeze(-1))
            if binary_logits is not None
            else torch.zeros(z.size(0), device=z.device)
        )
        react_i = react[triu_idx[0]]
        react_j = react[triu_idx[1]]

        if mode == "fused":
            return pair_raw + react_i + react_j, triu_idx

        if mode == "learned_fusion":
            w = torch.softmax(self.fusion_weights, dim=0) * 3
            return w[0] * pair_raw + w[1] * react_i + w[2] * react_j, triu_idx

        return pair_raw, triu_idx

    def compute_pair_scores(
        self, z_i, z_j, binary_logits_i=None, binary_logits_j=None, mode="pair_only"
    ):
        """Compute pair scores using the specified scoring mode."""
        if mode == "pair_only":
            return self._score_pair(z_i, z_j)

        react_i = (
            torch.sigmoid(binary_logits_i.squeeze(-1))
            if binary_logits_i is not None
            else torch.zeros(z_i.size(0), device=z_i.device)
        )
        react_j = (
            torch.sigmoid(binary_logits_j.squeeze(-1))
            if binary_logits_j is not None
            else torch.zeros(z_j.size(0), device=z_j.device)
        )

        if mode == "fused":
            pair_raw = self._score_pair(z_i, z_j)
            return pair_raw + react_i + react_j

        if mode == "learned_fusion":
            w = torch.softmax(self.fusion_weights, dim=0) * 3
            pair_raw = self._score_pair(z_i, z_j)
            return w[0] * pair_raw + w[1] * react_i + w[2] * react_j

        if mode == "pair_absorbs_binary":
            z_i_ext = torch.cat([z_i, react_i.unsqueeze(-1)], dim=-1)
            z_j_ext = torch.cat([z_j, react_j.unsqueeze(-1)], dim=-1)
            return self.pair_scorer_with_reactivity(z_i_ext, z_j_ext).squeeze(-1)

        return self.pair_scorer(z_i, z_j).squeeze(-1)

    def _lookup_bond_edge_data(
        self,
        bond_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if edge_index is None or edge_index.numel() == 0 or bond_indices.numel() == 0:
            return None, None

        # Keep max_n on-device (0-d tensor) to avoid host-device syncs per forward.
        max_n = (
            torch.maximum(bond_indices.max(), edge_index.max()) + 1
        )
        edge_u = torch.minimum(edge_index[0], edge_index[1])
        edge_v = torch.maximum(edge_index[0], edge_index[1])
        bond_u = torch.minimum(bond_indices[:, 0], bond_indices[:, 1])
        bond_v = torch.maximum(bond_indices[:, 0], bond_indices[:, 1])

        edge_hash = edge_u * max_n + edge_v
        bond_hash = bond_u * max_n + bond_v
        sorted_hash, sort_perm = edge_hash.sort()
        pos = torch.searchsorted(sorted_hash, bond_hash).clamp(max=sorted_hash.size(0) - 1)
        found = sorted_hash[pos] == bond_hash
        perm = sort_perm[pos]

        bond_edge_types = None
        if edge_attr is not None:
            bond_edge_types = edge_attr[perm, 0].clone()
            bond_edge_types[~found] = -1

        bond_edge_feat = None
        if edge_embeddings is not None:
            bond_edge_feat = edge_embeddings[perm].clone()
            bond_edge_feat[~found] = 0.0

        return bond_edge_types, bond_edge_feat

    def _node_repr(
        self,
        x,
        batch_idx,
        edge_index,
        edge_attr,
        step_index=None,
    ):
        """Phase 1: Compute node representations via GATv2Conv layers."""
        charges = (x[:, 1] + 5).clamp(0, 19).long()
        if step_index is None:
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
            step_index = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
        is_virtual = (step_index > 0).long()
        emb_parts = [
            self.atom_emb(x[:, 0].clamp(0, self.atom_emb.num_embeddings - 1).long()),
            self.charge_emb(charges.long()),
            self.h_count_emb(
                x[:, 2].clamp(0, self.h_count_emb.num_embeddings - 1).long()
            ),
            self.aromatic_emb(
                x[:, 3].clamp(0, self.aromatic_emb.num_embeddings - 1).long()
            ),
            self.symmetry_emb(x[:, 4].float().unsqueeze(-1)),
            self.is_virtual_emb(is_virtual)[batch_idx],
            self.step_idx_node_emb(step_index.clamp(0, 7))[batch_idx],
        ]
        feat_offset = 5
        if self._use_canonical_rank:
            emb_parts.append(
                self.canonical_rank_emb(x[:, feat_offset].float().unsqueeze(-1))
            )
            feat_offset += 1
        if self.extended_atom_features:
            emb_parts.extend(
                [
                    self.ring_emb(
                        x[:, feat_offset]
                        .clamp(0, self.ring_emb.num_embeddings - 1)
                        .long()
                    ),
                    self.ring_size_emb(x[:, feat_offset + 1].clamp(0, 9).long()),
                    self.degree_emb(x[:, feat_offset + 2].clamp(0, 7).long()),
                    self.hybridization_emb(x[:, feat_offset + 3].clamp(0, 7).long()),
                    self.catalyst_emb(
                        x[:, feat_offset + 4]
                        .clamp(0, self.catalyst_emb.num_embeddings - 1)
                        .long()
                    ),
                ]
            )
        h = torch.cat(emb_parts, dim=-1)
        h = self.node_proj(h)

        edge_embeddings = self.edge_emb(
            edge_attr.squeeze(-1).clamp(0, self.edge_emb.num_embeddings - 1)
        )

        if not self._skip_gat:
            for i, (layer, norm) in enumerate(zip(self.layers, self.gat_norms)):
                h_new = layer(h, edge_index, edge_attr=edge_embeddings)

                if self.training and self._gat_drop_path > 0:
                    drop_prob = self._drop_probs[i]
                    if (
                        drop_prob > 0
                        and torch.rand(1, device=h.device).item() < drop_prob
                    ):
                        continue

                if self._gat_layer_scale:
                    h = norm(h + self.gat_gammas[i] * h_new)
                else:
                    h = norm(h + h_new)
                h = torch.nn.functional.gelu(h)

        return h, edge_index, edge_embeddings

    def _bond_tokens(
        self,
        h,
        batch_idx,
        bond_indices,
        edge_index=None,
        edge_attr=None,
        edge_embeddings=None,
        step_index=None,
    ):
        """Phase 2: Compute bond tokens from node representations."""
        bond_indices_out = bond_indices
        bond_edge_types = None
        bond_edge_feat = None
        if edge_index is not None and (
            self.use_edge_bond_tokens or self._drop_head_bond_tokens_after_gnn
        ):
            bond_edge_types, bond_edge_feat = self._lookup_bond_edge_data(
                bond_indices,
                edge_index,
                edge_attr=edge_attr,
                edge_embeddings=edge_embeddings,
            )

        keep_mask = None
        if self._drop_head_bond_tokens_after_gnn:
            if bond_edge_types is None:
                raise RuntimeError(
                    "Head-token filtering requires edge_attr to resolve bond types."
                )
            keep_mask = bond_edge_types != 0
            bond_indices_out = bond_indices[keep_mask]
        else:
            keep_mask = torch.ones(
                bond_indices.size(0), dtype=torch.bool, device=bond_indices.device
            )

        u_idx = bond_indices_out[:, 0]
        v_idx = bond_indices_out[:, 1]
        bond_batch_vec = batch_idx[u_idx]

        h_u = h[u_idx]
        h_v = h[v_idx]
        pair_feat = h_u * h_v

        parts = [h_u, h_v, pair_feat]

        if (
            self.use_edge_bond_tokens
            and edge_index is not None
            and edge_embeddings is not None
        ):
            if bond_edge_feat is None:
                _, bond_edge_feat = self._lookup_bond_edge_data(
                    bond_indices,
                    edge_index,
                    edge_attr=edge_attr,
                    edge_embeddings=edge_embeddings,
                )
            if self._drop_head_bond_tokens_after_gnn:
                bond_edge_feat = bond_edge_feat[keep_mask]
            parts.append(bond_edge_feat)

        if self._step_embed_bond_proj and step_index is not None:
            step_emb = self.step_index_emb(step_index.clamp(0, 7))
            parts.append(step_emb[bond_batch_vec])

        raw_bond_tokens = torch.cat(parts, dim=-1)

        bond_tokens = self.bond_proj(raw_bond_tokens)
        if self._bond_proj_mode == "residual":
            bond_tokens = self.bond_proj_skip(bond_tokens + pair_feat)

        return bond_tokens, bond_batch_vec, bond_indices_out, keep_mask

    def encode_bonds(
        self,
        x,
        batch_idx,
        edge_index,
        edge_attr,
        bond_indices,
        step_index=None,
    ):
        """Run backbone + bond tokenization only. Returns (bond_tokens, bond_batch_vec).

        Returns pre-transformer bond tokens. For history encoding prefer
        encode_history_tokens() which also runs self-attention.
        """
        h, ei, ee = self._node_repr(
            x, batch_idx, edge_index, edge_attr, step_index=step_index
        )
        return self._bond_tokens(
            h,
            batch_idx,
            bond_indices,
            edge_index=ei,
            edge_attr=edge_attr,
            edge_embeddings=ee,
            step_index=step_index,
        )

    def empty_history(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build empty history context with a single zero token per sample.

        Returns (context, mask) with shape (B, 1, d_model) and (B, 1).
        The context token is zero-valued and the mask is all-True (= masked).
        BondTransformerBlock's all_masked fix unmasks it, so cross-attention
        runs on zeros and norm_cross is applied -- matching the behavior
        step-0 samples see in mixed training batches.
        """
        ctx = torch.zeros(batch_size, 1, self.d_model, device=device)
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        return ctx, mask

    def _pack_trajectory_tokens(
        self,
        bond_tokens: torch.Tensor,
        bond_batch_vec: torch.Tensor,
        trajectory_ids: torch.Tensor,
        step_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = bond_tokens.device
        D = bond_tokens.size(-1)

        # One row per unique trajectory id, ascending. traj_inverse maps each
        # state -> its row index.
        traj_unique, traj_inverse = torch.unique(
            trajectory_ids, sorted=True, return_inverse=True
        )
        B_traj = int(traj_unique.numel())
        empty_dense = bond_tokens.new_zeros((B_traj, 0, D))
        empty_mask = torch.zeros((B_traj, 0), dtype=torch.bool, device=device)
        empty_steps = torch.zeros((B_traj, 0), dtype=torch.long, device=device)
        empty_order = torch.empty(0, dtype=torch.long, device=device)
        empty_groups = torch.zeros((B_traj, 0), dtype=torch.long, device=device)

        if bond_tokens.numel() == 0 or bond_batch_vec.numel() == 0:
            return empty_dense, empty_mask, empty_steps, empty_order, empty_groups

        # Order states by (trajectory_rank, step). state_rank[s] is the global
        # rank of state s in this composite ordering — tying bonds together
        # within a state and steps in ascending order within a trajectory.
        S = trajectory_ids.numel()
        s_traj_rank = traj_inverse.to(torch.long)
        s_step = step_index.to(torch.long)
        state_stride = s_step.max() + 1
        state_key = s_traj_rank * state_stride + s_step
        state_order = torch.argsort(state_key, stable=True)
        state_rank = torch.empty(S, dtype=torch.long, device=device)
        state_rank[state_order] = torch.arange(S, device=device, dtype=torch.long)

        # Stable sort bonds by state_rank; ties use the input bond index to
        # preserve within-state ordering when bond_batch_vec is interleaved.
        b_state_rank = state_rank[bond_batch_vec]
        sort_idx = torch.argsort(b_state_rank, stable=True)

        sorted_traj_rank = s_traj_rank[bond_batch_vec[sort_idx]]
        sorted_step = s_step[bond_batch_vec[sort_idx]]

        counts = torch.bincount(sorted_traj_rank, minlength=B_traj)
        max_len = int(counts.max().item()) if counts.numel() > 0 else 0

        if max_len == 0:
            return empty_dense, empty_mask, empty_steps, empty_order, empty_groups

        row_starts = torch.zeros(B_traj, dtype=torch.long, device=device)
        if B_traj > 1:
            row_starts[1:] = counts.cumsum(0)[:-1]
        positions = (
            torch.arange(sort_idx.numel(), device=device, dtype=torch.long)
            - row_starts[sorted_traj_rank]
        )

        dense = bond_tokens.new_zeros((B_traj, max_len, D))
        mask = torch.zeros((B_traj, max_len), dtype=torch.bool, device=device)
        token_steps = torch.zeros((B_traj, max_len), dtype=torch.long, device=device)
        # Per-position state id (= bond_batch_vec for the bond at that position).
        # Used by MoE to compute capacity per state instead of per packed row.
        group_ids = torch.zeros((B_traj, max_len), dtype=torch.long, device=device)

        dense[sorted_traj_rank, positions] = bond_tokens[sort_idx]
        mask[sorted_traj_rank, positions] = True
        token_steps[sorted_traj_rank, positions] = sorted_step
        group_ids[sorted_traj_rank, positions] = bond_batch_vec[sort_idx]

        return dense, mask, token_steps, sort_idx.to(torch.long), group_ids

    def _trajectory_causal_attn_mask(self, token_steps: torch.Tensor) -> torch.Tensor:
        future = token_steps[:, None, :] > token_steps[:, :, None]
        return future.repeat_interleave(self.num_heads, dim=0)

    def encode_history_tokens(
        self,
        x,
        batch_idx,
        edge_index,
        edge_attr,
        bond_indices,
        step_index=None,
        history_context=None,
        history_mask=None,
    ):
        """Encode history graphs through the full forward path (GAT + bond proj + transformer).

        Uses the same transformer path as forward() — including cross-attention
        if history_context is provided — so train and eval produce identical
        history representations. Pass empty history for step-0 graphs.

        Returns (bond_tokens, bond_batch_vec).
        """
        h, ei, ee = self._node_repr(
            x,
            batch_idx,
            edge_index,
            edge_attr,
            step_index=step_index,
        )
        bond_tokens, bond_batch_vec, bond_indices_out, keep_mask = self._bond_tokens(
            h,
            batch_idx,
            bond_indices,
            edge_index=ei,
            edge_attr=edge_attr,
            edge_embeddings=ee,
            step_index=step_index,
        )
        dense_bonds, mask = to_dense_batch(bond_tokens, bond_batch_vec)

        mgr = self._residual_mgr
        pad_mask = mgr.build_pad_mask(mask)
        history_context, history_mask = mgr.expand_context(
            history_context, history_mask
        )
        if (
            self._skip_empty_history_cross_attn_for_history_encoder
            and history_mask is not None
            and history_mask.ndim == 2
            and history_mask.size(1) == 1
            and history_mask.all()
        ):
            history_context = None
            history_mask = None
        attn_output = mgr.expand(dense_bonds)

        attn_bias = None
        if self._graph_attention_bias:
            attn_bias = self._compute_bond_attn_bias(
                bond_indices_out, bond_batch_vec, dense_bonds.size(1)
            )

        attn_output = mgr.run_transformer(
            self.bond_transformer_blocks,
            attn_output,
            dense_bonds,
            pad_mask,
            history_context=history_context,
            history_mask=history_mask,
            attn_bias=attn_bias,
        )
        attn_output = mgr.reduce(attn_output)

        bond_context = attn_output[mask]
        return bond_context, bond_batch_vec

    def forward(
        self,
        x,
        batch_idx,
        edge_index,
        edge_attr,
        bond_indices,
        history_context=None,
        history_mask=None,
        step_index=None,
        trajectory_ids=None,
    ):

        assert step_index is not None, (
            "step_index must be provided for forward pass to ensure correct node embeddings (even if all zeros for single-step)."
        )

        if self.enable_history_context:
            assert history_context is not None, (
                "history_context must be provided when enable_history_context=True (pass empty [B,0,d] for step-0)"
            )
            assert history_mask is not None, (
                "history_mask must be provided when enable_history_context=True (pass empty [B,0] for step-0)"
            )

        h, ei, ee = self._node_repr(
            x,
            batch_idx,
            edge_index,
            edge_attr,
            step_index=step_index,
        )

        bond_tokens, bond_batch_vec, bond_indices_out, keep_mask = self._bond_tokens(
            h,
            batch_idx,
            bond_indices,
            edge_index=ei,
            edge_attr=edge_attr,
            edge_embeddings=ee,
            step_index=step_index,
        )

        use_trajectory_attention = (
            self._trajectory_attention
            and trajectory_ids is not None
            and trajectory_ids.numel() > 0
        )
        if use_trajectory_attention:
            dense_bonds, mask, token_steps, flat_order, group_ids = (
                self._pack_trajectory_tokens(
                    bond_tokens,
                    bond_batch_vec,
                    trajectory_ids,
                    step_index,
                )
            )
        else:
            dense_bonds, mask = to_dense_batch(bond_tokens, bond_batch_vec)
            token_steps = None
            flat_order = None
            group_ids = None


        mgr = self._residual_mgr
        pad_mask = mgr.build_pad_mask(mask)
        history_context, history_mask = mgr.expand_context(
            history_context, history_mask
        )
        attn_output = mgr.expand(dense_bonds)

        attn_bias = None
        if self._graph_attention_bias:
            attn_bias = self._compute_bond_attn_bias(
                bond_indices_out, bond_batch_vec, dense_bonds.size(1)
            )
        if use_trajectory_attention:
            if attn_bias is not None:
                raise RuntimeError(
                    "trajectory_attention does not currently support graph_attention_bias."
                )
            attn_bias = self._trajectory_causal_attn_mask(token_steps)

        attn_output = mgr.run_transformer(
            self.bond_transformer_blocks,
            attn_output,
            dense_bonds,
            pad_mask,
            history_context=history_context,
            history_mask=history_mask,
            attn_bias=attn_bias,
            group_ids=group_ids,
        )
        attn_output = mgr.reduce(attn_output)

        if use_trajectory_attention:
            packed_context = attn_output[mask]
            bond_context = torch.empty_like(bond_tokens)
            bond_context[flat_order] = packed_context
            state_dense_bonds, state_mask = to_dense_batch(bond_context, bond_batch_vec)
        else:
            bond_context = attn_output[mask]
            state_dense_bonds, state_mask = attn_output, mask

        graph_bond_repr = global_mean_pool(bond_context, bond_batch_vec)

        logits_binary = self.binary_head(bond_context)

        if self._use_attention_pool_virtual and step_index is not None:
            logits_stop = self.attention_pool_virtual(
                bond_context,
                bond_batch_vec,
                step_index,
                dense_bonds=state_dense_bonds,
                mask=state_mask,
            )
        elif step_index is not None:
            step_emb = self.step_index_emb(step_index.clamp(0, 7))
            gate_parts = [graph_bond_repr]
            if self.use_max_pool_virtual:
                graph_bond_max = global_max_pool(bond_context, bond_batch_vec)
                gate_parts.append(graph_bond_max)
            gate_parts.append(step_emb)

            if self.use_pair_score_virtual:
                B = graph_bond_repr.size(0)
                binary_react = torch.sigmoid(logits_binary.squeeze(-1))
                max_react = global_max_pool(
                    binary_react.unsqueeze(-1), bond_batch_vec
                ).squeeze(-1)

                max_ps_list = []
                mean_ps_list = []
                for mol_id in range(B):
                    mol_mask_flat = bond_batch_vec == mol_id
                    z_mol = bond_context[mol_mask_flat]
                    n = z_mol.size(0)
                    if n >= 2:
                        ti = torch.triu_indices(
                            n, n, offset=1, device=bond_context.device
                        )
                        ps = self.pair_scorer(z_mol[ti[0]], z_mol[ti[1]]).squeeze(-1)
                        max_ps_list.append(ps.max())
                        mean_ps_list.append(ps.mean())
                    else:
                        max_ps_list.append(
                            torch.tensor(0.0, device=bond_context.device)
                        )
                        mean_ps_list.append(
                            torch.tensor(0.0, device=bond_context.device)
                        )
                max_ps = torch.stack(max_ps_list)
                mean_ps = torch.stack(mean_ps_list)

                ps_stats = torch.stack([max_ps, mean_ps, max_react], dim=-1)  # [B, 3]
                gate_parts.append(self.pair_score_proj(ps_stats))

            virtual_input = self.virtual_gate(torch.cat(gate_parts, dim=-1))
            if self._use_step_aware_virtual:
                logits_step0 = self.virtual_output_head(virtual_input)
                logits_multi = self.virtual_output_head_multi(virtual_input)
                is_multi = (step_index >= 1).float().unsqueeze(-1)  # [B, 1]
                logits_stop = logits_step0 * (1 - is_multi) + logits_multi * is_multi
            else:
                logits_stop = self.virtual_output_head(virtual_input)
        else:
            logits_stop = self.virtual_output_head(graph_bond_repr)

        out_dict = {}

        # Collect MoE auxiliary losses from transformer blocks
        moe_loss = sum(
            block.moe_aux_loss
            for block in self.bond_transformer_blocks
            if hasattr(block, "moe_aux_loss") and block._use_moe
        )
        if isinstance(moe_loss, torch.Tensor):
            out_dict["moe_aux_loss"] = moe_loss

        out_dict["logits_binary"] = logits_binary
        out_dict["embeddings"] = bond_context
        out_dict["graph_bond_repr"] = graph_bond_repr
        out_dict["graph_node_repr"] = global_mean_pool(h, batch_idx)
        out_dict["logits_step_count"] = self.step_count_head(graph_bond_repr)
        out_dict["node_embeddings"] = h
        out_dict["bond_indices_out"] = bond_indices_out
        out_dict["bond_keep_mask"] = keep_mask
        out_dict["edge_embeddings"] = ee
        out_dict["edge_index"] = ei
        out_dict["trajectory_attention"] = use_trajectory_attention

        out_dict["logits_stop"] = logits_stop
        out_dict["logits_virtual"] = logits_stop

        if self.return_type_prediction:
            out_dict["logits_bond_type"] = self.bond_type_classifier(graph_bond_repr)

        return out_dict
