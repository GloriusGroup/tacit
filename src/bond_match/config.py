"""Pydantic configuration models for ranking-model training.

The top-level RankingConfig is organized into nested sub-configs
(ModelConfig, LossConfig, TrainConfig, EvalConfig, DataConfig, LoRAConfig).
Flat YAML configs without a top-level ``model`` key are restructured into
nested form by a model validator. ``load_config`` and ``save_config`` handle
YAML I/O.
"""

from typing import Literal, Optional
from enum import Enum
from pydantic import BaseModel, Field, model_validator
import yaml


class DetectionType(str, Enum):
    """Supported detection task types."""

    RANKING = "ranking"


class DataVersion(str, Enum):
    """Dataset format versions."""

    V2_GRAPH = "v2_graph"
    V3_BLOCKSPACE = "v3_blockspace"


class PairSelectionMode(str, Enum):
    """Bond-pair selection strategies used during evaluation."""

    ARGMAX = "argmax"
    SOFTMAX = "softmax"


class ResumeSchedulerMode(str, Enum):
    """Scheduler restoration modes used when resuming training."""

    CHECKPOINT = "checkpoint"
    REMAINING = "remaining"


class ModelConfig(BaseModel):
    """Architecture parameters for the BondGAT model.

    Covers core dimensions (``embed_dim``, ``num_heads``, ``num_layers``,
    ``transformer_layers``, ``hidden_dim``), boolean feature flags for
    optional architecture variants, and pair-scorer settings.
    ``use_canonical_rank`` adds a canonical atom rank feature for swap-type
    prediction.

    ``residual_connection_type`` accepts "add", "mhc", "block_attnres", or
    "full_attnres". For the mixture-of-experts settings, ``moe_expert_dim``
    is the per-expert dimension (None means equal to ``embed_dim``) and
    ``moe_expert_dropout`` is the dropout applied inside expert feed-forward
    networks (Switch Transformer recommends 0.3-0.4 for fine-tuning).
    ``skip_gat`` bypasses the GAT layers; ``graph_attention_bias`` encodes
    graph topology via an adjacency-based attention bias instead.
    """

    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    transformer_layers: int = 2
    hidden_dim: int = 256
    atom_max_num: int = 100
    num_bond_types: int = 5

    extended_atom_features: bool = False
    enable_history_context: bool = False
    use_pair_bond_type: bool = False
    swap_type_head: str = "mlp"
    use_symmetric_pair_scorer: bool = False
    use_max_pool_virtual: bool = False
    use_edge_bond_tokens: bool = False
    use_history_step_pos: bool = False
    use_attention_pair_scorer: bool = False
    use_deep_virtual_head: bool = False
    use_step_aware_virtual: bool = False
    use_pair_score_virtual: bool = False
    use_attention_pool_virtual: bool = False

    use_canonical_rank: bool = False

    pair_scorer_dropout: float = 0.0
    bond_proj_mode: str = "default"
    use_layer_scale: float = 0.0
    gat_drop_path: float = 0.0
    step_embed_bond_proj: bool = False
    attention_score_scale: float = 0.0
    pair_score_backend: str = "legacy"
    pair_score_chunk_size: int = 512
    drop_head_bond_tokens_after_gnn: bool = False
    skip_empty_history_cross_attn_for_history_encoder: bool = False
    trajectory_attention: bool = False

    residual_connection_type: str = "add"
    mhc_num_streams: int = 2
    attnres_block_size: int = 6

    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_loss_coef: float = 0.01
    moe_expert_dim: Optional[int] = None
    moe_expert_dropout: float = 0.0

    skip_gat: bool = False
    graph_attention_bias: bool = False
    graph_attention_bias_hops: int = 1

    class Config:
        """Pydantic settings; unknown fields are ignored during validation."""

        extra = "ignore"


class LossConfig(BaseModel):
    """Loss function parameters.

    Fields from ``step_count_aux_weight`` onward configure auxiliary loss
    terms.
    """

    ranking_loss_type: str = "softmax"
    temperature: float = 0.1
    margin: float = 0.5
    lambda_anchor: float = 0.05
    train_no_reaction_class: bool = False
    negative_reaction_weight: float = 0.0
    beta: float = 0.5
    bond_type_weight: float = 0.1
    binary_weight: float = 1.0
    hard_negative_k: int = 0
    virtual_pos_weight: float = 1.0
    label_smoothing: float = 0.0
    pair_scoring_mode: str = "pair_only"
    step_count_aux_weight: float = 0.0
    pair_score_target: float = 0.0
    multistep_loss_scale: float = 1.0
    bond_contrastive_weight: float = 0.0
    virtual_focal_gamma: float = 0.0
    binary_focal_gamma: float = 0.0
    virtual_consistency_weight: float = 0.0
    binary_virtual_consistency_weight: float = 0.0
    virtual_margin: float = 0.0
    stop_rank_aux_weight: float = 0.0

    class Config:
        """Pydantic settings; unknown fields are ignored during validation."""

        extra = "ignore"


class TrainConfig(BaseModel):
    """Optimizer, scheduler, and training-loop parameters.

    ``optimizer_type`` selects "adamw" or "muon_clip"; the ``muon_*`` fields
    apply only to the latter. ``moe_freeze_router`` freezes
    mixture-of-experts gate weights during fine-tuning, and
    ``moe_router_lr_scale`` is a learning-rate multiplier for the router
    (for example, 0.05 trains the router 20x slower).
    """

    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 20
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    resume_scheduler_mode: ResumeSchedulerMode = ResumeSchedulerMode.REMAINING
    warmup_steps: int = 0
    warmup_batches: int = 0
    eta_min: float = 1e-6
    virtual_lr_scale: float = 1.0
    moe_freeze_router: bool = False
    moe_router_lr_scale: float = 1.0
    dtype: str = "float32"
    num_gpus: int = 1
    ema_decay: float = 0.0
    optimizer_type: str = "adamw"
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_update_clip: float = 5.0
    muon_all_params: bool = False
    num_dataloader_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    torch_compile: bool = False
    save_every_epoch: bool = False
    drop_product_history_prob: float = 0.0

    class Config:
        """Pydantic settings; unknown fields are ignored during validation."""

        extra = "ignore"


class EvalConfig(BaseModel):
    """Inference and evaluation parameters.

    For ``pair_softmax_topk``, 0 applies a plain softmax over all pairs and
    a positive value restricts the softmax to the top-k pairs only.
    """

    max_eval_steps: int = 5
    eval_beam_width: int = 1
    pair_selection_mode: PairSelectionMode = PairSelectionMode.ARGMAX
    pair_selection_temperature: float = 1.0
    pair_softmax_topk: int = 0
    eval_every_n_batches: Optional[int] = None

    class Config:
        """Pydantic settings; unknown fields are ignored during validation."""

        extra = "ignore"


class DataConfig(BaseModel):
    """Dataset paths and loading parameters."""

    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    data_version: DataVersion = DataVersion.V2_GRAPH
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None
    test_limit: Optional[int] = None
    force_recalculate_cache: bool = False
    shard_size: int = 50_000
    include_product: bool = False
    photo_oversampling_factor: float = 1.0
    multistep_oversampling_factor: float = 1.0
    late_step_oversampling_factor: float = 1.0
    product_sampling_factor: float = 1.0
    enable_blockspace_product: bool = True
    history_grad: bool = False
    history_noise_std: float = 0.0
    history_noise_ramp_batches: int = 0
    sample_mode: str = "step"

    class Config:
        """Pydantic settings; unknown fields are ignored during validation."""

        extra = "ignore"


class LoRAConfig(BaseModel):
    """LoRA finetuning configuration."""

    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    freeze_base_model: bool = True
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "node_proj",
            "bond_proj",
            "bond_transformer_blocks",
            "symmetry_emb",
            "canonical_rank_emb",
            "pair_score_proj",
            "pair_scorer",
            "pair_scorer_with_reactivity",
            "virtual_output_head",
            "virtual_output_head_multi",
            "virtual_gate",
            "step_count_head",
            "binary_head",
            "bond_type_classifier",
            "pair_bond_type_classifier",
            "attention_pool_virtual",
        ]
    )

    class Config:
        """Pydantic settings; unknown fields are ignored during validation."""

        extra = "ignore"


_MODEL_FIELDS = set(ModelConfig.model_fields.keys())
_LOSS_FIELDS = set(LossConfig.model_fields.keys())
_TRAIN_FIELDS = set(TrainConfig.model_fields.keys())
_EVAL_FIELDS = set(EvalConfig.model_fields.keys())
_DATA_FIELDS = set(DataConfig.model_fields.keys())

_TOP_LEVEL_FIELDS = {
    "detection_type",
    "experiment_name",
    "disable_mlflow",
    "load_from_checkpoint",
    "checkpoint_path",
    "save_checkpoint_dir",
    "resume_training_state",
    "resume_from_batch",
    "sweep_limit",
    "sweep_epochs",
    "n_trials",
}

_IGNORED_FIELDS = {
    "backbone",
    "enable_matrix_output",
    "use_all_mechanisms",
    "use_split_attn",
    "use_self_cross_attention",
    "use_ensemble_pair_scorer",
    "use_attn_nonlinear_head",
    "bond_graph_conv_layers",
    "refine_edges",
    "enrich_nodes_from_bonds",
    "use_engram_gat",
    "use_engram_transformer",
    "engram_vocab_size",
    "engram_embed_dim",
    "engram_gat_layer_ids",
    "engram_transformer_layer_ids",
    "use_learnable_tau",
}


def _restructure_flat_config(values: dict) -> dict:
    """Convert a flat YAML dict into the nested sub-config structure.

    Keys are routed into the ``model``, ``loss``, ``train``, ``eval``, and
    ``data`` sections according to the field names of the corresponding
    sub-config classes. Keys in ``_IGNORED_FIELDS`` are ignored; keys in
    ``_TOP_LEVEL_FIELDS`` (fields defined directly on RankingConfig) and any
    unrecognized keys remain at the top level.

    Args:
        values: Flat configuration dictionary.

    Returns:
        Nested configuration dictionary with sub-config sections.
    """
    model_d, loss_d, train_d, eval_d, data_d = {}, {}, {}, {}, {}
    top = {}

    for k, v in values.items():
        if k in _IGNORED_FIELDS:
            continue
        if k in _TOP_LEVEL_FIELDS:
            top[k] = v
        elif k in _MODEL_FIELDS:
            model_d[k] = v
        elif k in _LOSS_FIELDS:
            loss_d[k] = v
        elif k in _TRAIN_FIELDS:
            train_d[k] = v
        elif k in _EVAL_FIELDS:
            eval_d[k] = v
        elif k in _DATA_FIELDS:
            data_d[k] = v
        else:
            top[k] = v

    top["model"] = model_d
    top["loss"] = loss_d
    top["train"] = train_d
    top["eval"] = eval_d
    top["data"] = data_d
    return top


class RankingConfig(BaseModel):
    """Top-level configuration for ranking-based bond-pair prediction.

    Composed of nested sub-configs (``model``, ``loss``, ``train``, ``eval``,
    ``data``, ``lora``) plus experiment-level fields (MLflow, checkpointing)
    and Optuna sweep parameters. Inputs without a top-level ``model`` key are
    treated as flat configs and restructured into nested sections. Property
    proxies expose nested fields at the top level (for example,
    ``config.embed_dim`` resolves to ``config.model.embed_dim``).
    """

    detection_type: Literal[DetectionType.RANKING] = DetectionType.RANKING

    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    data: DataConfig = DataConfig()
    lora: LoRAConfig = LoRAConfig()

    experiment_name: Optional[str] = None
    disable_mlflow: bool = False
    load_from_checkpoint: bool = False
    checkpoint_path: Optional[str] = None
    save_checkpoint_dir: Optional[str] = None
    resume_training_state: bool = True
    resume_from_batch: int = -1

    sweep_limit: int = 5000
    sweep_epochs: int = 5
    n_trials: int = 30

    @model_validator(mode="before")
    @classmethod
    def _flatten_compat(cls, values):
        """Restructure flat config dicts into nested form.

        Any dict without a top-level ``model`` key is passed through
        ``_restructure_flat_config``; partial nested dicts must therefore
        include a ``model`` key to bypass this restructuring.

        Args:
            values: Raw input passed to model validation.

        Returns:
            The restructured dict, or the input unchanged if it is not a
            flat-config dict.
        """
        if isinstance(values, dict) and "model" not in values:
            return _restructure_flat_config(values)
        return values

    @property
    def embed_dim(self):
        """Alias for ``model.embed_dim``."""
        return self.model.embed_dim

    @property
    def num_heads(self):
        """Alias for ``model.num_heads``."""
        return self.model.num_heads

    @property
    def num_layers(self):
        """Alias for ``model.num_layers``."""
        return self.model.num_layers

    @property
    def transformer_layers(self):
        """Alias for ``model.transformer_layers``."""
        return self.model.transformer_layers

    @property
    def hidden_dim(self):
        """Alias for ``model.hidden_dim``."""
        return self.model.hidden_dim

    @property
    def atom_max_num(self):
        """Alias for ``model.atom_max_num``."""
        return self.model.atom_max_num

    @property
    def num_bond_types(self):
        """Alias for ``model.num_bond_types``."""
        return self.model.num_bond_types

    @property
    def extended_atom_features(self):
        """Alias for ``model.extended_atom_features``."""
        return self.model.extended_atom_features

    @property
    def enable_history_context(self):
        """Alias for ``model.enable_history_context``."""
        return self.model.enable_history_context

    @property
    def use_pair_bond_type(self):
        """Alias for ``model.use_pair_bond_type``."""
        return self.model.use_pair_bond_type

    @property
    def swap_type_head(self):
        """Alias for ``model.swap_type_head``."""
        return self.model.swap_type_head

    @property
    def use_symmetric_pair_scorer(self):
        """Alias for ``model.use_symmetric_pair_scorer``."""
        return self.model.use_symmetric_pair_scorer

    @property
    def use_max_pool_virtual(self):
        """Alias for ``model.use_max_pool_virtual``."""
        return self.model.use_max_pool_virtual

    @property
    def use_edge_bond_tokens(self):
        """Alias for ``model.use_edge_bond_tokens``."""
        return self.model.use_edge_bond_tokens

    @property
    def use_history_step_pos(self):
        """Alias for ``model.use_history_step_pos``."""
        return self.model.use_history_step_pos

    @property
    def use_attention_pair_scorer(self):
        """Alias for ``model.use_attention_pair_scorer``."""
        return self.model.use_attention_pair_scorer

    @property
    def use_deep_virtual_head(self):
        """Alias for ``model.use_deep_virtual_head``."""
        return self.model.use_deep_virtual_head

    @property
    def use_step_aware_virtual(self):
        """Alias for ``model.use_step_aware_virtual``."""
        return self.model.use_step_aware_virtual

    @property
    def use_pair_score_virtual(self):
        """Alias for ``model.use_pair_score_virtual``."""
        return self.model.use_pair_score_virtual

    @property
    def use_attention_pool_virtual(self):
        """Alias for ``model.use_attention_pool_virtual``."""
        return self.model.use_attention_pool_virtual

    @property
    def use_canonical_rank(self):
        """Alias for ``model.use_canonical_rank``."""
        return self.model.use_canonical_rank

    @property
    def pair_scorer_dropout(self):
        """Alias for ``model.pair_scorer_dropout``."""
        return self.model.pair_scorer_dropout

    @property
    def bond_proj_mode(self):
        """Alias for ``model.bond_proj_mode``."""
        return self.model.bond_proj_mode

    @property
    def use_layer_scale(self):
        """Alias for ``model.use_layer_scale``."""
        return self.model.use_layer_scale

    @property
    def gat_drop_path(self):
        """Alias for ``model.gat_drop_path``."""
        return self.model.gat_drop_path

    @property
    def step_embed_bond_proj(self):
        """Alias for ``model.step_embed_bond_proj``."""
        return self.model.step_embed_bond_proj

    @property
    def attention_score_scale(self):
        """Alias for ``model.attention_score_scale``."""
        return self.model.attention_score_scale

    @property
    def pair_score_backend(self):
        """Alias for ``model.pair_score_backend``."""
        return self.model.pair_score_backend

    @property
    def pair_score_chunk_size(self):
        """Alias for ``model.pair_score_chunk_size``."""
        return self.model.pair_score_chunk_size

    @property
    def drop_head_bond_tokens_after_gnn(self):
        """Alias for ``model.drop_head_bond_tokens_after_gnn``."""
        return self.model.drop_head_bond_tokens_after_gnn

    @property
    def skip_empty_history_cross_attn_for_history_encoder(self):
        """Alias for ``model.skip_empty_history_cross_attn_for_history_encoder``."""
        return self.model.skip_empty_history_cross_attn_for_history_encoder

    @property
    def trajectory_attention(self):
        """Alias for ``model.trajectory_attention``."""
        return self.model.trajectory_attention

    @property
    def residual_connection_type(self):
        """Alias for ``model.residual_connection_type``."""
        return self.model.residual_connection_type

    @property
    def mhc_num_streams(self):
        """Alias for ``model.mhc_num_streams``."""
        return self.model.mhc_num_streams

    @property
    def attnres_block_size(self):
        """Alias for ``model.attnres_block_size``."""
        return self.model.attnres_block_size

    @property
    def use_moe(self):
        """Alias for ``model.use_moe``."""
        return self.model.use_moe

    @property
    def moe_num_experts(self):
        """Alias for ``model.moe_num_experts``."""
        return self.model.moe_num_experts

    @property
    def moe_top_k(self):
        """Alias for ``model.moe_top_k``."""
        return self.model.moe_top_k

    @property
    def moe_loss_coef(self):
        """Alias for ``model.moe_loss_coef``."""
        return self.model.moe_loss_coef

    @property
    def moe_expert_dim(self):
        """Alias for ``model.moe_expert_dim``."""
        return self.model.moe_expert_dim

    @property
    def skip_gat(self):
        """Alias for ``model.skip_gat``."""
        return self.model.skip_gat

    @property
    def graph_attention_bias(self):
        """Alias for ``model.graph_attention_bias``."""
        return self.model.graph_attention_bias

    @property
    def graph_attention_bias_hops(self):
        """Alias for ``model.graph_attention_bias_hops``."""
        return self.model.graph_attention_bias_hops
    @property
    def lora_enabled(self):
        """Alias for ``lora.enabled``."""
        return self.lora.enabled

    @property
    def lora_rank(self):
        """Alias for ``lora.rank``."""
        return self.lora.rank

    @property
    def lora_alpha(self):
        """Alias for ``lora.alpha``."""
        return self.lora.alpha

    @property
    def lora_dropout(self):
        """Alias for ``lora.dropout``."""
        return self.lora.dropout

    @property
    def lora_freeze_base_model(self):
        """Alias for ``lora.freeze_base_model``."""
        return self.lora.freeze_base_model

    @property
    def lora_target_modules(self):
        """Alias for ``lora.target_modules``."""
        return self.lora.target_modules
    @property
    def ranking_loss_type(self):
        """Alias for ``loss.ranking_loss_type``."""
        return self.loss.ranking_loss_type

    @property
    def temperature(self):
        """Alias for ``loss.temperature``."""
        return self.loss.temperature

    @property
    def lambda_anchor(self):
        """Alias for ``loss.lambda_anchor``."""
        return self.loss.lambda_anchor

    @property
    def train_no_reaction_class(self):
        """Alias for ``loss.train_no_reaction_class``."""
        return self.loss.train_no_reaction_class

    @property
    def negative_reaction_weight(self):
        """Alias for ``loss.negative_reaction_weight``."""
        return self.loss.negative_reaction_weight

    @property
    def beta(self):
        """Alias for ``loss.beta``."""
        return self.loss.beta

    @property
    def bond_type_weight(self):
        """Alias for ``loss.bond_type_weight``."""
        return self.loss.bond_type_weight

    @property
    def binary_weight(self):
        """Alias for ``loss.binary_weight``."""
        return self.loss.binary_weight

    @property
    def hard_negative_k(self):
        """Alias for ``loss.hard_negative_k``."""
        return self.loss.hard_negative_k

    @property
    def virtual_pos_weight(self):
        """Alias for ``loss.virtual_pos_weight``."""
        return self.loss.virtual_pos_weight

    @property
    def label_smoothing(self):
        """Alias for ``loss.label_smoothing``."""
        return self.loss.label_smoothing

    @property
    def pair_scoring_mode(self):
        """Alias for ``loss.pair_scoring_mode``."""
        return self.loss.pair_scoring_mode

    @property
    def step_count_aux_weight(self):
        """Alias for ``loss.step_count_aux_weight``."""
        return self.loss.step_count_aux_weight

    @property
    def pair_score_target(self):
        """Alias for ``loss.pair_score_target``."""
        return self.loss.pair_score_target

    @property
    def multistep_loss_scale(self):
        """Alias for ``loss.multistep_loss_scale``."""
        return self.loss.multistep_loss_scale

    @property
    def bond_contrastive_weight(self):
        """Alias for ``loss.bond_contrastive_weight``."""
        return self.loss.bond_contrastive_weight

    @property
    def virtual_focal_gamma(self):
        """Alias for ``loss.virtual_focal_gamma``."""
        return self.loss.virtual_focal_gamma

    @property
    def binary_focal_gamma(self):
        """Alias for ``loss.binary_focal_gamma``."""
        return self.loss.binary_focal_gamma

    @property
    def virtual_consistency_weight(self):
        """Alias for ``loss.virtual_consistency_weight``."""
        return self.loss.virtual_consistency_weight

    @property
    def binary_virtual_consistency_weight(self):
        """Alias for ``loss.binary_virtual_consistency_weight``."""
        return self.loss.binary_virtual_consistency_weight

    @property
    def virtual_margin(self):
        """Alias for ``loss.virtual_margin``."""
        return self.loss.virtual_margin

    @property
    def stop_rank_aux_weight(self):
        """Alias for ``loss.stop_rank_aux_weight``."""
        return self.loss.stop_rank_aux_weight
    @property
    def learning_rate(self):
        """Alias for ``train.learning_rate``."""
        return self.train.learning_rate

    @property
    def weight_decay(self):
        """Alias for ``train.weight_decay``."""
        return self.train.weight_decay

    @property
    def batch_size(self):
        """Alias for ``train.batch_size``."""
        return self.train.batch_size

    @property
    def epochs(self):
        """Alias for ``train.epochs``."""
        return self.train.epochs

    @property
    def gradient_clip(self):
        """Alias for ``train.gradient_clip``."""
        return self.train.gradient_clip

    @property
    def gradient_accumulation_steps(self):
        """Alias for ``train.gradient_accumulation_steps``."""
        return self.train.gradient_accumulation_steps

    @property
    def use_scheduler(self):
        """Alias for ``train.use_scheduler``."""
        return self.train.use_scheduler

    @property
    def scheduler_type(self):
        """Alias for ``train.scheduler_type``."""
        return self.train.scheduler_type

    @property
    def resume_scheduler_mode(self):
        """Alias for ``train.resume_scheduler_mode``."""
        return self.train.resume_scheduler_mode

    @property
    def warmup_steps(self):
        """Alias for ``train.warmup_steps``."""
        return self.train.warmup_steps

    @property
    def warmup_batches(self):
        """Alias for ``train.warmup_batches``."""
        return self.train.warmup_batches

    @property
    def eta_min(self):
        """Alias for ``train.eta_min``."""
        return self.train.eta_min

    @property
    def virtual_lr_scale(self):
        """Alias for ``train.virtual_lr_scale``."""
        return self.train.virtual_lr_scale

    @property
    def moe_freeze_router(self):
        """Alias for ``train.moe_freeze_router``."""
        return self.train.moe_freeze_router

    @property
    def moe_router_lr_scale(self):
        """Alias for ``train.moe_router_lr_scale``."""
        return self.train.moe_router_lr_scale

    @property
    def moe_expert_dropout(self):
        """Alias for ``model.moe_expert_dropout``."""
        return self.model.moe_expert_dropout

    @property
    def dtype(self):
        """Alias for ``train.dtype``."""
        return self.train.dtype

    @property
    def num_gpus(self):
        """Alias for ``train.num_gpus``."""
        return self.train.num_gpus

    @property
    def num_dataloader_workers(self):
        """Alias for ``train.num_dataloader_workers``."""
        return self.train.num_dataloader_workers

    @property
    def pin_memory(self):
        """Alias for ``train.pin_memory``."""
        return self.train.pin_memory

    @property
    def persistent_workers(self):
        """Alias for ``train.persistent_workers``."""
        return self.train.persistent_workers

    @property
    def prefetch_factor(self):
        """Alias for ``train.prefetch_factor``."""
        return self.train.prefetch_factor

    @property
    def torch_compile(self):
        """Alias for ``train.torch_compile``."""
        return self.train.torch_compile

    @property
    def ema_decay(self):
        """Alias for ``train.ema_decay``."""
        return self.train.ema_decay

    @property
    def optimizer_type(self):
        """Alias for ``train.optimizer_type``."""
        return self.train.optimizer_type

    @property
    def muon_momentum(self):
        """Alias for ``train.muon_momentum``."""
        return self.train.muon_momentum

    @property
    def muon_ns_steps(self):
        """Alias for ``train.muon_ns_steps``."""
        return self.train.muon_ns_steps

    @property
    def muon_update_clip(self):
        """Alias for ``train.muon_update_clip``."""
        return self.train.muon_update_clip

    @property
    def muon_all_params(self):
        """Alias for ``train.muon_all_params``."""
        return self.train.muon_all_params

    @property
    def save_every_epoch(self):
        """Alias for ``train.save_every_epoch``."""
        return self.train.save_every_epoch

    @property
    def drop_product_history_prob(self):
        """Alias for ``train.drop_product_history_prob``."""
        return self.train.drop_product_history_prob
    @property
    def max_eval_steps(self):
        """Alias for ``eval.max_eval_steps``."""
        return self.eval.max_eval_steps

    @property
    def eval_beam_width(self):
        """Alias for ``eval.eval_beam_width``."""
        return self.eval.eval_beam_width

    @property
    def pair_selection_mode(self):
        """Alias for ``eval.pair_selection_mode``."""
        return self.eval.pair_selection_mode

    @property
    def pair_selection_temperature(self):
        """Alias for ``eval.pair_selection_temperature``."""
        return self.eval.pair_selection_temperature

    @property
    def pair_softmax_topk(self):
        """Alias for ``eval.pair_softmax_topk``."""
        return self.eval.pair_softmax_topk

    @property
    def eval_every_n_batches(self):
        """Alias for ``eval.eval_every_n_batches``."""
        return self.eval.eval_every_n_batches


    @property
    def train_path(self):
        """Alias for ``data.train_path``."""
        return self.data.train_path

    @property
    def val_path(self):
        """Alias for ``data.val_path``."""
        return self.data.val_path

    @property
    def test_path(self):
        """Alias for ``data.test_path``."""
        return self.data.test_path

    @property
    def data_version(self):
        """Alias for ``data.data_version``."""
        return self.data.data_version

    @property
    def train_limit(self):
        """Alias for ``data.train_limit``."""
        return self.data.train_limit

    @property
    def val_limit(self):
        """Alias for ``data.val_limit``."""
        return self.data.val_limit

    @property
    def test_limit(self):
        """Alias for ``data.test_limit``."""
        return self.data.test_limit

    @property
    def force_recalculate_cache(self):
        """Alias for ``data.force_recalculate_cache``."""
        return self.data.force_recalculate_cache

    @property
    def shard_size(self):
        """Alias for ``data.shard_size``."""
        return self.data.shard_size

    @property
    def include_product(self):
        """Alias for ``data.include_product``."""
        return self.data.include_product

    @property
    def photo_oversampling_factor(self):
        """Alias for ``data.photo_oversampling_factor``."""
        return self.data.photo_oversampling_factor

    @property
    def multistep_oversampling_factor(self):
        """Alias for ``data.multistep_oversampling_factor``."""
        return self.data.multistep_oversampling_factor

    @property
    def late_step_oversampling_factor(self):
        """Alias for ``data.late_step_oversampling_factor``."""
        return self.data.late_step_oversampling_factor

    @property
    def product_sampling_factor(self):
        """Alias for ``data.product_sampling_factor``."""
        return self.data.product_sampling_factor

    @property
    def enable_blockspace_product(self):
        """Alias for ``data.enable_blockspace_product``."""
        return self.data.enable_blockspace_product




    @property
    def history_grad(self):
        """Alias for ``data.history_grad``."""
        return self.data.history_grad

    @property
    def sample_mode(self):
        """Alias for ``data.sample_mode``."""
        return self.data.sample_mode

    @property
    def history_noise_std(self):
        """Alias for ``data.history_noise_std``."""
        return self.data.history_noise_std

    @property
    def history_noise_ramp_batches(self):
        """Alias for ``data.history_noise_ramp_batches``."""
        return self.data.history_noise_ramp_batches

    class Config:
        """Pydantic settings; unknown fields are ignored during validation."""

        extra = "ignore"


ConfigType = RankingConfig


def load_config(yaml_path: str) -> RankingConfig:
    """Load a RankingConfig from a YAML file.

    Supports nested YAML files and flat YAML files without a top-level
    ``model`` section.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Validated RankingConfig instance.
    """
    with open(yaml_path, "r") as f:
        raw_config = yaml.safe_load(f)
    return RankingConfig.model_validate(raw_config)


def save_config(config: RankingConfig, yaml_path: str):
    """Save a RankingConfig to a YAML file.

    Args:
        config: Configuration to serialize.
        yaml_path: Destination file path.
    """
    with open(yaml_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
