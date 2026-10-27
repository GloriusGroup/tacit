"""Trainer for the ranking-based bond-pair prediction model."""

import os
import random
import subprocess
from contextlib import nullcontext
from functools import partial
import mlflow
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
from bond_match.config import RankingConfig, ResumeSchedulerMode
from bond_match.dataset import (
    ShardedReactionDataset,
    ShardAwareSampler,
    collate_fn,
    trajectory_collate_fn,
)
from bond_match.sampler import WeightedDistributedSampler
from bond_match.modules.lora import (
    checkpoint_has_lora_weights,
    collect_lora_stats,
    freeze_non_lora_parameters,
)
from bond_match.modules.muon_clip import MuonClip
from bond_match.model import BondGAT
from bond_match.model.heads import (
    AtomSwapTypeHead,
    BondSplitSwapHead,
    MLPSwapTypeHead,
    NodeAttentionSwapHead,
)

from bond_match.training.ddp import (
    is_main_process,
)
from bond_match.training.ema import (
    ModelEMA,
)
from bond_match.training.eval_collector import (
    EvalCollector,
)
from bond_match.training.loss import (
    build_loss,
)
from bond_match.training.mlflow_utils import (
    _safe_mlflow_log,
)
from bond_match.training.optimizer import (
    _DualOptimizer,
)
from bond_match.training.state import (
    BatchInputs,
    EvalMetrics,
    Precision,
)

class Trainer:
    """Ranking-based bond prediction trainer.

    Supports single-GPU, multi-GPU (DDP), and mixed-precision training.
    """

    def __init__(self, config: RankingConfig):
        """Build the model, loss, optimizer, scheduler, EMA, and dataloaders.

        Loss-function parameters (e.g. learnable temperature) are included in
        the optimizer. In DDP, only rank 0 builds the dataset cache while the
        other ranks wait at a barrier, avoiding races on shard files and
        redundant work; rank 0 then waits at a second barrier so all ranks
        load the finished cache.

        Args:
            config: Validated training configuration.
        """
        self.config = config
        self.use_ddp = config.num_gpus > 1 and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.use_ddp else 1
        self.device = self._resolve_device()
        self.precision = Precision(config.dtype)
        self.use_amp = self.precision in (Precision.BFLOAT16, Precision.FLOAT16)
        self.amp_dtype = (
            torch.bfloat16 if self.precision == Precision.BFLOAT16 else torch.float16
        )

        self._deferred_checkpoint = None
        self.start_epoch = 0
        self._resume_batch_idx = -1
        self._last_batch_idx = -1
        self._global_batch_idx = 0

        self.model = self._build_model()
        self.raw_model = self.model.module if self.use_ddp else self.model
        self.loss_fn = build_loss(config).to(self.device)
        if config.label_smoothing > 0:
            self.loss_fn.bond_type_ce = nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing
            )
        self.optimizer = self._build_optimizer(config)
        self.scheduler = self._build_scheduler(config.epochs)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=(self.precision == Precision.FLOAT16))
            if self.use_amp
            else None
        )

        self._restore_training_state()

        self.ema = None
        if config.ema_decay > 0:
            self.ema = ModelEMA(self.raw_model, decay=config.ema_decay)
            if hasattr(self, "_ema_checkpoint") and self._ema_checkpoint is not None:
                self.ema.load_state_dict(self._ema_checkpoint)
                if is_main_process():
                    print(
                        f"EMA enabled (decay={config.ema_decay}, restored from checkpoint)"
                    )
                self._ema_checkpoint = None
            elif is_main_process():
                print(f"EMA enabled (decay={config.ema_decay})")

        if self.use_ddp and not is_main_process():
            dist.barrier()
        self.train_loader, self.val_loader = self._build_dataloaders()
        if self.use_ddp and is_main_process():
            dist.barrier()

        self._mlflow_enabled = not config.disable_mlflow

        if is_main_process():
            self._print_config_summary()
            if self._mlflow_enabled:
                self._setup_mlflow()

    def _resolve_device(self) -> torch.device:
        """Return the CUDA device for this rank, or CPU if CUDA is unavailable."""
        if self.use_ddp:
            return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> nn.Module:
        """Construct the BondGAT model with optional checkpoint loading, LoRA, compilation, and DDP wrapping.

        Returns:
            nn.Module: The model, wrapped in DistributedDataParallel when DDP
            is active.
        """
        model = BondGAT.from_config(self.config).to(self.device)

        if self.config.load_from_checkpoint:
            self._load_checkpoint(model)

        if self.config.lora_enabled:
            if self.config.lora_freeze_base_model:
                freeze_non_lora_parameters(model)
            if is_main_process():
                stats = collect_lora_stats(model)
                if stats.replaced_modules == 0:
                    raise RuntimeError(
                        "LoRA was enabled but no target linear modules were wrapped. "
                        "Check lora.target_modules in the config."
                    )
                pct = 100.0 * stats.trainable_params / max(stats.total_params, 1)
                print(
                    "LoRA enabled: "
                    f"{stats.replaced_modules} linear modules, "
                    f"{stats.trainable_params:,}/{stats.total_params:,} trainable params "
                    f"({pct:.2f}%)"
                    )

        self._freeze_unused_config_modules(model)

        if self.config.torch_compile:
            if is_main_process():
                print("Compiling model with torch.compile(dynamic=True)...")
            model = torch.compile(model, dynamic=True)

        if self.use_ddp:
            local_rank = int(os.environ["LOCAL_RANK"])
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        return model

    def _freeze_unused_config_modules(self, model: nn.Module) -> None:
        """Disable gradients for module parameters unused under the current config."""
        cfg = self.config
        frozen_prefixes = []
        if cfg.pair_scoring_mode != "pair_absorbs_binary":
            frozen_prefixes.append("pair_scorer_with_reactivity")
        if cfg.pair_scoring_mode != "learned_fusion":
            frozen_prefixes.append("fusion_weights")
        if cfg.step_count_aux_weight <= 0:
            frozen_prefixes.append("step_count_head")
        if cfg.use_pair_bond_type:
            frozen_prefixes.append("bond_type_classifier")
        if cfg.use_attention_pool_virtual:
            frozen_prefixes.extend(
                [
                    "virtual_output_head",
                    "virtual_output_head_multi",
                    "step_index_emb",
                    "virtual_gate",
                    "pair_score_proj",
                ]
            )
        if cfg.residual_connection_type == "block_attnres":
            frozen_prefixes.extend(
                [
                    "bond_transformer_blocks.0.ar_sa.query",
                    "bond_transformer_blocks.0.ar_sa.norm",
                ]
            )

        n_frozen = 0
        for name, param in model.named_parameters():
            if any(name == p or name.startswith(f"{p}.") for p in frozen_prefixes):
                param.requires_grad = False
                n_frozen += 1

        if n_frozen and is_main_process():
            print(f"Froze {n_frozen} unused parameter tensors: {', '.join(frozen_prefixes)}")

    # Parameters in the fast LR group when ``virtual_lr_scale > 1``.
    _FAST_PARAM_KEYWORDS = (
        "virtual",
        "bond_type_classifier",
        "pair_bond_type_classifier",
        "step_index_emb",
        "step_idx_node_emb",
        "cross_attn",
        "norm_cross",
        "step_count_head",
        "history_step_pos_emb",
    )

    # Top-level module names optimized by AdamW (instead of MuonClip) in the
    # split muon_clip configuration.
    _GAT_MODULE_NAMES = frozenset(
        {
            "layers",
            "gat_norms",
            "atom_emb",
            "charge_emb",
            "h_count_emb",
            "aromatic_emb",
            "symmetry_emb",
            "is_virtual_emb",
            "step_idx_node_emb",
            "edge_emb",
            "node_proj",
        }
    )

    def _build_optimizer(self, config: RankingConfig) -> optim.Optimizer:
        """Build the optimizer selected by ``config.optimizer_type``.

        Parameter groups are assembled in stages: MoE router parameters
        (``to_gates``) are either frozen or given a scaled learning rate;
        when ``virtual_lr_scale > 1`` the remaining parameters are split into
        a fast group (virtual/bond-type/history heads) and a slow group; loss
        function parameters form their own group. For ``"muon_clip"``,
        either MuonClip optimizes all parameters (``muon_all_params``, with
        update clipping), or the parameters are split between MuonClip
        (transformer and heads) and AdamW (GAT and embeddings) combined via
        ``_DualOptimizer``. Otherwise AdamW is used for all groups.

        Args:
            config: Training configuration.

        Returns:
            optim.Optimizer: The configured optimizer.
        """
        router_params = self._collect_router_params(config)
        slow_params, fast_params, scaled_lr = self._split_fast_slow_params(config)
        loss_params = list(self.loss_fn.parameters())

        param_groups = [{"params": slow_params, "lr": config.learning_rate}]
        if fast_params:
            param_groups.append({"params": fast_params, "lr": scaled_lr})
        if router_params:
            router_lr = config.learning_rate * config.moe_router_lr_scale
            param_groups.append({"params": router_params, "lr": router_lr})
        if loss_params:
            param_groups.append({"params": loss_params, "lr": config.learning_rate})

        if config.optimizer_type == "muon_clip":
            return self._build_muon_optimizer(
                config, param_groups, slow_params, fast_params, loss_params
            )

        return optim.AdamW(
            param_groups, lr=config.learning_rate, weight_decay=config.weight_decay, fused=True
        )

    def _collect_router_params(self, config: RankingConfig) -> list[nn.Parameter]:
        """Collect MoE router (``to_gates``) parameters, freezing them when configured.

        Returns:
            Router parameters to optimize; empty when the router is frozen.
        """
        router_params: list[nn.Parameter] = []
        n_frozen_router = 0
        for name, param in self.raw_model.named_parameters():
            if "to_gates" in name:
                if config.moe_freeze_router:
                    param.requires_grad = False
                    n_frozen_router += 1
                else:
                    router_params.append(param)

        if is_main_process():
            if config.moe_freeze_router:
                print(f"MoE router FROZEN ({n_frozen_router} tensors)")
            elif config.moe_router_lr_scale != 1.0:
                print(
                    f"MoE router LR scale: {config.moe_router_lr_scale} ({len(router_params)} tensors)"
                )
        return router_params

    def _split_fast_slow_params(
        self, config: RankingConfig
    ) -> tuple[list[nn.Parameter], list[nn.Parameter], float]:
        """Split trainable non-router parameters into slow and fast LR groups.

        With ``virtual_lr_scale > 1``, parameters whose names match
        ``_FAST_PARAM_KEYWORDS`` train at the scaled learning rate; otherwise
        all parameters land in the slow group.

        Returns:
            Tuple of (slow params, fast params, fast learning rate).
        """
        if config.virtual_lr_scale <= 1.0:
            slow_params = [
                p
                for n, p in self.raw_model.named_parameters()
                if p.requires_grad and "to_gates" not in n
            ]
            return slow_params, [], config.learning_rate

        scaled_lr = config.learning_rate * config.virtual_lr_scale
        fast_params: list[nn.Parameter] = []
        slow_params: list[nn.Parameter] = []
        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if "to_gates" in name:
                continue
            if any(k in name for k in self._FAST_PARAM_KEYWORDS):
                fast_params.append(param)
            else:
                slow_params.append(param)

        if is_main_process():
            print(
                f"Differential LR: {len(slow_params)} params @ {config.learning_rate}, "
                f"{len(fast_params)} params @ {scaled_lr}"
            )
        return slow_params, fast_params, scaled_lr

    def _build_muon_optimizer(
        self,
        config: RankingConfig,
        param_groups: list[dict],
        slow_params: list[nn.Parameter],
        fast_params: list[nn.Parameter],
        loss_params: list[nn.Parameter],
    ) -> optim.Optimizer:
        """Build the MuonClip optimizer (all-params or split-with-AdamW variant).

        Returns:
            A single MuonClip over all parameter groups when
            ``muon_all_params`` is set, otherwise a ``_DualOptimizer``
            combining MuonClip (transformer and heads) with AdamW (GAT,
            embeddings, and loss parameters).
        """
        if config.muon_all_params:
            all_params = slow_params + fast_params + loss_params
            if is_main_process():
                n2d = sum(1 for p in all_params if p.ndim >= 2)
                n1d = sum(1 for p in all_params if p.ndim < 2)
                print(
                    f"MuonClip (all): {n2d} Muon (2D+), {n1d} momentum (1D), "
                    f"update_clip={config.muon_update_clip}"
                )
            muon_opt = MuonClip(
                param_groups,
                lr=config.learning_rate,
                momentum=config.muon_momentum,
                weight_decay=config.weight_decay,
                ns_steps=config.muon_ns_steps,
                update_clip=config.muon_update_clip,
            )
            muon_opt.set_model(self.raw_model)
            return muon_opt

        muon_params: list[nn.Parameter] = []
        adamw_params: list[nn.Parameter] = []
        for name, param in self.raw_model.named_parameters():
            top_name = name.split(".")[0]
            if top_name in self._GAT_MODULE_NAMES:
                adamw_params.append(param)
            else:
                muon_params.append(param)
        adamw_params.extend(loss_params)
        if is_main_process():
            print(
                f"MuonClip (split): {len(muon_params)} Muon (transformer+heads), "
                f"{len(adamw_params)} AdamW (GAT+emb+loss), "
                f"update_clip={config.muon_update_clip}"
            )
        muon_opt = MuonClip(
            muon_params,
            lr=config.learning_rate,
            momentum=config.muon_momentum,
            weight_decay=config.weight_decay,
            ns_steps=config.muon_ns_steps,
            update_clip=config.muon_update_clip,
        )
        muon_opt.set_model(self.raw_model)
        return _DualOptimizer(
            muon_opt,
            optim.AdamW(
                adamw_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                fused=True
            ),
        )

    def _build_scheduler(self, total_epochs: int, include_warmup: bool = True):
        """Build a cosine-annealing scheduler with optional linear warmup.

        ``config.warmup_steps`` denotes warmup epochs; the scheduler is
        stepped once per epoch.

        Args:
            total_epochs: Cosine annealing horizon (``T_max``), clamped to
                at least 1.
            include_warmup: Whether to prepend the linear warmup phase.

        Returns:
            A ``CosineAnnealingLR``, or a ``SequentialLR`` of linear warmup
            followed by cosine annealing.
        """
        cfg = self.config
        total_epochs = max(1, total_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=cfg.eta_min,
        )
        if include_warmup and cfg.warmup_steps > 0:
            warmup = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=cfg.warmup_steps,
            )
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[cfg.warmup_steps],
            )
        return cosine

    def _restart_scheduler_for_remaining_epochs(self) -> int:
        """Rebuild the scheduler to anneal over the epochs left after resuming.

        Current per-group learning rates are preserved as the new base rates,
        and no warmup phase is included.

        Returns:
            int: Number of remaining epochs used as the new annealing horizon.
        """
        remaining_epochs = max(1, self.config.epochs - self.start_epoch)
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.scheduler = self._build_scheduler(remaining_epochs, include_warmup=False)
        for group, lr in zip(self.optimizer.param_groups, current_lrs):
            group["lr"] = lr
            group["initial_lr"] = lr
        if hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = current_lrs
        return remaining_epochs

    def _resolve_checkpoint_path(self) -> str:
        """Return the checkpoint path to load: an explicit ``checkpoint_path`` or the experiment-named default."""
        cfg = self.config
        if cfg.checkpoint_path:
            print(
                "Loading from explicitly specified checkpoint path, ignoring experiment_name:",
                cfg.checkpoint_path,
            )
            return cfg.checkpoint_path
        checkpoint_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "checkpoints",
            "train",
        )
        save_name = (
            f"{cfg.experiment_name}.pt" if cfg.experiment_name else "bondnn_ranking.pt"
        )
        return os.path.join(checkpoint_dir, save_name)

    def _load_checkpoint(self, model: nn.Module) -> None:
        """Load model weights (and loss-function state) from a checkpoint.

        Accepts both raw state dict checkpoints and checkpoint dicts with
        ``model_state_dict``. When config options change
        the parameter layout (canonical rank, non-MLP swap head, MoE expert
        dropout, LoRA), a partial load is performed: keys with missing
        counterparts or mismatched shapes are skipped, and expert output
        keys are remapped from ``.net.2.`` to ``.net.3.`` when expert
        dropout shifts module indices. The full checkpoint dict is stashed
        in ``_deferred_checkpoint`` because the optimizer does not yet exist
        when this is called from ``_build_model``; training state is
        restored later by ``_restore_training_state``.

        Args:
            model: Model to load weights into.

        Raises:
            RuntimeError: If the checkpoint contains LoRA weights but LoRA
                is disabled in the config.
        """
        cfg = self.config
        ckpt_path = self._resolve_checkpoint_path()
        if is_main_process():
            print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            if "loss_fn_state_dict" in checkpoint and list(self.loss_fn.parameters()):
                self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
        else:
            model_state = checkpoint

        if checkpoint_has_lora_weights(model_state) and not cfg.lora_enabled:
            raise RuntimeError(
                "Checkpoint contains LoRA weights, but config has lora.enabled=false. "
                "Use the matching LoRA config to load this checkpoint."
            )

        allow_partial_load = any(
            [
                cfg.use_canonical_rank,
                cfg.swap_type_head != "mlp",
                cfg.moe_expert_dropout > 0,
                cfg.lora_enabled,
            ]
        )

        if allow_partial_load:
            current_state = model.state_dict()
            compatible_state = {}
            skipped_keys = []

            remapped_state = dict(model_state)
            if cfg.moe_expert_dropout > 0:
                remap_additions = {}
                for key, value in model_state.items():
                    if "experts.experts" in key and ".net.2." in key:
                        new_key = key.replace(".net.2.", ".net.3.")
                        if new_key in current_state and key not in current_state:
                            if current_state[new_key].shape == value.shape:
                                remap_additions[new_key] = value
                remapped_state.update(remap_additions)
                if remap_additions and is_main_process():
                    print(
                        f"  remapped {len(remap_additions)} expert output keys (net.2 -> net.3 for dropout)"
                    )

            for key, value in remapped_state.items():
                if key not in current_state:
                    skipped_keys.append(f"{key} (missing)")
                    continue
                if current_state[key].shape != value.shape:
                    skipped_keys.append(
                        f"{key} ({tuple(value.shape)} -> {tuple(current_state[key].shape)})"
                    )
                    continue
                compatible_state[key] = value

            model.load_state_dict(compatible_state, strict=False)

            if is_main_process():
                missing = sorted(
                    set(current_state.keys()) - set(compatible_state.keys())
                )
                if missing:
                    print(
                        f"  partial checkpoint load: {len(missing)} model keys initialized fresh; "
                        f"examples: {missing[:5]}"
                    )
                if skipped_keys:
                    print(
                        f"  skipped incompatible checkpoint keys: {len(skipped_keys)}; "
                        f"examples: {skipped_keys[:5]}"
                    )
        else:
            model.load_state_dict(model_state, strict=True)

        self._deferred_checkpoint = checkpoint

    def _remap_optimizer_state(self, old_osd: dict) -> None:
        """Load per-parameter optimizer state from a checkpoint with a different group layout.

        The old checkpoint stored all parameters in one group in
        ``named_parameters()`` order, whereas the current optimizer may split
        them across multiple groups (e.g. separate router parameters).
        Parameters are matched by identity (``data_ptr``) to their old state
        keys, and a state dict matching the current group layout is rebuilt
        and loaded.

        Args:
            old_osd: Optimizer state dict from the checkpoint.
        """
        old_flat_keys: list[int] = []
        for g in old_osd["param_groups"]:
            old_flat_keys.extend(g["params"])

        named_params = [
            p for _, p in self.raw_model.named_parameters() if p.requires_grad
        ]

        ptr_to_old_key: dict[int, int] = {}
        for i, param in enumerate(named_params):
            if i < len(old_flat_keys):
                ptr_to_old_key[param.data_ptr()] = old_flat_keys[i]

        new_state: dict[int, dict] = {}
        new_groups: list[dict] = []
        new_idx = 0
        n_restored = 0
        for group in self.optimizer.param_groups:
            group_indices: list[int] = []
            for param in group["params"]:
                old_key = ptr_to_old_key.get(param.data_ptr())
                if old_key is not None and old_key in old_osd["state"]:
                    new_state[new_idx] = old_osd["state"][old_key]
                    n_restored += 1
                group_indices.append(new_idx)
                new_idx += 1
            pg = {k: v for k, v in group.items() if k != "params"}
            pg["params"] = group_indices
            new_groups.append(pg)

        self.optimizer.load_state_dict({"state": new_state, "param_groups": new_groups})
        if is_main_process():
            n_total = sum(len(g["params"]) for g in self.optimizer.param_groups)
            print(
                f"  Remapped optimizer state ({len(old_osd['param_groups'])} -> "
                f"{len(self.optimizer.param_groups)} groups): "
                f"{n_restored}/{n_total} param states restored"
            )

    def _restore_training_state(self) -> None:
        """Restore optimizer, scheduler, scaler, and epoch/batch position from the deferred checkpoint.

        When ``resume_training_state`` is false, only the EMA state is kept
        for later restore. Optimizer state with a mismatched group count is
        remapped via ``_remap_optimizer_state``. A checkpoint with a stored
        ``batch_idx`` (or an explicit ``resume_from_batch``) resumes
        mid-epoch; otherwise training resumes at the following epoch. With
        ``resume_scheduler_mode == REMAINING``, the scheduler is rebuilt over
        the remaining epochs. EMA restore is deferred to ``__init__`` because
        the EMA object is created after this method runs.
        """
        checkpoint = self._deferred_checkpoint
        if checkpoint is None or not isinstance(checkpoint, dict):
            return

        if not self.config.resume_training_state:
            if is_main_process():
                print(
                    "  Loaded model weights only; using fresh optimizer/scheduler (resume_training_state=false)"
                )
            self._ema_checkpoint = (
                checkpoint.get("ema_state_dict")
                if isinstance(checkpoint, dict)
                else None
            )
            self._deferred_checkpoint = None
            return

        restored = []
        scheduler_restored = False
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as e:
                if "different number of parameter groups" in str(e):
                    self._remap_optimizer_state(checkpoint["optimizer_state_dict"])
                else:
                    raise
            restored.append("optimizer")
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            restored.append("scheduler")
            scheduler_restored = True
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            restored.append("scaler")
        if "epoch" in checkpoint and checkpoint["epoch"] >= 0:
            batch_idx = checkpoint.get("batch_idx", -1)
            if self.config.resume_from_batch >= 0:
                batch_idx = self.config.resume_from_batch
            if batch_idx >= 0:
                self.start_epoch = checkpoint["epoch"]
                self._resume_batch_idx = batch_idx
                restored.append(
                    f"epoch={self.start_epoch}, batch={self._resume_batch_idx}"
                )
            else:
                self.start_epoch = checkpoint["epoch"] + 1
                restored.append(f"epoch={self.start_epoch}")

        if (
            scheduler_restored
            and self.config.resume_scheduler_mode == ResumeSchedulerMode.REMAINING
            and 0 < self.start_epoch < self.config.epochs
        ):
            remaining_epochs = self._restart_scheduler_for_remaining_epochs()
            restored.append(f"scheduler_remaining_epochs={remaining_epochs}")

        if restored and is_main_process():
            print(f"  Restored training state: {', '.join(restored)}")

        self._ema_checkpoint = checkpoint.get("ema_state_dict")
        self._deferred_checkpoint = None

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation dataloaders over the sharded reaction datasets.

        Selects the collate function by sample mode (step vs. trajectory) and
        a weighted sampler when photo or multistep oversampling is enabled.
        Under DDP a (weighted) distributed sampler shards the training set,
        while the validation loader covers the full dataset because only rank 0
        evaluates. Dataloader workers are seeded from the torch initial seed.

        Returns:
            tuple[DataLoader, DataLoader]: Train and validation loaders.
        """
        cfg = self.config
        if not cfg.include_product and is_main_process():
            print(
                "include_product=false: training will not include explicit product stop "
                "targets; stop is learned indirectly from stop_rank_aux/negative samples."
            )
        force_recalculate_cache = cfg.force_recalculate_cache
        if self.use_ddp and not is_main_process():
            force_recalculate_cache = False
        dv = cfg.data_version
        if hasattr(dv, "value"):
            dv = dv.value
        sample_mode = getattr(cfg, "sample_mode", "step")
        use_trajectory_mode = sample_mode == "trajectory"
        enable_history = cfg.enable_history_context and not use_trajectory_mode

        train_dataset = ShardedReactionDataset(
            cfg.train_path,
            include_product=cfg.include_product,
            limit=cfg.train_limit,
            force_recalculate_cache=force_recalculate_cache,
            extended_features=cfg.extended_atom_features,
            data_version=dv,
            use_all_mechanisms=False,
            enable_history=enable_history,
            shard_size=cfg.shard_size,
            canonical_rank=cfg.use_canonical_rank,
            enable_blockspace_product=cfg.enable_blockspace_product,
            sample_mode=sample_mode,
        )
        val_dataset = ShardedReactionDataset(
            cfg.val_path,
            include_product=cfg.include_product,
            limit=cfg.val_limit,
            force_recalculate_cache=force_recalculate_cache,
            extended_features=cfg.extended_atom_features,
            data_version=dv,
            enable_history=enable_history,
            shard_size=cfg.shard_size,
            canonical_rank=cfg.use_canonical_rank,
            enable_blockspace_product=cfg.enable_blockspace_product,
            sample_mode=sample_mode,
        )

        if cfg.include_product and cfg.product_sampling_factor < 1.0:
            train_dataset.subsample_products(cfg.product_sampling_factor)

        collate_kwargs = dict(
            enable_matrix_output=False,
        )
        if use_trajectory_mode:
            collate = partial(trajectory_collate_fn, **collate_kwargs)
        else:
            collate = partial(
                collate_fn,
                enable_history=cfg.enable_history_context,
                **collate_kwargs,
            )

        use_weighted = (
            cfg.photo_oversampling_factor != 1.0
            or cfg.multistep_oversampling_factor != 1.0
        )

        nw = cfg.num_dataloader_workers
        base_seed = torch.initial_seed() % (2**32)
        g = torch.Generator()
        g.manual_seed(base_seed)

        def _worker_init_fn(worker_id):
            """Seed numpy and random in each dataloader worker."""
            worker_seed = torch.initial_seed() % (2**32) + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        pin_memory = bool(cfg.pin_memory)
        persistent_workers = bool(cfg.persistent_workers)
        prefetch_factor = cfg.prefetch_factor
        if nw <= 0:
            pin_memory = False
            persistent_workers = False
            prefetch_factor = None
        elif prefetch_factor is not None and prefetch_factor < 1:
            prefetch_factor = 1

        self._loader_settings = {
            "num_workers": nw,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }

        loader_kwargs = dict(
            pin_memory=pin_memory,
            num_workers=nw,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            worker_init_fn=_worker_init_fn,
            generator=g,
        )

        self._train_sampler = self._build_train_sampler(
            train_dataset, use_weighted, base_seed
        )
        sampling_kwargs = (
            {"sampler": self._train_sampler}
            if self._train_sampler is not None
            else {"shuffle": True}
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate,
            **sampling_kwargs,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate,
            **loader_kwargs,
        )

        return train_loader, val_loader

    def _build_train_sampler(
        self,
        train_dataset: ShardedReactionDataset,
        use_weighted: bool,
        base_seed: int,
    ):
        """Build the training sampler for the current DDP/weighting setup.

        Returns:
            A (weighted) distributed sampler under DDP, a ShardAwareSampler
            for weighted single-process training, or None for plain shuffling.
        """
        if self.use_ddp:
            if use_weighted:
                weights = self._compute_sample_weights(train_dataset)
                return WeightedDistributedSampler(
                    train_dataset,
                    weights,
                    seed=base_seed,
                    shard_size=getattr(train_dataset, "shard_size", 0),
                )
            return DistributedSampler(train_dataset, shuffle=True)

        if use_weighted:
            weights = self._compute_sample_weights(train_dataset)
            return ShardAwareSampler(
                train_dataset,
                weights=weights,
                num_samples=len(weights),
                seed=base_seed,
            )
        return None

    def _compute_sample_weights(self, dataset: ShardedReactionDataset) -> list[float]:
        """Compute per-sample oversampling weights from dataset metadata.

        Weights multiply the photo factor for catalyst samples, the multistep
        factor for non-product samples at step index >= 1, and the late-step
        factor at step index >= 2.

        Args:
            dataset: Dataset providing per-sample metadata.

        Returns:
            list[float]: One weight per sample.
        """
        cfg = self.config
        weights = []
        for meta in dataset.metadata:
            w = 1.0
            if meta.get("has_catalyst", False):
                w *= cfg.photo_oversampling_factor
            if meta.get("step_index", 0) >= 1 and meta.get("type") != "product":
                w *= cfg.multistep_oversampling_factor
            if meta.get("step_index", 0) >= 2 and meta.get("type") != "product":
                w *= cfg.late_step_oversampling_factor
            weights.append(w)
        return weights

    def _print_config_summary(self) -> None:
        """Print device, loss, model size, and dataloader settings."""
        cfg = self.config
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Device: {self.device}, Precision: {cfg.dtype}, DDP: {self.use_ddp}")
        print(
            f"include_product: {cfg.include_product}, "
            f"multistep_oversampling: {cfg.multistep_oversampling_factor}"
        )
        if cfg.late_step_oversampling_factor != 1.0:
            print(f"late_step_oversampling: {cfg.late_step_oversampling_factor}")
        if cfg.history_noise_std > 0:
            print(
                f"history_noise_std: {cfg.history_noise_std}"
                f" (ramp_batches={cfg.history_noise_ramp_batches})"
            )
        if cfg.include_product:
            print(f"product_sampling_factor: {cfg.product_sampling_factor}")
        print(
            f"Loss: {cfg.ranking_loss_type}, pair_scoring_mode: {cfg.pair_scoring_mode}"
        )
        print(f"Model params: {param_count:,}")
        print(
            f"DataLoader workers: {self._loader_settings['num_workers']}, "
            f"pin_memory: {self._loader_settings['pin_memory']}, "
            f"persistent_workers: {self._loader_settings['persistent_workers']}, "
            f"prefetch_factor: {self._loader_settings['prefetch_factor']}, "
            f"shard_size: {cfg.shard_size}"
        )
        if cfg.gradient_accumulation_steps > 1:
            print(
                f"Gradient accumulation: {cfg.gradient_accumulation_steps} steps "
                f"(effective batch size = {cfg.batch_size * cfg.gradient_accumulation_steps})"
            )

    def _setup_mlflow(self) -> None:
        """Start the MLflow run and log git state, config parameters, and parameter count.

        Config values are truncated to 500 characters to satisfy MLflow's
        per-parameter value limit.
        """
        cfg = self.config
        mlflow.set_experiment(cfg.experiment_name or "bond_match_ranking")
        self._mlflow_run = mlflow.start_run(run_name=cfg.experiment_name)

        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=os.path.dirname(__file__),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            git_dirty = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=os.path.dirname(__file__),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            mlflow.set_tag("git_commit", git_commit)
            mlflow.set_tag("git_dirty", "yes" if git_dirty else "no")
        except Exception:
            pass

        params = cfg.model_dump()
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        mlflow.log_param("param_count", param_count)

    def _log_epoch_metrics(
        self, epoch: int, train_loss: float, lr: float, metrics: EvalMetrics
    ) -> None:
        """Log epoch-level training and evaluation metrics to MLflow."""
        log_dict = {
            "train_loss": train_loss,
            "val_loss": metrics.val_loss,
            "learning_rate": lr,
            "mrr": metrics.mrr,
            "top1": metrics.top1,
            "top3": metrics.top3,
            "top5": metrics.top5,
            "top10": metrics.top10,
            "bond_type_acc": metrics.bond_type_acc,
            "sim_separation": metrics.sim_separation,
        }
        _safe_mlflow_log(mlflow.log_metrics, log_dict, step=epoch + 1)

    def _log_checkpoint_artifact(self, filepath: str) -> None:
        """Log a checkpoint file as an MLflow artifact."""
        _safe_mlflow_log(mlflow.log_artifact, filepath, artifact_path="checkpoints")

    def _checkpoint_save_dir(self) -> str:
        """Return the checkpoint output directory: ``save_checkpoint_dir`` or the package default."""
        if self.config.save_checkpoint_dir:
            return os.path.abspath(self.config.save_checkpoint_dir)
        return os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "checkpoints",
            "train",
        )

    def _save_checkpoint(self, filename: str, epoch: int = -1) -> str:
        """Save model, optimizer, scheduler, and resume state to a checkpoint file.

        Loss-function, scaler, and EMA state dicts are included when present.

        Args:
            filename: Checkpoint file name within the save directory.
            epoch: Epoch index stored for resume; -1 if not epoch-bound.

        Returns:
            str: Path of the written checkpoint.
        """
        checkpoint_dir = self._checkpoint_save_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, filename)
        save_dict = {
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "batch_idx": self._last_batch_idx,
        }
        if list(self.loss_fn.parameters()):
            save_dict["loss_fn_state_dict"] = self.loss_fn.state_dict()
        if self.scaler is not None:
            save_dict["scaler_state_dict"] = self.scaler.state_dict()
        if self.ema is not None:
            save_dict["ema_state_dict"] = self.ema.state_dict()
        torch.save(save_dict, path)
        return path

    def _save_ema_checkpoint(self, filename: str, epoch: int = -1) -> str:
        """Save a checkpoint with EMA weights as ``model_state_dict`` for inference.

        Args:
            filename: Checkpoint file name within the save directory.
            epoch: Epoch index stored in the checkpoint.

        Returns:
            str: Path of the written checkpoint.
        """
        checkpoint_dir = self._checkpoint_save_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, filename)
        torch.save({"model_state_dict": self.ema.state_dict(), "epoch": epoch}, path)
        return path

    def train(self) -> None:
        """Run the full training loop over the configured number of epochs.

        After each epoch, rank 0 logs evaluation metrics and overwrites the
        resumable checkpoint ``<experiment_name>.pt``, so that the file holds the
        final model after the last epoch. No checkpoint is selected by an
        evaluation metric. Under DDP, all ranks synchronize at a barrier after
        every epoch. The MLflow run is ended by the caller (``main``).
        """
        cfg = self.config

        if cfg.enable_history_context:
            assert self.train_loader.collate_fn.keywords.get("enable_history", False), (
                "Model has enable_history_context=True but train dataloader was built "
                "without enable_history — history context would be silently dropped"
            )

        save_name = (
            f"{cfg.experiment_name}.pt" if cfg.experiment_name else "bondnn_ranking.pt"
        )

        if is_main_process() and (self.start_epoch > 0 or self._resume_batch_idx >= 0):
            batch_str = (
                f", batch {self._resume_batch_idx}"
                if self._resume_batch_idx >= 0
                else ""
            )
            print(
                f"Resuming training from epoch {self.start_epoch + 1}/{cfg.epochs}{batch_str}"
            )

        for epoch in range(self.start_epoch, cfg.epochs):
            if self._train_sampler is not None:
                self._train_sampler.set_epoch(epoch)

            train_loss = self._train_epoch(epoch, save_name)

            self.scheduler.step()

            if is_main_process():
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} - Avg Loss: {train_loss:.4f}, LR: {lr:.6f}"
                )

                metrics = self.evaluate()

                self._last_batch_idx = -1
                ckpt_path = self._save_checkpoint(save_name, epoch=epoch)
                if cfg.save_every_epoch:
                    epoch_name = f"{save_name.replace('.pt', '')}_epoch{epoch + 1}.pt"
                    epoch_ckpt_path = self._save_checkpoint(epoch_name, epoch=epoch)
                if self._mlflow_enabled:
                    self._log_epoch_metrics(epoch, train_loss, lr, metrics)
                    self._log_checkpoint_artifact(ckpt_path)
                    if cfg.save_every_epoch:
                        self._log_checkpoint_artifact(epoch_ckpt_path)

            if self.use_ddp:
                dist.barrier()

    def _intra_epoch_eval_save(self, epoch: int, batch_idx: int, save_name: str) -> None:
        """Run evaluation and save a checkpoint mid-epoch.

        Args:
            epoch: Current epoch index.
            batch_idx: Current per-rank batch index within the epoch.
            save_name: Base checkpoint file name.
        """
        global_step = (epoch * len(self.train_loader) + batch_idx + 1) * self.world_size
        print(
            f"\n  [Intra-epoch eval] Epoch {epoch + 1}/{self.config.epochs}, batch {batch_idx + 1}/{len(self.train_loader)}"
        )

        metrics = self.evaluate()

        tag = f"batch{batch_idx + 1}"
        ckpt_path = self._save_checkpoint(
            f"{save_name.replace('.pt', '')}_{tag}.pt", epoch=epoch
        )

        if self._mlflow_enabled:
            _safe_mlflow_log(
                mlflow.log_metrics,
                {
                    "intra_mrr": metrics.mrr,
                    "intra_top1": metrics.top1,
                    "intra_stop_acc": metrics.stop_acc,
                    "intra_val_loss": metrics.val_loss,
                },
                step=global_step,
            )
            self._log_checkpoint_artifact(ckpt_path)

        self.model.train()

    def _train_epoch(self, epoch: int, save_name: str) -> float:
        """Train one epoch with gradient accumulation and optional batch-level warmup.

        Batch counters use a global convention: per-rank batch indices are
        multiplied by ``world_size`` so that ``warmup_batches``,
        ``eval_every_n_batches``, and MLflow steps refer to total batches
        across all GPUs regardless of GPU count. ``eval_every_n_batches`` is
        therefore divided by ``world_size`` to obtain the per-rank interval.
        Batch-level linear warmup (``warmup_batches > 0``) overrides the
        epoch-level warmup schedule. When resuming mid-epoch, batches up to
        the stored index are skipped. A smoothed loss (exponential moving
        average with smoothing factor 0.01) is logged alongside the raw batch
        loss.

        Args:
            epoch: Epoch index.
            save_name: Base checkpoint file name for intra-epoch saves.

        Returns:
            float: Mean training loss over the epoch.
        """
        self.model.train()
        cfg = self.config
        accum_steps = cfg.gradient_accumulation_steps
        total_loss = torch.zeros((), device=self.device)
        log_interval = 100
        ema_loss = None
        ema_alpha = 0.01
        eval_every = cfg.eval_every_n_batches or 0
        warmup_batches = cfg.warmup_batches
        base_lr = cfg.learning_rate
        batches_per_epoch = len(self.train_loader)
        ws = self.world_size
        eval_every_local = max(1, eval_every // ws) if eval_every > 0 else 0

        self.optimizer.zero_grad()

        loop = tqdm(
            enumerate(self.train_loader),
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            total=batches_per_epoch,
            disable=not is_main_process(),
        )

        skip_to = self._resume_batch_idx
        if skip_to >= 0 and epoch == self.start_epoch:
            if is_main_process():
                print(f"  Skipping to batch {skip_to + 1} (resuming mid-epoch)")

        for batch_idx, batch in loop:
            if skip_to >= 0 and epoch == self.start_epoch and batch_idx <= skip_to:
                if batch_idx == skip_to:
                    self._resume_batch_idx = -1
                continue

            if warmup_batches > 0:
                global_batch = (epoch * batches_per_epoch + batch_idx) * ws
                if global_batch < warmup_batches:
                    warmup_lr = base_lr * (global_batch + 1) / warmup_batches
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = warmup_lr
            else:
                global_batch = (epoch * batches_per_epoch + batch_idx) * ws

            self._global_batch_idx = global_batch

            self._last_batch_idx = batch_idx

            loss = self._train_step(batch, batch_idx, accum_steps)

            is_sync_step = (batch_idx + 1) % accum_steps == 0 or (
                batch_idx + 1
            ) == batches_per_epoch
            if is_sync_step:
                self._optimizer_step()

            total_loss = total_loss + loss
            ema_loss = (
                loss
                if ema_loss is None
                else (1 - ema_alpha) * ema_loss + ema_alpha * loss
            )

            should_report = is_main_process() and (
                (batch_idx + 1) % log_interval == 0
                or (batch_idx + 1) == batches_per_epoch
            )
            if should_report:
                loss_value = float(loss.cpu())
                loop.set_postfix(loss=loss_value)

            if (
                self._mlflow_enabled
                and should_report
                and (batch_idx + 1) % log_interval == 0
            ):
                step = (epoch * batches_per_epoch + batch_idx + 1) * ws
                log_dict = {"batch_loss": loss_value, "smooth_loss": float(ema_loss.cpu())}
                _safe_mlflow_log(mlflow.log_metrics, log_dict, step=step)

            if eval_every_local > 0 and (batch_idx + 1) % eval_every_local == 0:
                if is_main_process():
                    self._intra_epoch_eval_save(epoch, batch_idx, save_name)
                if self.use_ddp:
                    dist.barrier()

        return float((total_loss / len(self.train_loader)).cpu())

    def _forward(self, inputs: BatchInputs, use_ddp_model: bool = False) -> dict:
        """Run a model forward pass with encoded history context.

        When ``drop_head_bond_tokens_after_gnn`` is enabled and the model
        returns a ``bond_keep_mask``, the per-bond tensors of ``inputs``
        (``y``, ``bond_indices``, ``bond_batch_vec``) are filtered in place
        to match the retained bond tokens.

        Args:
            inputs: Batch tensors on the target device.
            use_ddp_model: Use the DDP-wrapped model (required for gradient
                synchronization during training) instead of the raw model.

        Returns:
            dict: Model outputs.
        """
        model = self.model if use_ddp_model else self.raw_model
        history_context, history_mask = self._encode_history(
            inputs.history_batch,
            inputs.history_map,
            inputs.step_index.shape[0],
            drop_parent_mask=getattr(inputs, "drop_history_parent_mask", None),
        )
        out = model(
            inputs.x,
            inputs.batch_idx,
            inputs.edge_index,
            inputs.edge_attr,
            inputs.bond_indices,
            history_context=history_context,
            history_mask=history_mask,
            step_index=inputs.step_index,
            trajectory_ids=inputs.trajectory_ids,
        )
        keep_mask = out.get("bond_keep_mask")
        if (
            self.config.drop_head_bond_tokens_after_gnn
            and keep_mask is not None
            and keep_mask.numel() == inputs.y.numel()
        ):
            inputs.y = inputs.y[keep_mask]
            inputs.bond_indices = inputs.bond_indices[keep_mask]
            inputs.bond_batch_vec = inputs.bond_batch_vec[keep_mask]
        return out

    def _ddp_parameter_guard(self) -> torch.Tensor:
        """Return a zero-valued term that touches every trainable parameter.

        Added to the loss under DDP so that all parameters participate in the
        autograd graph each step, keeping gradient reduction in sync even
        when conditional branches leave some parameters unused.

        Returns:
            torch.Tensor: Scalar tensor equal to zero.
        """
        guard = None
        for param in self.raw_model.parameters():
            if param.requires_grad and param.numel() > 0:
                term = param.reshape(-1)[0] * 0.0
                guard = term if guard is None else guard + term
        if guard is None:
            return torch.zeros((), device=self.device)
        return guard

    def _current_history_noise_std(self) -> float:
        """Return the history-noise standard deviation, ramped linearly over the configured batch count."""
        base_std = self.config.history_noise_std
        if base_std <= 0:
            return 0.0

        ramp_batches = self.config.history_noise_ramp_batches
        if ramp_batches <= 0:
            return base_std

        ramp = min(1.0, float(self._global_batch_idx + 1) / float(ramp_batches))
        return base_std * ramp

    def _train_step(
        self, batch: tuple, batch_idx: int, accum_steps: int
    ) -> torch.Tensor:
        """Forward/backward pass for one batch, including configured auxiliary losses.

        The base ranking loss is augmented by ``_add_auxiliary_losses`` (see
        the individual ``_*_loss`` helpers for the terms). Under DDP,
        gradient synchronization is suppressed via ``no_sync()`` on
        non-accumulation-boundary steps and a parameter guard keeps all
        parameters in the autograd graph. The loss is divided by
        ``accum_steps`` before backward.

        Args:
            batch: Collated batch tuple.
            batch_idx: Per-rank batch index within the epoch.
            accum_steps: Gradient accumulation steps.

        Returns:
            torch.Tensor: Detached, unscaled (multiplied back by
            ``accum_steps``) loss value.
        """
        inputs = BatchInputs.from_batch(batch, self.device)
        self._apply_product_history_dropout(inputs)

        is_sync_step = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(
            self.train_loader
        )
        ddp_ctx = (
            self.model.no_sync()
            if (self.use_ddp and not is_sync_step)
            else nullcontext()
        )

        with ddp_ctx:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = self._forward(inputs, use_ddp_model=True)
                out["model_ref"] = self.raw_model
                loss = self.loss_fn(
                    out,
                    inputs.y,
                    inputs.bond_batch_vec,
                    inputs.is_over_target,
                    inputs.is_negative_reaction,
                )
                loss = self._add_auxiliary_losses(loss, out, inputs)

                if self.use_ddp:
                    loss = loss + self._ddp_parameter_guard()

                loss = loss / accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss.detach().float() * accum_steps

    def _add_auxiliary_losses(
        self, loss: torch.Tensor, out: dict, inputs: BatchInputs
    ) -> torch.Tensor:
        """Apply the configured auxiliary loss terms in their fixed order.

        The multistep scale multiplies everything added before it (base loss,
        step-count aux, MoE aux, calibration) but not the bond-contrastive or
        focal terms added after it.
        """
        step_aux = self._step_count_aux_loss(out, inputs)
        if step_aux is not None:
            loss = loss + step_aux

        if "moe_aux_loss" in out:
            loss = loss + out["moe_aux_loss"]

        calibration = self._pair_score_calibration_loss(out, inputs)
        if calibration is not None:
            loss = loss + calibration

        multistep_scale = self._multistep_loss_scale(inputs)
        if multistep_scale is not None:
            loss = loss * multistep_scale

        contrastive = self._bond_contrastive_loss(out, inputs)
        if contrastive is not None:
            loss = loss + contrastive

        focal_adjustment = self._focal_binary_adjustment(out, inputs)
        if focal_adjustment is not None:
            loss = loss + focal_adjustment

        return loss

    def _step_count_aux_loss(
        self, out: dict, inputs: BatchInputs
    ) -> torch.Tensor | None:
        """Weighted cross-entropy on the step-count head (``step_count_aux_weight``)."""
        step_w = self.config.step_count_aux_weight
        if step_w <= 0 or "logits_step_count" not in out:
            return None
        step_targets = inputs.step_index.clamp(0, 3).long()
        return step_w * F.cross_entropy(out["logits_step_count"], step_targets)

    def _pair_score_calibration_loss(
        self, out: dict, inputs: BatchInputs
    ) -> torch.Tensor | None:
        """Push the per-molecule max binary logit toward ``pair_score_target``.

        Applies only to reactive (non-stop) states, with a fixed weight of
        0.01.
        """
        ps_target = self.config.pair_score_target
        if ps_target <= 0 or "logits_matrix" in out:
            return None
        binary_logits = out["logits_binary"].squeeze(-1)
        max_per_mol = global_max_pool(
            binary_logits.unsqueeze(-1),
            inputs.bond_batch_vec,
        ).squeeze(-1)
        reactive_mask = (inputs.is_over_target == 0).float()
        if reactive_mask.sum() == 0:
            return None
        calib_loss = (
            (max_per_mol - ps_target) ** 2 * reactive_mask
        ).sum() / reactive_mask.sum().clamp(min=1)
        return 0.01 * calib_loss

    def _multistep_loss_scale(self, inputs: BatchInputs) -> torch.Tensor | None:
        """Batch-level loss scale proportional to the fraction of multi-step samples.

        Returns:
            Scale factor in ``(1, multistep_loss_scale)``, or None when
            scaling is disabled or the batch is uniformly single- or
            multi-step.
        """
        ms_scale = self.config.multistep_loss_scale
        if ms_scale <= 1.0:
            return None
        is_multistep = (inputs.step_index >= 1).float()
        n_total = inputs.step_index.size(0)
        n_multi = is_multistep.sum()
        if n_multi > 0 and n_multi < n_total:
            return 1.0 + (ms_scale - 1.0) * (n_multi / n_total)
        return None

    def _bond_contrastive_loss(
        self, out: dict, inputs: BatchInputs
    ) -> torch.Tensor | None:
        """InfoNCE loss over normalized bond embeddings (temperature 0.07).

        Per molecule, one reactive bond is the anchor and the other the
        positive; the remaining bonds act as negatives.
        """
        bcw = self.config.bond_contrastive_weight
        if bcw <= 0:
            return None
        z_norm = F.normalize(out["embeddings"], dim=-1)
        contrastive_losses = []
        for mol_id in torch.unique(inputs.bond_batch_vec):
            mol_mask = inputs.bond_batch_vec == mol_id
            z_mol = z_norm[mol_mask]
            y_mol = inputs.y[mol_mask]
            reactive_idx = torch.where(y_mol > 0)[0]
            if reactive_idx.numel() == 2 and z_mol.size(0) >= 3:
                z_anchor = z_mol[reactive_idx[0]]
                sims = (z_anchor.unsqueeze(0) * z_mol).sum(dim=-1) / 0.07
                target_idx = reactive_idx[1]
                contrastive_losses.append(
                    F.cross_entropy(sims.unsqueeze(0), target_idx.unsqueeze(0))
                )
        if not contrastive_losses:
            return None
        return bcw * torch.stack(contrastive_losses).mean()

    def _focal_binary_adjustment(
        self, out: dict, inputs: BatchInputs
    ) -> torch.Tensor | None:
        """Swap the standard BCE contribution for a focal one (``binary_focal_gamma``).

        Adds ``binary_weight * (focal - standard)``, which cancels the plain
        BCE term inside the ranking loss and replaces it with the focal
        variant.
        """
        bfg = self.config.binary_focal_gamma
        if bfg <= 0:
            return None
        b_logits = out["logits_binary"]
        b_targets = (inputs.y > 0).float().unsqueeze(-1)
        standard_b = F.binary_cross_entropy_with_logits(b_logits, b_targets)
        bce_raw = F.binary_cross_entropy_with_logits(
            b_logits, b_targets, reduction="none"
        )
        p = torch.sigmoid(b_logits)
        p_t = p * b_targets + (1 - p) * (1 - b_targets)
        focal_weight = (1 - p_t) ** bfg
        focal_b = (focal_weight * bce_raw).mean()
        return self.config.binary_weight * (focal_b - standard_b)

    def _apply_product_history_dropout(self, inputs: BatchInputs) -> None:
        """Randomly re-present product/stop states as first-step stop states.

        With probability ``drop_product_history_prob``, eligible samples
        (stop states past step 0) get ``step_index`` reset to 0 and their
        history marked for dropping, so the model also learns to stop without
        history context.
        """
        prob = getattr(self.config, "drop_product_history_prob", 0.0)
        if prob <= 0:
            return

        # Only product/stop states that already live in the post-step regime are eligible.
        product_mask = (inputs.is_over_target == 1) & (inputs.step_index > 0)
        if not torch.any(product_mask):
            return

        sampled = (
            torch.rand(inputs.step_index.shape[0], device=inputs.step_index.device)
            < prob
        )
        drop_mask = product_mask & sampled
        if not torch.any(drop_mask):
            return

        # Present a subset of final products as first-step stop states:
        # no history and step_index=0, matching regular step-0 semantics.
        inputs.step_index = inputs.step_index.clone()
        inputs.step_index[drop_mask] = 0
        inputs.drop_history_parent_mask = drop_mask

    def _optimizer_step(self) -> None:
        """Clip gradients, step the optimizer (and scaler), and update the EMA."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.gradient_clip
            )
            self.optimizer.step()
        self.optimizer.zero_grad()
        if self.ema is not None:
            self.ema.update(self.raw_model)

    def _encode_history(
        self,
        history_batch,
        history_map,
        batch_size: int,
        drop_parent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Encode history molecule graphs into padded per-sample context tokens.

        History bond tokens are encoded with empty history context (matching
        eval's forward path), optionally augmented with step-position
        embeddings, padded per parent sample, perturbed with ramped Gaussian
        noise during training, and blanked for samples flagged in
        ``drop_parent_mask``.

        Args:
            history_batch: Collated history graph tensors, or None.
            history_map: Maps each history molecule to its parent sample.
            batch_size: Number of samples in the current batch.
            drop_parent_mask: Per-sample mask of histories to blank out.

        Returns:
            Tuple of (history context ``[B, T, D]``, key-padding mask
            ``[B, T]`` with True marking positions to ignore), or
            (None, None) when history context is disabled.
        """
        if history_batch is None or history_map is None:
            if self.config.enable_history_context:
                return self.raw_model.empty_history(batch_size, self.device)
            return None, None

        (
            hx,
            _,
            h_batch_idx,
            h_ei,
            h_ea,
            h_bonds,
            _,
            _,
            _,
            _,
            h_step_index,
        ) = history_batch

        hx = hx.to(self.device)
        h_batch_idx = h_batch_idx.to(self.device)
        h_ei = h_ei.to(self.device)
        h_ea = h_ea.to(self.device)
        h_bonds = h_bonds.to(self.device)
        h_step_index = h_step_index.to(self.device)
        history_map = history_map.to(self.device)

        use_history_grad = self.config.history_grad
        ctx = torch.no_grad() if not use_history_grad else torch.enable_grad()
        with ctx:
            # Pass empty history context so cross-attention runs on empty input,
            # matching eval's forward() path for consistency
            n_hist_mols = h_step_index.numel()
            empty_ctx, empty_mask = self.raw_model.empty_history(
                n_hist_mols, self.device
            )
            h_bond_tokens, h_bond_batch = self.raw_model.encode_history_tokens(
                hx,
                h_batch_idx,
                h_ei,
                h_ea,
                h_bonds,
                step_index=h_step_index,
                history_context=empty_ctx,
                history_mask=empty_mask,
            )

        if self.config.use_history_step_pos and hasattr(
            self.raw_model, "history_step_pos_emb"
        ):
            step_per_token = h_step_index[h_bond_batch].clamp(0, 7)
            h_bond_tokens = h_bond_tokens + self.raw_model.history_step_pos_emb(
                step_per_token
            )

        parent_map = history_map[h_bond_batch]
        history_context, history_mask_raw = to_dense_batch(
            h_bond_tokens,
            parent_map,
            batch_size=batch_size,
        )
        history_mask = ~history_mask_raw  # key_padding_mask: True=ignore

        # Simulate imperfect generated histories during training.
        noise_std = self._current_history_noise_std()
        if self.model.training and noise_std > 0:
            noise = torch.randn_like(history_context) * noise_std
            valid = ~history_mask.unsqueeze(-1)  # [B, T, 1] True=valid
            history_context = history_context + noise * valid

        if drop_parent_mask is not None and torch.any(drop_parent_mask):
            history_context = history_context.clone()
            history_mask = history_mask.clone()
            history_context[drop_parent_mask] = 0.0
            history_mask[drop_parent_mask] = True

        return history_context, history_mask

    def evaluate(self) -> EvalMetrics:
        """Evaluate ranking metrics and loss over the validation loader.

        Uses EMA weights when EMA is enabled.

        Returns:
            Aggregated evaluation metrics from the EvalCollector.
        """
        if self.config.enable_history_context:
            assert self.val_loader.collate_fn.keywords.get("enable_history", False), (
                "Model has enable_history_context=True but validation dataloader was built "
                "without enable_history — history context would be silently dropped"
            )

        use_ema = self.ema is not None
        ema_ctx = self.ema.apply(self.raw_model) if use_ema else nullcontext()
        with ema_ctx:
            if is_main_process():
                print(f"  Evaluating {'(EMA weights)' if use_ema else '(raw weights)'}")
            self.raw_model.eval()
            collector = EvalCollector()

            with torch.no_grad():
                for batch in tqdm(
                    self.val_loader, desc="Evaluating", disable=not is_main_process()
                ):
                    inputs = BatchInputs.from_batch(batch, self.device)
                    out = self._forward(inputs)
                    self._eval_batch(out, inputs, collector)
                    out["model_ref"] = self.raw_model
                    loss = self.loss_fn(
                        out,
                        inputs.y,
                        inputs.bond_batch_vec,
                        inputs.is_over_target,
                        inputs.is_negative_reaction,
                    )
                    collector.add_loss(loss.item())

        return collector.compute_metrics()

    def _eval_batch(
        self, out: dict, inputs: BatchInputs, collector: EvalCollector
    ) -> None:
        """Collect ranking, stop, and bond-type metrics for one validation batch.

        Per molecule, all pair scores plus the stop logit are ranked; stop
        accuracy is recorded for every molecule, while ranking and bond-type
        metrics are recorded only for non-stop predictions on molecules with
        an annotated reactive pair.
        """
        embeddings = out["embeddings"]
        binary_logits = out["logits_binary"]
        stop_logits = torch.nan_to_num(
            out["logits_stop"].squeeze(-1), nan=-1e9, posinf=-1e9, neginf=-1e9
        )
        bond_type_logits = out.get("logits_bond_type", None)
        model = self.raw_model

        preds = torch.zeros_like(inputs.y)

        for mol_idx, mol_id in enumerate(torch.unique(inputs.bond_batch_vec)):
            mol_mask = inputs.bond_batch_vec == mol_id
            mol_indices = torch.where(mol_mask)[0]
            n_bonds = mol_indices.size(0)

            mol_targets = inputs.y[mol_indices]
            z_mol = embeddings[mol_mask]

            raw_pair_scores, triu_idx = model.compute_pair_scores_upper_triangle(
                z_mol,
                binary_logits=binary_logits[mol_indices],
                mode=self.config.pair_scoring_mode,
            )
            pair_scores = torch.nan_to_num(
                raw_pair_scores, nan=-1e9, posinf=-1e9, neginf=-1e9
            )

            action_scores = torch.cat(
                [pair_scores, stop_logits[mol_idx].unsqueeze(0)], dim=0
            )
            top_action_idx = int(torch.argmax(action_scores).item())
            pred_is_over = top_action_idx == pair_scores.numel()
            collector.add_stop(
                int(pred_is_over == bool(inputs.is_over_target[mol_idx].item())),
                1,
            )

            if pred_is_over or n_bonds <= 1:
                continue

            u_local = triu_idx[0][top_action_idx]
            v_local = triu_idx[1][top_action_idx]

            bi_all = out.get("bond_indices_out")
            pred_class = self._predict_bond_type(
                model,
                z_mol,
                u_local,
                v_local,
                bond_type_logits,
                mol_idx,
                node_embeddings=out.get("node_embeddings"),
                mol_bond_indices=bi_all[mol_mask] if bi_all is not None else None,
            )
            preds[mol_indices[u_local]] = pred_class
            preds[mol_indices[v_local]] = pred_class

            self._collect_reactive_metrics(
                collector,
                mol_targets,
                raw_pair_scores,
                action_scores,
                triu_idx,
                pred_class,
            )

        collector.add_preds(
            preds.cpu().numpy().tolist(),
            inputs.y.cpu().numpy().tolist(),
        )

    @staticmethod
    def _collect_reactive_metrics(
        collector: EvalCollector,
        mol_targets: torch.Tensor,
        raw_pair_scores: torch.Tensor,
        action_scores: torch.Tensor,
        triu_idx: torch.Tensor,
        pred_class: int,
    ) -> None:
        """Record ranking and bond-type metrics for one reactive molecule.

        No-ops when the molecule does not contain exactly two reactive bonds.
        """
        reactive_idx = torch.where(mol_targets > 0)[0]
        if reactive_idx.numel() != 2:
            return

        collector.total_reactive += 1
        r1, r2 = sorted([reactive_idx[0].item(), reactive_idx[1].item()])

        true_class = mol_targets[reactive_idx[0]].item()
        collector.add_bond_type(pred_class, true_class)

        true_pair_mask = (triu_idx[0] == r1) & (triu_idx[1] == r2)
        true_pair_list_idx = torch.where(true_pair_mask)[0]
        if true_pair_list_idx.numel() == 0:
            return

        list_idx = true_pair_list_idx.item()
        action_ranks = torch.argsort(action_scores, descending=True)
        rank = (action_ranks == list_idx).nonzero(as_tuple=True)[0].item()
        collector.add_ranking_hit(rank, 1.0 / (rank + 1))

        neg_mask = ~true_pair_mask
        neg_sims = raw_pair_scores[neg_mask][:5].tolist() if neg_mask.any() else []
        collector.add_sim(raw_pair_scores[list_idx].item(), neg_sims)

    @staticmethod
    def _predict_bond_type(
        model: nn.Module,
        z_mol: torch.Tensor,
        u_local: torch.Tensor,
        v_local: torch.Tensor,
        bond_type_logits: torch.Tensor | None,
        mol_idx: int,
        node_embeddings: torch.Tensor | None = None,
        mol_bond_indices: torch.Tensor | None = None,
    ) -> int:
        """Predict the swap/bond-type class for the selected bond pair.

        Uses the pair swap classifier when the model defines one, otherwise
        the per-molecule bond-type logits; defaults to class 1 when neither
        is available.

        Args:
            model: Raw (unwrapped) model.
            z_mol: Bond embeddings of the molecule.
            u_local: Index of the first selected bond within the molecule.
            v_local: Index of the second selected bond within the molecule.
            bond_type_logits: Per-molecule bond-type logits, if produced.
            mol_idx: Molecule position within the batch.
            node_embeddings: Node embeddings for atom-based swap heads.
            mol_bond_indices: Bond atom-index pairs for atom-based swap heads.

        Returns:
            Predicted class label (1-based; never the "stable" class 0).
        """
        if model.use_pair_bond_type and hasattr(model, "pair_bond_type_classifier"):
            classifier = model.pair_bond_type_classifier

            if isinstance(classifier, MLPSwapTypeHead):
                logits = classifier(z_mol[u_local], z_mol[v_local])
            else:
                assert node_embeddings is not None and mol_bond_indices is not None, (
                    "_predict_bond_type: node_embeddings or mol_bond_indices is None."
                )
                h_u1 = node_embeddings[mol_bond_indices[u_local, 0]]
                h_u2 = node_embeddings[mol_bond_indices[u_local, 1]]
                h_v1 = node_embeddings[mol_bond_indices[v_local, 0]]
                h_v2 = node_embeddings[mol_bond_indices[v_local, 1]]

                if isinstance(classifier, BondSplitSwapHead):
                    logits = classifier(
                        z_mol[u_local],
                        z_mol[v_local],
                        h_u1,
                        h_u2,
                        h_v1,
                        h_v2,
                    )
                elif isinstance(classifier, (AtomSwapTypeHead, NodeAttentionSwapHead)):
                    logits = classifier(h_u1, h_u2, h_v1, h_v2)
                else:
                    raise TypeError(f"Unknown classifier: {type(classifier)}")
            if isinstance(classifier, NodeAttentionSwapHead):
                return torch.argmax(logits[0]).item() + 1  # 2-class: 0→1, 1→2
            return torch.argmax(logits[0, 1:]).item() + 1

        if bond_type_logits is not None and mol_idx < bond_type_logits.size(0):
            return torch.argmax(bond_type_logits[mol_idx][1:]).item() + 1

        return 1
