"""
Ranking losses for bond-pair prediction.

The losses optimize pair-level ranking: the annotated reactive bond pair is
scored above alternative bond pairs and the stop action. This matches the
pair-level evaluation metric used during mechanism prediction.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from bond_match.model.heads import (
    AtomSwapTypeHead,
    BondSplitSwapHead,
    MLPSwapTypeHead,
    NodeAttentionSwapHead,
)
from bond_match.model.scorers import AttentionPairScorer

PAIR_SCORE_CHUNK_SIZE = 32768


class _MolPairInfo(NamedTuple):
    """Per-molecule bookkeeping carried between the score-precompute and
    loss-accumulation passes of ``EBMPairRankingLoss._forward_loop``."""

    mol_id: torch.Tensor
    mol_mask: torch.Tensor
    n_bonds: int
    n_pairs: int
    reactive_idx: torch.Tensor | None
    y_mol: torch.Tensor
    triu_idx: torch.Tensor | None


def _zero(device: torch.device) -> torch.Tensor:
    """Return a scalar zero tensor on ``device``."""
    return torch.tensor(0.0, device=device)


def _stack_mean(losses: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Return the mean of the stacked loss terms, or scalar zero if empty."""
    if losses:
        return torch.stack(losses).mean()
    return _zero(device)


def _binary_reactivity_targets(
    targets: torch.Tensor,
    bond_batch_vec: torch.Tensor,
    is_over_targets: torch.Tensor,
    is_negative_reaction: torch.Tensor | None,
) -> torch.Tensor:
    """Build per-bond binary reactivity targets.

    Bonds of yield-0 negative reactions (that are not stop states) are forced
    to 0 so the binary head does not learn reactivity from them.

    Args:
        targets: Per-bond integer labels; > 0 marks reactive bonds.
        bond_batch_vec: Molecule index for every bond token.
        is_over_targets: Per-molecule stop labels.
        is_negative_reaction: Optional per-molecule yield-0 negative flags.

    Returns:
        Float targets of shape ``(n_bonds, 1)``.
    """
    binary_targets = (targets > 0).float().unsqueeze(-1)
    if is_negative_reaction is not None:
        negative_mech_mask = is_negative_reaction[bond_batch_vec] & (
            ~is_over_targets[bond_batch_vec].bool()
        )
        binary_targets[negative_mech_mask] = 0.0
    return binary_targets


def _ordered_reactive_pair(
    reactive_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two reactive bond indices ordered as (lower, higher)."""
    if reactive_idx[0] < reactive_idx[1]:
        return reactive_idx[0], reactive_idx[1]
    return reactive_idx[1], reactive_idx[0]


def _true_pair_action_index(
    triu_idx: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, default_idx: int
) -> int:
    """Return the position of pair (r1, r2) in the upper-triangle pair list.

    Falls back to ``default_idx`` (the stop action) when the pair is not
    present, e.g. because one bond was dropped.
    """
    true_pair_mask = (triu_idx[0] == r1) & (triu_idx[1] == r2)
    true_pair_idx = torch.where(true_pair_mask)[0]
    if true_pair_idx.numel() > 0:
        return true_pair_idx[0].item()
    return default_idx


def _select_hard_negatives(
    action_scores: torch.Tensor, target_action_idx: int, hard_negative_k: int
) -> tuple[torch.Tensor, int]:
    """Restrict the contrastive set to the target plus the top-k negatives.

    Args:
        action_scores: All action scores (pairs followed by stop).
        target_action_idx: Index of the target action within ``action_scores``.
        hard_negative_k: Number of highest-scoring negatives to keep; 0
            disables subsampling.

    Returns:
        Tuple of (selected scores, target index within the selection). When
        subsampling is disabled or there are not enough negatives, the input
        is returned unchanged.
    """
    if hard_negative_k > 0 and action_scores.numel() > hard_negative_k + 1:
        target_score = action_scores[target_action_idx]
        neg_mask = torch.ones_like(action_scores, dtype=torch.bool)
        neg_mask[target_action_idx] = False
        neg_scores = action_scores[neg_mask]
        k = min(hard_negative_k, neg_scores.size(0))
        top_k_neg, _ = torch.topk(neg_scores, k)
        return torch.cat([target_score.unsqueeze(0), top_k_neg]), 0
    return action_scores, target_action_idx


def _compute_swap_logits(
    model: nn.Module,
    out_dict: dict,
    bond_batch_vec: torch.Tensor,
    mol_id: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    z_mol: torch.Tensor,
) -> torch.Tensor:
    """Compute swap type logits for a given bond pair.

    Handles both AtomSwapTypeHead (uses node embeddings) and
    BondSplitSwapHead (uses post-transformer bond embeddings + optional node guide).

    Args:
        model: Model exposing ``pair_bond_type_classifier``.
        out_dict: Model outputs (provides node embeddings and bond indices).
        bond_batch_vec: Molecule index for every bond token.
        mol_id: Molecule id to select bonds for.
        r1: Index of the first reactive bond within the molecule.
        r2: Index of the second reactive bond within the molecule.
        z_mol: Bond embeddings of this molecule.

    Returns:
        Swap-type logits for the (r1, r2) pair.
    """
    classifier = model.pair_bond_type_classifier

    # MLPSwapTypeHead consumes concatenated bond embeddings directly.
    if isinstance(classifier, MLPSwapTypeHead):
        return classifier(z_mol[r1], z_mol[r2])

    # AtomSwapTypeHead and BondSplitSwapHead require endpoint node embeddings.
    h_nodes = out_dict.get("node_embeddings")
    bi = out_dict.get("bond_indices_out")
    assert h_nodes is not None, (
        "use_pair_bond_type=True but 'node_embeddings' missing from model output."
    )
    assert bi is not None, (
        "use_pair_bond_type=True but 'bond_indices_out' missing from model output."
    )
    mol_bond_mask = bond_batch_vec == mol_id
    mol_bond_indices = bi[mol_bond_mask]
    h_u1 = h_nodes[mol_bond_indices[r1, 0]]
    h_u2 = h_nodes[mol_bond_indices[r1, 1]]
    h_v1 = h_nodes[mol_bond_indices[r2, 0]]
    h_v2 = h_nodes[mol_bond_indices[r2, 1]]

    if isinstance(classifier, BondSplitSwapHead):
        return classifier(z_mol[r1], z_mol[r2], h_u1, h_u2, h_v1, h_v2)
    elif isinstance(classifier, (AtomSwapTypeHead, NodeAttentionSwapHead)):
        return classifier(h_u1, h_u2, h_v1, h_v2)
    else:
        raise TypeError(f"Unknown swap type classifier: {type(classifier)}")


def _swap_target(model: nn.Module, y_val: int, device: torch.device) -> torch.Tensor:
    """Remap dataset target to match head output classes.

    NodeAttentionSwapHead outputs 2 classes (0=swap0, 1=swap1).
    All other heads output 3 classes (0=stable, 1=swap0, 2=swap1).
    Dataset labels: 1=swap0, 2=swap1.
    """
    if isinstance(model.pair_bond_type_classifier, NodeAttentionSwapHead):
        return torch.tensor([y_val - 1], device=device, dtype=torch.long)
    return torch.tensor([y_val], device=device, dtype=torch.long)


def _per_mol_bond_type_loss(
    bond_type_ce: nn.Module,
    out_dict: dict,
    bond_batch_vec: torch.Tensor,
    mol_id: torch.Tensor,
    mol_idx_in_batch: int,
    r1: torch.Tensor | None,
    r2: torch.Tensor | None,
    z_mol: torch.Tensor,
    y_mol: torch.Tensor,
    is_negative_mech: bool,
    device: torch.device,
) -> torch.Tensor | None:
    """Bond-/swap-type CE loss for one molecule's annotated reactive pair.

    Prefers the pair swap classifier when the model defines one, otherwise
    falls back to per-molecule ``logits_bond_type``.

    Returns:
        Loss term, or None when the molecule has no reactive pair, is a
        negative-mechanism sample, or neither classification path exists.
    """
    if r1 is None or is_negative_mech:
        return None

    model = out_dict["model_ref"]
    if getattr(model, "use_pair_bond_type", False) and hasattr(
        model, "pair_bond_type_classifier"
    ):
        pair_bt_logits = _compute_swap_logits(
            model, out_dict, bond_batch_vec, mol_id, r1, r2, z_mol
        )
        bt_target = _swap_target(model, y_mol[r1].item(), device)
        return bond_type_ce(pair_bt_logits, bt_target)

    bond_type_logits = out_dict.get("logits_bond_type")
    if bond_type_logits is not None:
        bt_target = torch.tensor([y_mol[r1].item()], device=device)
        return bond_type_ce(bond_type_logits[mol_idx_in_batch].unsqueeze(0), bt_target)

    return None


def _compute_pair_scores_chunked(
    model: nn.Module,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    binary_logits_i: torch.Tensor | None = None,
    binary_logits_j: torch.Tensor | None = None,
    mode: str = "pair_only",
) -> torch.Tensor:
    """Compute pair scores in manageable chunks to cap peak GPU memory."""
    n_pairs = z_i.size(0)
    if n_pairs <= PAIR_SCORE_CHUNK_SIZE:
        return model.compute_pair_scores(
            z_i, z_j, binary_logits_i, binary_logits_j, mode=mode
        )

    chunks = []
    for start in range(0, n_pairs, PAIR_SCORE_CHUNK_SIZE):
        end = min(start + PAIR_SCORE_CHUNK_SIZE, n_pairs)
        chunks.append(
            model.compute_pair_scores(
                z_i[start:end],
                z_j[start:end],
                binary_logits_i[start:end] if binary_logits_i is not None else None,
                binary_logits_j[start:end] if binary_logits_j is not None else None,
                mode=mode,
            )
        )
    return torch.cat(chunks, dim=0)


def _reactive_stop_aux_scores(
    action_scores: torch.Tensor, target_action_idx: int
) -> tuple[torch.Tensor | None, int | None]:
    """Build {wrong pairs + is_over} scores for reactive-state stop aux loss."""
    stop_idx = action_scores.size(0) - 1
    if target_action_idx == stop_idx or action_scores.numel() <= 1:
        return None, None

    aux_mask = torch.ones_like(action_scores, dtype=torch.bool)
    aux_mask[target_action_idx] = False
    aux_scores = action_scores[aux_mask]
    aux_target_idx = aux_scores.size(0) - 1  # stop stays last in action_scores
    return aux_scores, aux_target_idx


def _negative_reaction_absolute_loss(
    pair_scores: torch.Tensor, stop_score: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Push stop up and all pair actions down on yield-0 negative reactions."""
    loss_stop_up = F.softplus(-stop_score)
    if pair_scores.numel() > 0:
        loss_pairs_down = F.softplus(pair_scores).mean()
    else:
        loss_pairs_down = _zero(device)
    return loss_stop_up + loss_pairs_down


class PairRankingLoss(nn.Module):
    """Temperature-scaled softmax ranking loss over pair and stop actions.

    Per molecule, all upper-triangle bond-pair scores plus the stop logit form
    one action distribution; the annotated reactive pair (or stop) is the
    cross-entropy target. Auxiliary terms: per-bond binary reactivity BCE,
    bond-/swap-type classification, a stop-rank auxiliary loss, and an
    absolute push-down loss for yield-0 negative reactions.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        beta: float = 0.5,
        bond_type_weight: float = 0.5,
        binary_weight: float = 1.0,
        hard_negative_k: int = 0,
        virtual_pos_weight: float = 1.0,
        pair_scoring_mode: str = "pair_only",
        stop_rank_aux_weight: float = 0.0,
        negative_reaction_weight: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.bond_type_weight = bond_type_weight
        self.binary_weight = binary_weight
        self.hard_negative_k = hard_negative_k
        self.pair_scoring_mode = pair_scoring_mode
        self.stop_rank_aux_weight = stop_rank_aux_weight
        self.negative_reaction_weight = negative_reaction_weight
        self.bond_type_ce = nn.CrossEntropyLoss()
        self.binary_ce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        out_dict: dict,
        targets: torch.Tensor,
        bond_batch_vec: torch.Tensor,
        is_over_targets: torch.Tensor,
        is_negative_reaction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined ranking loss for one batch.

        Args:
            out_dict: Model outputs; must contain ``embeddings``,
                ``logits_binary``, ``logits_stop`` and ``model_ref``.
            targets: Per-bond integer labels; > 0 marks the reactive pair.
            bond_batch_vec: Molecule index for every bond token.
            is_over_targets: Per-molecule stop labels (1 = stop state).
            is_negative_reaction: Optional per-molecule yield-0 negative flags.

        Returns:
            Scalar total loss.
        """
        device = targets.device
        z = out_dict["embeddings"]
        binary_logits = out_dict["logits_binary"]
        stop_logits = out_dict["logits_stop"].squeeze(-1)
        model = out_dict["model_ref"]

        # Binary auxiliary loss for per-bond reactivity.
        loss_binary = self.binary_ce(
            binary_logits,
            _binary_reactivity_targets(
                targets, bond_batch_vec, is_over_targets, is_negative_reaction
            ),
        )

        # Ranking and bond-type losses over real bond pairs and the stop action.
        pair_ranking_losses: list[torch.Tensor] = []
        stop_rank_aux_losses: list[torch.Tensor] = []
        negative_reaction_losses: list[torch.Tensor] = []
        bond_type_losses: list[torch.Tensor] = []
        unique_mols = torch.unique(bond_batch_vec)

        for mol_idx_in_batch, mol_id in enumerate(unique_mols):
            mol_mask = bond_batch_vec == mol_id
            z_mol = z[mol_mask]
            y_mol = targets[mol_mask]
            stop_score = stop_logits[mol_idx_in_batch]

            pair_scores, triu_indices = model.compute_pair_scores_upper_triangle(
                z_mol,
                binary_logits=binary_logits[mol_mask],
                mode=self.pair_scoring_mode,
            )
            all_action_scores = torch.cat(
                [pair_scores, stop_score.unsqueeze(0)], dim=0
            )
            stop_action_idx = all_action_scores.size(0) - 1

            reactive_idx = torch.where(y_mol > 0)[0]
            r1: torch.Tensor | None = None
            r2: torch.Tensor | None = None
            target_action_idx = stop_action_idx
            if reactive_idx.numel() == 2:
                r1, r2 = _ordered_reactive_pair(reactive_idx)
                target_action_idx = _true_pair_action_index(
                    triu_indices, r1, r2, stop_action_idx
                )

            is_negative_sample = bool(
                is_negative_reaction is not None
                and is_negative_reaction[mol_idx_in_batch].item()
            )
            is_negative_mech = (
                is_negative_sample and target_action_idx != stop_action_idx
            )

            if all_action_scores.numel() > 1:
                selected, selected_target_idx = _select_hard_negatives(
                    all_action_scores, target_action_idx, self.hard_negative_k
                )
                if is_negative_mech and self.negative_reaction_weight > 0:
                    negative_reaction_losses.append(
                        F.softplus(all_action_scores[target_action_idx])
                    )
                else:
                    pair_ranking_losses.append(
                        F.cross_entropy(
                            (selected / self.temperature).unsqueeze(0),
                            torch.tensor(
                                [selected_target_idx], dtype=torch.long, device=device
                            ),
                        )
                    )

                if (
                    self.stop_rank_aux_weight > 0
                    and reactive_idx.numel() == 2
                    and not is_negative_mech
                ):
                    aux_scores, aux_target_idx = _reactive_stop_aux_scores(
                        all_action_scores, target_action_idx
                    )
                    if aux_scores is not None and aux_scores.numel() > 1:
                        stop_rank_aux_losses.append(
                            F.cross_entropy(
                                (aux_scores / self.temperature).unsqueeze(0),
                                torch.tensor(
                                    [aux_target_idx], dtype=torch.long, device=device
                                ),
                            )
                        )

            if (
                self.negative_reaction_weight > 0
                and is_negative_sample
                and not is_negative_mech
            ):
                negative_reaction_losses.append(
                    _negative_reaction_absolute_loss(pair_scores, stop_score, device)
                )

            bt_loss = _per_mol_bond_type_loss(
                self.bond_type_ce,
                out_dict,
                bond_batch_vec,
                mol_id,
                mol_idx_in_batch,
                r1,
                r2,
                z_mol,
                y_mol,
                is_negative_mech,
                device,
            )
            if bt_loss is not None:
                bond_type_losses.append(bt_loss)

        return (
            _stack_mean(pair_ranking_losses, device)
            + (self.binary_weight * loss_binary)
            + (self.bond_type_weight * _stack_mean(bond_type_losses, device))
            + (self.stop_rank_aux_weight * _stack_mean(stop_rank_aux_losses, device))
            + (
                self.negative_reaction_weight
                * _stack_mean(negative_reaction_losses, device)
            )
        )


class EBMPairRankingLoss(nn.Module):
    """
    Energy-Based Model (EBM) loss for bond pair prediction.

    The pair scorer outputs a score S (where S = -E in energy terms).  The
    EBM contrastive term is:

        L_ce = -S_plus + logsumexp(S_all)

    which is equivalent to the standard energy form E_plus + logsumexp(-E_all)
    but keeps the convention that higher score = more reactive, so existing
    evaluation code (picks highest-scoring pair) works without changes.

    The key advantage over PairRankingLoss is the anchor regularisation:

        L_anchor = lambda * mean(S^2)

    This keeps scores centred near zero, preventing scale drift and giving
    meaningful absolute probabilities via softmax(S) or exp(S).  PairRankingLoss
    divides by temperature which only preserves relative ranking; the anchor
    term preserves absolute calibration.
    """

    def __init__(
        self,
        beta: float = 0.5,
        bond_type_weight: float = 0.5,
        binary_weight: float = 1.0,
        lambda_anchor: float = 0.05,
        hard_negative_k: int = 0,
        virtual_pos_weight: float = 1.0,
        train_no_reaction_class: bool = False,
        negative_reaction_weight: float = 0.0,
        pair_scoring_mode: str = "pair_only",
        stop_rank_aux_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.bond_type_weight = bond_type_weight
        self.binary_weight = binary_weight
        self.lambda_anchor = lambda_anchor
        self.hard_negative_k = hard_negative_k
        self.train_no_reaction_class = train_no_reaction_class
        self.pair_scoring_mode = pair_scoring_mode
        self.virtual_pos_weight = virtual_pos_weight
        self.stop_rank_aux_weight = stop_rank_aux_weight
        self.negative_reaction_weight = negative_reaction_weight
        self.bond_type_ce = nn.CrossEntropyLoss()
        self.binary_ce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        out_dict: dict,
        targets: torch.Tensor,
        bond_batch_vec: torch.Tensor,
        is_over_targets: torch.Tensor,
        is_negative_reaction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the EBM loss, dispatching to the vectorized fast path when eligible.

        Args:
            out_dict: Model outputs; must contain ``embeddings``,
                ``logits_binary``, ``logits_stop`` and ``model_ref``.
            targets: Per-bond integer labels; > 0 marks the reactive pair.
            bond_batch_vec: Molecule index for every bond token.
            is_over_targets: Per-molecule stop labels (1 = stop state).
            is_negative_reaction: Optional per-molecule yield-0 negative flags.

        Returns:
            Scalar total loss.
        """
        if self._can_use_fast_path(out_dict, is_negative_reaction):
            return self._forward_fast(
                out_dict, targets, bond_batch_vec, is_over_targets
            )
        return self._forward_loop(
            out_dict, targets, bond_batch_vec, is_over_targets, is_negative_reaction
        )

    def _can_use_fast_path(
        self, out_dict: dict, is_negative_reaction: torch.Tensor | None
    ) -> bool:
        """Eligibility for the vectorized path.

        Requires the AttentionPairScorer backend (which exposes
        ``score_full_batched``), no hard-negative subsampling, no negative
        reaction loss, and a model that exposes the per-batch scorer/binary
        head used by the batched fused mode.
        """
        if self.hard_negative_k != 0:
            return False
        if self.negative_reaction_weight != 0:
            return False
        if self.pair_scoring_mode not in ("pair_only", "fused"):
            return False
        # is_negative_reaction is consulted only in slow-path branches; if all
        # samples are non-negative we can still run the fast path.
        if is_negative_reaction is not None and bool(is_negative_reaction.any()):
            return False
        model = out_dict.get("model_ref")
        if model is None:
            return False
        if not isinstance(getattr(model, "pair_scorer", None), AttentionPairScorer):
            return False
        backend = getattr(model, "_pair_score_backend", "legacy")
        if backend == "legacy":
            return False
        return True

    def _forward_fast(
        self,
        out_dict: dict,
        targets: torch.Tensor,
        bond_batch_vec: torch.Tensor,
        is_over_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized loss path — see ``_forward_loop`` for the reference impl.

        Pads bonds per molecule, computes the full pair score matrix in one
        scorer call, then evaluates EBM, anchor, stop-rank-aux and bond-type
        losses without any per-molecule Python loop.
        """
        device = targets.device
        z = out_dict["embeddings"]
        binary_logits = out_dict["logits_binary"]
        stop_logits = out_dict["logits_stop"].squeeze(-1)
        model = out_dict["model_ref"]

        # Binary auxiliary loss over all bond tokens.
        loss_binary = self.binary_ce(
            binary_logits,
            _binary_reactivity_targets(targets, bond_batch_vec, is_over_targets, None),
        )

        # Pad bond-level tensors to [B_mol, max_n, ...].
        z_pad, mask_pad = to_dense_batch(z, bond_batch_vec)  # [B,M,D], [B,M]
        y_pad, _ = to_dense_batch(targets, bond_batch_vec)  # [B,M]
        binary_pad, _ = to_dense_batch(
            binary_logits.squeeze(-1), bond_batch_vec
        )  # [B,M]
        B, M, _ = z_pad.shape

        if M < 2:
            # No molecule contains a valid bond pair, so the ranking term is zero.
            return self.binary_weight * loss_binary + _zero(device)

        score_full = model.pair_scorer.score_full_batched(z_pad)  # [B,M,M]
        if self.pair_scoring_mode in ("fused", "learned_fusion"):
            react = torch.sigmoid(binary_pad)  # [B,M]
            if self.pair_scoring_mode == "fused":
                score_full = score_full + react.unsqueeze(2) + react.unsqueeze(1)
            else:  # learned_fusion
                w = torch.softmax(model.fusion_weights, dim=0) * 3
                score_full = (
                    w[0] * score_full
                    + w[1] * react.unsqueeze(2)
                    + w[2] * react.unsqueeze(1)
                )

        triu = torch.triu_indices(M, M, offset=1, device=device)  # [2,P]
        P = triu.size(1)
        pair_scores = score_full[:, triu[0], triu[1]]  # [B,P]
        pair_valid = mask_pad[:, triu[0]] & mask_pad[:, triu[1]]  # [B,P]

        # Action scores contain pair scores followed by the stop score.
        action_scores = torch.cat(
            [pair_scores, stop_logits.unsqueeze(-1)], dim=-1
        )  # [B,P+1]
        action_mask = torch.cat(
            [
                pair_valid,
                torch.ones((B, 1), dtype=torch.bool, device=device),
            ],
            dim=-1,
        )  # [B,P+1]
        stop_idx_const = P  # last column

        # Per-molecule target index; valid samples contain zero or two reactive bonds.
        react_mask = (y_pad > 0) & mask_pad  # [B,M]
        react_count = react_mask.sum(dim=1)  # [B]
        has_reactive = react_count == 2

        # First reactive position per molecule, defined only for reactive rows.
        r1 = react_mask.float().argmax(dim=1)  # [B]
        # Second reactive position after excluding the first.
        react_mask_no_r1 = react_mask.clone()
        react_mask_no_r1.scatter_(1, r1.unsqueeze(1), False)
        r2 = react_mask_no_r1.float().argmax(dim=1)  # [B]
        target_pair_idx = (
            r1 * (M - 1) - r1 * (r1 - 1) // 2 + (r2 - r1 - 1)
        )  # [B], valid only for has_reactive
        target_idx = torch.where(
            has_reactive,
            target_pair_idx,
            torch.full_like(r1, stop_idx_const),
        )

        # EBM and anchor losses.
        action_count = action_mask.sum(dim=-1)  # >=1 always (stop present)
        include_ranking = action_count > 1  # need at least 1 pair to have a
        # contrastive set; mols with M=1 still contribute only the stop and
        # are excluded, matching the loop's ``action_scores.numel() > 1``.

        masked_action = action_scores.masked_fill(~action_mask, float("-inf"))
        lse = torch.logsumexp(masked_action, dim=-1)  # [B]
        S_plus = action_scores.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)
        ebm_per_mol = -S_plus + lse
        if self.virtual_pos_weight > 1.0:
            is_stop_target = ~has_reactive
            ebm_per_mol = torch.where(
                is_stop_target,
                ebm_per_mol * self.virtual_pos_weight,
                ebm_per_mol,
            )
        loss_ebm_pos = self._masked_mean(ebm_per_mol, include_ranking, device)

        # Anchor: per-mol mean of valid action_scores^2, then mean across mols.
        sq = action_scores * action_scores
        sq = sq * action_mask.to(sq.dtype)
        per_mol_anchor = sq.sum(-1) / action_count.to(sq.dtype).clamp(min=1)
        loss_anchor = self._masked_mean(per_mol_anchor, include_ranking, device)

        # Stop-rank auxiliary loss for reactive molecules with a pair target.
        if self.stop_rank_aux_weight > 0:
            aux_action_mask = action_mask.clone()
            # Drop the target position from the aux contrastive set.
            aux_action_mask.scatter_(1, target_idx.unsqueeze(1), False)
            aux_count = aux_action_mask.sum(dim=-1)
            include_aux = has_reactive & (aux_count > 1)
            masked_aux = action_scores.masked_fill(
                ~aux_action_mask, float("-inf")
            )
            aux_lse = torch.logsumexp(masked_aux, dim=-1)
            s_stop = action_scores[:, stop_idx_const]
            aux_per_mol = -s_stop + aux_lse
            loss_stop_rank_aux = self._masked_mean(
                aux_per_mol, include_aux, device
            )
        else:
            loss_stop_rank_aux = _zero(device)

        # Bond-type loss batched across reactive molecules.
        loss_bond_type = self._batched_bond_type_loss(
            out_dict,
            model,
            bond_batch_vec,
            z,
            y_pad,
            has_reactive,
            r1,
            r2,
            mask_pad,
            device,
        )

        total = (
            loss_ebm_pos
            + self.binary_weight * loss_binary
            + self.bond_type_weight * loss_bond_type
            + self.stop_rank_aux_weight * loss_stop_rank_aux
            + self.lambda_anchor * loss_anchor
        )
        return total

    @staticmethod
    def _masked_mean(
        values: torch.Tensor, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Mean of ``values`` where ``mask`` is True, or scalar zero if empty."""
        if mask.any():
            return values[mask].mean()
        return _zero(device)

    def _batched_bond_type_loss(
        self,
        out_dict: dict,
        model: nn.Module,
        bond_batch_vec: torch.Tensor,
        z_flat: torch.Tensor,
        y_pad: torch.Tensor,
        has_reactive: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        mask_pad: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Batched swap classifier / bond-type-logits loss across reactive mols."""
        bond_type_logits = out_dict.get("logits_bond_type", None)
        use_swap = (
            getattr(model, "use_pair_bond_type", False)
            and hasattr(model, "pair_bond_type_classifier")
        )
        if not use_swap and bond_type_logits is None:
            return _zero(device)
        if not has_reactive.any():
            return _zero(device)

        react_idx = torch.where(has_reactive)[0]  # [B_react]
        r1_sel = r1[react_idx]
        r2_sel = r2[react_idx]
        # y_pad is padded per-mol; gather targets from the r1 column.
        bt_targets_raw = y_pad[react_idx, r1_sel].long()

        # Per-mol z lookup: z_flat positions of bond r1/r2 within each mol.
        # `to_dense_batch` orders bonds by batch index then by their order in
        # ``z_flat``; recover the global flat offset of each mol's first bond
        # via cumulative counts on the flat ``bond_batch_vec``.
        unique_mols, mol_counts = torch.unique_consecutive(
            bond_batch_vec, return_counts=True
        )
        # ``unique_mols`` matches ``to_dense_batch`` row order because both rely
        # on sorted unique batch values.
        offsets = torch.zeros_like(mol_counts)
        offsets[1:] = mol_counts.cumsum(0)[:-1]
        # offsets[b] is the flat index of mol b's first bond.
        global_r1 = offsets[react_idx] + r1_sel
        global_r2 = offsets[react_idx] + r2_sel
        z_r1 = z_flat[global_r1]  # [B_react, D]
        z_r2 = z_flat[global_r2]

        if use_swap:
            classifier = model.pair_bond_type_classifier

            if isinstance(classifier, MLPSwapTypeHead):
                pair_bt_logits = classifier(z_r1, z_r2)
            else:
                # Heads that need node embeddings.
                h_nodes = out_dict.get("node_embeddings")
                bi = out_dict.get("bond_indices_out")
                assert h_nodes is not None and bi is not None, (
                    "use_pair_bond_type=True requires 'node_embeddings' and"
                    " 'bond_indices_out' in model output."
                )
                bond_idx_r1 = bi[global_r1]  # [B_react, 2]
                bond_idx_r2 = bi[global_r2]
                h_u1 = h_nodes[bond_idx_r1[:, 0]]
                h_u2 = h_nodes[bond_idx_r1[:, 1]]
                h_v1 = h_nodes[bond_idx_r2[:, 0]]
                h_v2 = h_nodes[bond_idx_r2[:, 1]]
                if isinstance(classifier, BondSplitSwapHead):
                    pair_bt_logits = classifier(z_r1, z_r2, h_u1, h_u2, h_v1, h_v2)
                elif isinstance(classifier, (AtomSwapTypeHead, NodeAttentionSwapHead)):
                    pair_bt_logits = classifier(h_u1, h_u2, h_v1, h_v2)
                else:
                    raise TypeError(
                        f"Unknown swap type classifier: {type(classifier)}"
                    )

            if isinstance(classifier, NodeAttentionSwapHead):
                bt_targets = bt_targets_raw - 1
            else:
                bt_targets = bt_targets_raw
            return self.bond_type_ce(pair_bt_logits, bt_targets)

        # Per-mol bond_type_logits path (model output has logits_bond_type).
        bt_logits = bond_type_logits[react_idx]
        return self.bond_type_ce(bt_logits, bt_targets_raw)

    def _precompute_pair_scores(
        self,
        out_dict: dict,
        targets: torch.Tensor,
        bond_batch_vec: torch.Tensor,
    ) -> tuple[list[_MolPairInfo], list[torch.Tensor], torch.Tensor | None]:
        """Batch all pair-score computation ahead of loss accumulation.

        The pair scoring is batched across all molecules in one GPU call to
        avoid per-molecule kernel launch overhead. With an exact pair-score
        backend, scores are computed per molecule via
        ``compute_pair_scores_upper_triangle``; otherwise all upper-triangle
        embedding pairs are concatenated and scored in one chunked call.

        Returns:
            Tuple of (per-molecule info, per-molecule scores for the exact
            backend, concatenated scores for the legacy backend or None).
        """
        device = targets.device
        z = out_dict["embeddings"]
        binary_logits = out_dict["logits_binary"]
        model = out_dict["model_ref"]
        use_exact_pair_backend = (
            getattr(model, "_pair_score_backend", "legacy") != "legacy"
        )
        use_binary = self.pair_scoring_mode != "pair_only"

        mol_pair_info: list[_MolPairInfo] = []
        all_scores_per_mol: list[torch.Tensor] = []
        all_z_i: list[torch.Tensor] = []
        all_z_j: list[torch.Tensor] = []
        all_bl_i: list[torch.Tensor] = []
        all_bl_j: list[torch.Tensor] = []

        for mol_id in torch.unique(bond_batch_vec):
            mol_mask = bond_batch_vec == mol_id
            z_mol = z[mol_mask]
            y_mol = targets[mol_mask]
            n_bonds = z_mol.size(0)
            reactive_idx = torch.where(y_mol > 0)[0]
            has_reactive = reactive_idx.numel() == 2 and n_bonds >= 2

            if n_bonds < 2:
                if use_exact_pair_backend:
                    all_scores_per_mol.append(torch.empty(0, device=device))
                mol_pair_info.append(
                    _MolPairInfo(mol_id, mol_mask, n_bonds, 0, None, y_mol, None)
                )
                continue

            if use_exact_pair_backend:
                scores, triu_idx = model.compute_pair_scores_upper_triangle(
                    z_mol,
                    binary_logits=binary_logits[mol_mask] if use_binary else None,
                    mode=self.pair_scoring_mode,
                )
                all_scores_per_mol.append(scores)
            else:
                triu_idx = torch.triu_indices(
                    n_bonds, n_bonds, offset=1, device=device
                )
                all_z_i.append(z_mol[triu_idx[0]])
                all_z_j.append(z_mol[triu_idx[1]])
                if use_binary:
                    binary_mol = binary_logits[mol_mask]
                    all_bl_i.append(binary_mol[triu_idx[0]])
                    all_bl_j.append(binary_mol[triu_idx[1]])

            mol_pair_info.append(
                _MolPairInfo(
                    mol_id,
                    mol_mask,
                    n_bonds,
                    triu_idx.size(1),
                    reactive_idx if has_reactive else None,
                    y_mol,
                    triu_idx,
                )
            )

        all_scores = None
        if not use_exact_pair_backend and all_z_i:
            all_scores = _compute_pair_scores_chunked(
                model,
                torch.cat(all_z_i, dim=0),
                torch.cat(all_z_j, dim=0),
                torch.cat(all_bl_i, dim=0) if use_binary else None,
                torch.cat(all_bl_j, dim=0) if use_binary else None,
                mode=self.pair_scoring_mode,
            )

        return mol_pair_info, all_scores_per_mol, all_scores

    def _forward_loop(
        self,
        out_dict: dict,
        targets: torch.Tensor,
        bond_batch_vec: torch.Tensor,
        is_over_targets: torch.Tensor,
        is_negative_reaction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference per-molecule loss path.

        Pair scores are precomputed in batched GPU calls; the per-molecule
        loss accumulation (indexing + logsumexp) stays as a Python loop
        because it is bookkeeping rather than GPU-bound computation.
        """
        device = targets.device
        z = out_dict["embeddings"]
        binary_logits = out_dict["logits_binary"]
        stop_logits = out_dict["logits_stop"].squeeze(-1)
        model = out_dict["model_ref"]
        use_exact_pair_backend = (
            getattr(model, "_pair_score_backend", "legacy") != "legacy"
        )

        loss_binary = self.binary_ce(
            binary_logits,
            _binary_reactivity_targets(
                targets, bond_batch_vec, is_over_targets, is_negative_reaction
            ),
        )

        mol_pair_info, all_scores_per_mol, all_scores = self._precompute_pair_scores(
            out_dict, targets, bond_batch_vec
        )

        # Accumulate per-molecule losses using precomputed pair scores.
        ebm_losses: list[torch.Tensor] = []
        stop_rank_aux_losses: list[torch.Tensor] = []
        negative_reaction_losses: list[torch.Tensor] = []
        anchor_losses: list[torch.Tensor] = []
        bond_type_losses: list[torch.Tensor] = []
        score_offset = 0

        for mol_idx_in_batch, info in enumerate(mol_pair_info):
            if use_exact_pair_backend:
                scores = all_scores_per_mol[mol_idx_in_batch]
            else:
                scores = (
                    all_scores[score_offset : score_offset + info.n_pairs]
                    if info.n_pairs > 0
                    else torch.empty(0, device=device)
                )
            score_offset += info.n_pairs

            action_scores = torch.cat(
                [scores, stop_logits[mol_idx_in_batch].unsqueeze(0)], dim=0
            )
            stop_action_idx = action_scores.size(0) - 1
            r1: torch.Tensor | None = None
            r2: torch.Tensor | None = None
            target_action_idx = stop_action_idx

            if info.reactive_idx is not None:
                r1, r2 = _ordered_reactive_pair(info.reactive_idx)
                target_action_idx = _true_pair_action_index(
                    info.triu_idx, r1, r2, stop_action_idx
                )

            is_negative_sample = bool(
                is_negative_reaction is not None
                and is_negative_reaction[mol_idx_in_batch].item()
            )
            is_negative_mech = is_negative_sample and info.reactive_idx is not None

            if action_scores.numel() > 1:
                S_plus = action_scores[target_action_idx]
                selected, _ = _select_hard_negatives(
                    action_scores, target_action_idx, self.hard_negative_k
                )

                anchor_losses.append(torch.mean(action_scores**2))
                if is_negative_mech and self.negative_reaction_weight > 0:
                    negative_reaction_losses.append(
                        F.softplus(action_scores[target_action_idx])
                    )
                else:
                    ebm_loss = -S_plus + torch.logsumexp(selected, dim=0)
                    is_stop_target = info.reactive_idx is None
                    if is_stop_target and self.virtual_pos_weight > 1.0:
                        ebm_loss = ebm_loss * self.virtual_pos_weight
                    ebm_losses.append(ebm_loss)

                if (
                    self.stop_rank_aux_weight > 0
                    and info.reactive_idx is not None
                    and not is_negative_mech
                ):
                    aux_scores, aux_target_idx = _reactive_stop_aux_scores(
                        action_scores, target_action_idx
                    )
                    if aux_scores is not None and aux_scores.numel() > 1:
                        s_stop = aux_scores[aux_target_idx]
                        stop_rank_aux_losses.append(
                            -s_stop + torch.logsumexp(aux_scores, dim=0)
                        )

                if (
                    self.negative_reaction_weight > 0
                    and is_negative_sample
                    and not is_negative_mech
                ):
                    negative_reaction_losses.append(
                        _negative_reaction_absolute_loss(
                            scores, stop_logits[mol_idx_in_batch], device
                        )
                    )

            bt_loss = _per_mol_bond_type_loss(
                self.bond_type_ce,
                out_dict,
                bond_batch_vec,
                info.mol_id,
                mol_idx_in_batch,
                r1,
                r2,
                z[info.mol_mask],
                info.y_mol,
                is_negative_mech,
                device,
            )
            if bt_loss is not None:
                bond_type_losses.append(bt_loss)

        return (
            _stack_mean(ebm_losses, device)
            + self.binary_weight * loss_binary
            + self.bond_type_weight * _stack_mean(bond_type_losses, device)
            + self.stop_rank_aux_weight * _stack_mean(stop_rank_aux_losses, device)
            + (
                self.negative_reaction_weight
                * _stack_mean(negative_reaction_losses, device)
            )
            + self.lambda_anchor * _stack_mean(anchor_losses, device)
        )


def get_ranking_loss(loss_type: str = "softmax", **kwargs) -> nn.Module:
    """Factory function to get ranking loss by name."""
    losses = {
        "softmax": PairRankingLoss,
        "pair": PairRankingLoss,
        "ebm": EBMPairRankingLoss,
    }

    if loss_type not in losses:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from {list(losses.keys())}"
        )

    return losses[loss_type](**kwargs)
