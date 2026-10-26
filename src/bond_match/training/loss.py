"""Construction of the ranking loss module from the training configuration."""

import torch.nn as nn
from bond_match.config import RankingConfig
from bond_match.losses.ranking_loss import (
    EBMPairRankingLoss,
    PairRankingLoss,
)


def build_loss(config: RankingConfig) -> nn.Module:
    """Instantiate the ranking loss selected by ``config.ranking_loss_type``.

    Args:
        config: Training configuration providing loss hyperparameters.

    Returns:
        nn.Module: ``PairRankingLoss`` for ``"softmax"`` or
        ``EBMPairRankingLoss`` for ``"ebm"``.

    Raises:
        ValueError: If ``config.ranking_loss_type`` is not recognized.
    """
    common = dict(
        beta=config.beta,
        bond_type_weight=config.bond_type_weight,
        hard_negative_k=config.hard_negative_k,
        virtual_pos_weight=config.virtual_pos_weight,
        negative_reaction_weight=config.negative_reaction_weight,
        pair_scoring_mode=config.pair_scoring_mode,
        stop_rank_aux_weight=config.stop_rank_aux_weight,
    )

    if config.ranking_loss_type == "softmax":
        return PairRankingLoss(temperature=config.temperature, **common)

    if config.ranking_loss_type == "ebm":
        return EBMPairRankingLoss(
            lambda_anchor=config.lambda_anchor,
            train_no_reaction_class=config.train_no_reaction_class,
            binary_weight=config.binary_weight,
            **common,
        )

    raise ValueError(f"Unknown ranking_loss_type: {config.ranking_loss_type}")
