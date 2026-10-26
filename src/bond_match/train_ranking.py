"""Command-line entry point and public API for ranking-model training.

Provides the Trainer class, training utilities, and CLI entry point for DDP
and single-GPU bond-pair prediction training.
"""

from bond_match.training import *  # noqa: F401,F403
from bond_match.training import Trainer, main  # noqa: F401

from bond_match.config import RankingConfig, ResumeSchedulerMode, load_config  # noqa: F401
from bond_match.dataset import (  # noqa: F401
    ShardedReactionDataset,
    ShardAwareSampler,
    collate_fn,
    trajectory_collate_fn,
)
from bond_match.sampler import WeightedDistributedSampler  # noqa: F401
from bond_match.evaluate_ranking import Evaluator, seed_everything  # noqa: F401
from bond_match.losses.ranking_loss import (  # noqa: F401
    EBMPairRankingLoss,
    PairRankingLoss,
)
from bond_match.modules.lora import (  # noqa: F401
    checkpoint_has_lora_weights,
    collect_lora_stats,
    freeze_non_lora_parameters,
)
from bond_match.model import BondGAT  # noqa: F401

if __name__ == "__main__":
    main()
