"""Command-line entry point and public API for BondGAT ranking evaluation.

Provides the Evaluator class, evaluation utilities, and CLI entry point for
greedy and beam-search mechanism prediction.
"""

from bond_match.evaluation import *  # noqa: F401,F403
from bond_match.evaluation import Evaluator, seed_everything, main  # noqa: F401

from bond_match.blockspace import (  # noqa: F401
    decompose_molecules,
    matrices_to_smiles,
    mol_to_be_matrix,
)
from bond_match.config import load_config, PairSelectionMode  # noqa: F401
from bond_match.dataset import (  # noqa: F401
    _extract_entry_yield,
    _be_to_nx,
    augment_with_lp_edges,
    collate_fn,
    get_graph_features_nx,
    mol_to_nx,
    nx_to_mol,
    prepare_v3_reactant,
    trajectory_collate_fn,
)
from bond_match.modules.lora import checkpoint_has_lora_weights  # noqa: F401
from bond_match.model import BondGAT  # noqa: F401

if __name__ == "__main__":
    main()
