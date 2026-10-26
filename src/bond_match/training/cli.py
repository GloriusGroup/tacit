"""Command-line entry point for training the BondGAT ranking model."""

import argparse
import os
import mlflow
from bond_match.config import load_config
from bond_match.evaluate_ranking import seed_everything

from bond_match.training.ddp import (
    cleanup_ddp,
    is_main_process,
    setup_ddp,
)
from bond_match.training.mlflow_utils import (
    _safe_mlflow_log,
)
from bond_match.training.trainer import (
    Trainer,
)

def main():
    """Parse command-line arguments, set up DDP and MLflow, and run training.

    Expects ``--config`` (path to a YAML config) and an optional ``--seed``.
    DDP is activated when the ``LOCAL_RANK`` environment variable is set.
    The active MLflow run is ended and the process group destroyed on exit.
    """
    parser = argparse.ArgumentParser(description="Train BondGAT Ranking Model (V2)")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to YAML config"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        return

    config = load_config(args.config)

    ddp_active = "LOCAL_RANK" in os.environ
    if ddp_active:
        setup_ddp()

    if not os.path.exists(config.train_path):
        print(f"Error: Data path '{config.train_path}' not found.")
        if ddp_active:
            cleanup_ddp()
        return

    try:
        trainer = Trainer(config)
        trainer._config_path = os.path.abspath(args.config)
        if (
            is_main_process()
            and trainer._mlflow_enabled
            and hasattr(trainer, "_mlflow_run")
        ):
            _safe_mlflow_log(
                mlflow.log_artifact, trainer._config_path, artifact_path="config"
            )
        trainer.train()
    finally:
        if is_main_process() and mlflow.active_run():
            mlflow.end_run()
        if ddp_active:
            cleanup_ddp()

