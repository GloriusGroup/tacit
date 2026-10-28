"""Interactive command-line interface for evaluating BondGAT ranking models."""

import argparse
import logging
import os
import torch

from bond_match.evaluation.evaluator import (
    Evaluator,
)
from bond_match.evaluation.model_io import (
    load_eval_model,
)
from bond_match.evaluation.utils import (
    seed_everything,
)

def main():
    """Evaluate a model on a dataset, or run the interactive evaluation menu.

    Parses command-line arguments, loads the model and configuration and seeds
    all random number generators (seed taken from the ``RANKING_EVAL_SEED``
    environment variable, default 42). With ``--data``, the dataset is
    evaluated non-interactively. Otherwise a menu offers interactive SMILES
    prediction, test-set evaluation, and visual evaluation of a random
    test-set sample.
    """
    parser = argparse.ArgumentParser(description="Evaluate BondGAT Ranking Model (V2)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model.pt"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--debug_mode", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--save-csv", type=str, default=None, help="Save test set results to CSV"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Evaluate this JSONL file non-interactively and exit",
    )
    parser.add_argument(
        "--all-entries",
        action="store_true",
        help="With --data: evaluate every entry, without the blockspace filter",
    )
    parser.add_argument(
        "--enable-yield0-negatives",
        action="store_true",
        help="Treat yield-0 eval entries as reactant>>reactant targets",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug_mode else logging.WARNING)

    device = (
        torch.device("cpu")
        if args.cpu or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    print(f"Using Device: {device}")
    eval_seed = int(os.getenv("RANKING_EVAL_SEED", "42"))
    seed_everything(eval_seed)

    model, config = load_eval_model(args.config, args.checkpoint, device)

    evaluator = Evaluator(
        model,
        config,
        device,
        enable_yield0_negatives=args.enable_yield0_negatives,
        eval_seed=eval_seed,
    )

    if args.data:
        evaluator.evaluate_dataset(
            args.data, save_csv=args.save_csv, all_entries=args.all_entries
        )
        return

    test_path = config.test_path
    test_entries = None

    while True:
        print("\n=== Ranking Model Eval Menu ===")
        print("1. Interactive SMILES Prediction (Top 5)")
        print(f"2. Evaluate on Test Set ({os.path.basename(test_path)})")
        print("3. Random Sample from Test Set (Visual)")
        print("q. Quit")

        choice = input("Select option: ").strip().lower()

        if choice == "1":
            smi = input("\nEnter SMILES: ").strip()
            if smi:
                try:
                    if ">>" in smi:
                        entry = {"reaction_smiles": smi, "success": True}
                    else:
                        entry = {"reaction_smiles": smi + ">>", "success": True}
                    evaluator.evaluate_single(entry=entry)
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback

                    traceback.print_exc()

        elif choice == "2":
            success_choice = (
                input("Include entries with success=False? (y/n) [Default: n]: ")
                .strip()
                .lower()
            )
            require_success = success_choice != "y"
            evaluator.evaluate_dataset(
                test_path, save_csv=args.save_csv, require_success=require_success
            )

        elif choice == "3":
            test_entries = evaluator.evaluate_random_sample(test_path, test_entries)

        elif choice == "q":
            break
        else:
            print("Invalid option.")

