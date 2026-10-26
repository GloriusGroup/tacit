"""Score high-throughput experimentation (HTE) wells with TACIT.

Each well is a full reaction SMILES ``reactants>>product`` that lists every species in the well (substrates,
reagents, additives and solvent) and the expected product. The score of a well is 1 - P(stop) for the first step,
i.e. the model's probability that a reaction takes place. A well is positive if its label (conversion or yield in %)
is at least ``--threshold``. Wells the model cannot encode are ranked lowest and count as incorrect.

    python src/bond_match/score_hte.py --config runs/tacit/config.yaml --checkpoint tacit.pt \\
        --wells wells.csv --label-column conversion_pct --out scores.csv

``wells.csv`` needs a ``reaction_smiles`` column and the label column. The HTE datasets of the paper are not
redistributed here; see the paper's SI for their sources and preprocessing.
"""

import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from bond_match.evaluation import Evaluator, load_eval_model
from bond_match.evaluation.utils import canonicalize_smiles, seed_everything


def canonical_reaction(rxn: str) -> str:
    reactants, product = rxn.split(">>")
    return canonicalize_smiles(reactants) + ">>" + canonicalize_smiles(product)


def score_wells(config: str, checkpoint: str, wells: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Return ``wells`` with ``score`` (1 - P(stop) at the first step) and ``correct`` (top-1 product) columns."""
    seed_everything(42)
    model, cfg = load_eval_model(config, checkpoint, device)
    evaluator = Evaluator(model, cfg, device)
    with tempfile.TemporaryDirectory() as tmp:
        data, out = os.path.join(tmp, "wells.jsonl"), os.path.join(tmp, "pred.csv")
        with open(data, "w") as f:
            for rxn in wells.reaction_smiles:
                f.write(json.dumps({"reaction_smiles": canonical_reaction(rxn)}) + "\n")
        evaluator.evaluate_dataset(data, save_csv=out, all_entries=True)
        pred = pd.read_csv(out).set_index("entry_idx").reindex(range(len(wells)))
    wells = wells.reset_index(drop=True).copy()
    wells["score"] = 1 - pred.root_stop_prob.values
    wells["correct"] = pred.match.fillna(0).astype(bool).values
    return wells


def metrics(wells: pd.DataFrame, label_column: str, threshold: float) -> dict:
    """ROC-AUC, PR-AUC (average precision) and share of positive wells with the correct top-1 product."""
    y = wells[label_column].values >= threshold
    s = wells.score.values.astype(float)
    s = np.where(np.isfinite(s), s, np.nanmin(s) - 1)
    return {
        "wells": len(wells),
        "positive_rate": y.mean(),
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "correct_pct": 100 * wells.correct.values[y].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wells", required=True, help="CSV with reaction_smiles and the label column")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--threshold", type=float, default=20.0, help="positive if label >= threshold (default 20)")
    parser.add_argument("--out", help="write per-well scores to this CSV")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    wells = score_wells(args.config, args.checkpoint, pd.read_csv(args.wells), device)
    if args.out:
        wells.to_csv(args.out, index=False)
    for key, value in metrics(wells, args.label_column, args.threshold).items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
