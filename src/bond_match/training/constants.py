"""Shared training constants: the MLflow logging executor and top-k cutoffs."""

import concurrent.futures
from rdkit import RDLogger


"""Training loop for ranking-based bond-pair prediction.

A single Trainer class drives both DDP and single-GPU runs with shared
training/evaluation logic. Reuses seed_everything from evaluate_ranking.
"""
RDLogger.DisableLog("rdApp.*")
_mlflow_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
TOP_K_VALUES = (1, 3, 5, 10)
if __name__ == "__main__":
    main()
