"""Accumulation of per-batch evaluation results into summary metrics."""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from bond_match.training.constants import (
    TOP_K_VALUES,
)
from bond_match.training.state import (
    EvalMetrics,
)

class EvalCollector:
    """Accumulates per-batch evaluation results into final metrics."""

    def __init__(self):
        """Initialize empty accumulators for predictions, ranks, similarities, and losses."""
        self.all_preds: list[float] = []
        self.all_labels: list[float] = []
        self.reciprocal_ranks: list[float] = []
        self.top_hits: dict[int, int] = {k: 0 for k in TOP_K_VALUES}
        self.total_reactive: int = 0
        self.positive_sims: list[float] = []
        self.negative_sims: list[float] = []
        self.bt_preds: list[int] = []
        self.bt_labels: list[int] = []
        self.stop_correct: int = 0
        self.stop_total: int = 0
        self.loss_sum: float = 0.0
        self.loss_count: int = 0
        self.nonfinite_sim_count: int = 0
        self.nonfinite_loss_count: int = 0

    def add_preds(self, preds: list, labels: list):
        """Append per-bond class predictions and ground-truth labels."""
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)

    def add_ranking_hit(self, rank: int, rr: float):
        """Record the reciprocal rank and increment top-k hit counters.

        Args:
            rank: Zero-based rank of the true pair; counts as a top-k hit
                when ``rank < k``.
            rr: Reciprocal rank value to accumulate for MRR.
        """
        self.reciprocal_ranks.append(rr)
        for k in TOP_K_VALUES:
            if rank < k:
                self.top_hits[k] += 1

    def add_bond_type(self, pred: int, label: int):
        """Record a predicted and true bond-type class."""
        self.bt_preds.append(pred)
        self.bt_labels.append(label)

    def add_sim(self, positive: float, negatives: list[float]):
        """Record similarity scores for a positive pair and its negatives.

        Non-finite values are skipped and counted in ``nonfinite_sim_count``.
        """
        if np.isfinite(positive):
            self.positive_sims.append(positive)
        else:
            self.nonfinite_sim_count += 1
        finite_negatives = [neg for neg in negatives if np.isfinite(neg)]
        self.nonfinite_sim_count += len(negatives) - len(finite_negatives)
        self.negative_sims.extend(finite_negatives)

    def add_stop(self, correct: int, total: int):
        """Accumulate stop-action correctness counts."""
        self.stop_correct += correct
        self.stop_total += total

    def add_virtual(self, correct: int, total: int):
        """Alias for ``add_stop``."""
        self.add_stop(correct, total)

    def add_loss(self, loss: float):
        """Accumulate a batch loss; non-finite values are skipped and counted."""
        if not np.isfinite(loss):
            self.nonfinite_loss_count += 1
            return
        self.loss_sum += loss
        self.loss_count += 1

    def compute_metrics(self) -> EvalMetrics:
        """Compute summary metrics from the accumulated results and print a report.

        Reports bond classification, ranking metrics (MRR, top-k accuracy),
        similarity separation (Cohen's d), bond-type accuracy, stop-action
        accuracy, and average validation loss, each only when the corresponding
        data was collected.

        Returns:
            EvalMetrics: Populated metrics container.
        """
        preds_arr = np.array(self.all_preds)
        labels_arr = np.array(self.all_labels)

        print("\n" + "=" * 60)
        print("RANKING-BASED EVALUATION RESULTS")
        print("=" * 60)

        print("\n--- Bond Classification Report ---")
        print(
            classification_report(
                labels_arr,
                preds_arr,
                labels=[0, 1, 2],
                target_names=["Stable", "Class1", "Class2"],
                zero_division=0,
            )
        )
        print("Confusion Matrix:")
        print(confusion_matrix(labels_arr, preds_arr, labels=[0, 1, 2]))

        metrics = EvalMetrics()

        if self.total_reactive > 0:
            metrics.mrr = (
                float(np.mean(self.reciprocal_ranks)) if self.reciprocal_ranks else 0.0
            )
            metrics.top1 = self.top_hits[1] / self.total_reactive
            metrics.top3 = self.top_hits[3] / self.total_reactive
            metrics.top5 = self.top_hits[5] / self.total_reactive
            metrics.top10 = self.top_hits[10] / self.total_reactive

            print("\n--- Ranking Metrics (Pair Retrieval) ---")
            print(f"Total Reactive Molecules: {self.total_reactive}")
            print(f"Mean Reciprocal Rank (MRR): {metrics.mrr:.4f}")
            for k in TOP_K_VALUES:
                acc = self.top_hits[k] / self.total_reactive
                print(
                    f"Top-{k:<2d} Accuracy: {acc:.2%} ({self.top_hits[k]}/{self.total_reactive})"
                )

        if self.positive_sims and self.negative_sims:
            pos_mean = np.mean(self.positive_sims)
            pos_std = np.std(self.positive_sims)
            neg_mean = np.mean(self.negative_sims)
            neg_std = np.std(self.negative_sims)
            metrics.sim_separation = float(
                (pos_mean - neg_mean) / (pos_std + neg_std + 1e-8)
            )

            print("\n--- Similarity Distribution ---")
            print(f"Positive pairs: {pos_mean:.3f} +/- {pos_std:.3f}")
            print(f"Negative pairs: {neg_mean:.3f} +/- {neg_std:.3f}")
            print(f"Separation (Cohen's d): {metrics.sim_separation:.3f}")

        if self.nonfinite_sim_count > 0:
            print(
                f"Skipped {self.nonfinite_sim_count} non-finite similarity scores"
            )

        if self.bt_preds and self.bt_labels:
            bt_p = np.array(self.bt_preds)
            bt_l = np.array(self.bt_labels)
            metrics.bond_type_acc = float((bt_p == bt_l).mean())

            print("\n--- Bond Type Classification ---")
            print(f"Accuracy: {metrics.bond_type_acc:.2%}")
            print(f"Distribution - Labels: {np.bincount(bt_l, minlength=3)}")
            print(f"Distribution - Preds:  {np.bincount(bt_p, minlength=3)}")

        if self.stop_total > 0:
            metrics.stop_acc = self.stop_correct / self.stop_total
            print(f"\n--- Stop Action Accuracy ---")
            print(
                f"Accuracy: {metrics.stop_acc:.2%} ({self.stop_correct}/{self.stop_total})"
            )

        if self.loss_count > 0:
            metrics.val_loss = self.loss_sum / self.loss_count
            print(f"\n--- Validation Loss ---")
            print(f"Avg Loss: {metrics.val_loss:.4f}")
            if self.nonfinite_loss_count > 0:
                print(f"Skipped {self.nonfinite_loss_count} non-finite loss batches")
        elif self.nonfinite_loss_count > 0:
            print(f"\n--- Validation Loss ---")
            print("Avg Loss: unavailable (no finite loss batches)")
            print(f"Skipped {self.nonfinite_loss_count} non-finite loss batches")

        print("=" * 60 + "\n")
        return metrics

