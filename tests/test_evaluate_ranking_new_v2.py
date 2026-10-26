"""Tests for evaluate_ranking: yield-zero entry rewriting, yield metric
summaries, deterministic evaluation-data loading, and target independence of
beam selection in run_iteration."""
import json
import random
from types import SimpleNamespace

import numpy as np
import torch

import bond_match.evaluate_ranking as ev
from bond_match.evaluate_ranking import (
    BEState,
    Candidate,
    Evaluator,
    ForwardResult,
    InferenceDataset,
    IterationResult,
    PairResult,
    SampleMeta,
    StepResult,
    StopReason,
    _load_eval_data,
    _sample_one_jsonl_entry,
    _build_yield_metrics_summary,
    _is_no_reaction_target,
    _maybe_rewrite_yield0_entry,
)


class _DummyModel:
    """Minimal model stub exposing the attributes Evaluator reads."""

    use_pair_bond_type = False


class _DummyConfig:
    """Minimal evaluation config with the fields Evaluator reads."""

    ranking_loss_type = "ebm"
    pair_selection_mode = ev.PairSelectionMode.ARGMAX
    pair_selection_temperature = 1.0
    pair_softmax_topk = 0
    pair_scoring_mode = "fused"
    eval_beam_width = 2
    max_eval_steps = 3
    enable_history_context = False
    sample_mode = "trajectory"


def _dummy_state():
    """Return a minimal two-atom BEState."""
    return BEState(
        be_matrix=np.zeros((2, 2)),
        atoms_identity=np.array([0, 1]),
    )


def test_rewrite_yield0_entry_supports_lowercase_yield():
    """_maybe_rewrite_yield0_entry handles a lowercase 'yield' metadata key,
    rewriting the product to the reactant in all SMILES fields without
    mutating the input entry."""
    entry = {
        "reaction_smiles": "CCO>>CC=O",
        "reaction_smiles_mapped": "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        "metadata": {
            "yield": 0,
            "Reaction SMILES": "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        },
    }

    updated = _maybe_rewrite_yield0_entry(entry, enable_yield0_negatives=True)

    assert entry["reaction_smiles"] == "CCO>>CC=O"
    assert updated["reaction_smiles"] == "CCO>>CCO"
    assert (
        updated["reaction_smiles_mapped"]
        == "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    )
    assert (
        updated["metadata"]["Reaction SMILES"]
        == "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    )


def test_inference_dataset_rewrites_target_and_reads_lowercase_yield():
    """InferenceDataset reads a lowercase 'yield' key and rewrites the target
    to the reactant for yield-zero entries."""
    entry = {
        "reaction_smiles": "CCO>>CC=O",
        "metadata": {"yield": 0},
    }

    dataset = InferenceDataset([entry], enable_yield0_negatives=True)
    sample = dataset[0]

    assert sample is not None
    assert sample["yield_value"] == 0
    assert sample["target_smiles"] == "CCO"


def test_yield_metrics_summary_highlights_top1_when_enabled():
    """_build_yield_metrics_summary reports no-reaction, reaction, and macro
    top-1 accuracies plus no-reaction F1 when yield-zero negatives are
    enabled."""
    lines = _build_yield_metrics_summary(
        enable_yield0_negatives=True,
        yield_bucket_total={"yield_0": 4, "yield_gt_0": 6},
        yield_bucket_correct={"yield_0": 3, "yield_gt_0": 5},
        yield_zero_confusion={"tp": 3, "fp": 1, "fn": 1, "tn": 5},
    )

    assert any("No-Reaction Top-1: 0.7500 (3/4)" in line for line in lines)
    assert any("Reaction Top-1:    0.8333 (5/6)" in line for line in lines)
    assert any("Macro Top-1:       0.7917" in line for line in lines)
    assert any("No-Reaction F1:    0.7500" in line for line in lines)


def test_yield_metrics_summary_is_hidden_when_disabled():
    """_build_yield_metrics_summary returns no lines when yield-zero
    negatives are disabled."""
    lines = _build_yield_metrics_summary(
        enable_yield0_negatives=False,
        yield_bucket_total={"yield_0": 4, "yield_gt_0": 6},
        yield_bucket_correct={"yield_0": 3, "yield_gt_0": 5},
        yield_zero_confusion={"tp": 3, "fp": 1, "fn": 1, "tn": 5},
    )

    assert lines == []


def test_is_no_reaction_target_detects_reactant_equals_target():
    """_is_no_reaction_target is True only when the target canonicalizes to
    the reactant."""
    assert _is_no_reaction_target("CCO", "OCC") is True
    assert _is_no_reaction_target("CCO", "CC=O") is False


def test_load_eval_data_is_stable_with_local_rng(tmp_path):
    """_load_eval_data returns identical subsets for identical local RNG
    seeds, independent of global RNG state."""
    jsonl_path = tmp_path / "eval.jsonl"
    with jsonl_path.open("w") as f:
        for i in range(20):
            f.write(
                json.dumps(
                    {
                        "success": True,
                        "reaction_smiles": f"C{i}CO>>C{i}C=O",
                        "num_steps": 1,
                    }
                )
                + "\n"
            )

    global_rng = random.Random(999)
    global_rng.shuffle(list(range(10)))

    run1 = _load_eval_data(
        str(jsonl_path),
        limit=5,
        require_success=True,
        rng=random.Random(7),
    )
    run2 = _load_eval_data(
        str(jsonl_path),
        limit=5,
        require_success=True,
        rng=random.Random(7),
    )

    assert [entry["reaction_smiles"] for entry in run1] == [
        entry["reaction_smiles"] for entry in run2
    ]


def test_sample_one_jsonl_entry_is_stable_with_local_rng(tmp_path):
    """_sample_one_jsonl_entry returns the same entry for identical local RNG
    seeds."""
    jsonl_path = tmp_path / "eval_single.jsonl"
    with jsonl_path.open("w") as f:
        for i in range(10):
            f.write(
                json.dumps(
                    {
                        "success": True,
                        "reaction_smiles": f"CC{i}>>CO{i}",
                        "num_steps": 1,
                    }
                )
                + "\n"
            )

    entry1 = _sample_one_jsonl_entry(str(jsonl_path), rng=random.Random(11))
    entry2 = _sample_one_jsonl_entry(str(jsonl_path), rng=random.Random(11))

    assert entry1 is not None
    assert entry2 is not None
    assert entry1["reaction_smiles"] == entry2["reaction_smiles"]




def test_run_iteration_target_does_not_select_beam():
    """run_iteration follows an identical search path (product, step count,
    stop reason) regardless of the target; only is_match differs between a
    matching and a non-matching target."""
    evaluator = Evaluator(_DummyModel(), _DummyConfig(), torch.device("cpu"))

    def fake_forward_from_state(state, history_cache=None, step_index=0):
        return ForwardResult(
            out={"state_name": getattr(state, "name", "root")},
            bond_indices=torch.tensor([[0, 1], [1, 2]]),
            state=state,
            G=None,
            sample={"state_name": getattr(state, "name", "root")},
        )

    def fake_score_best_pair(out, bond_indices, top_k=1):
        if out["state_name"] == "root":
            return [
                Candidate(
                    is_over=False,
                    b1=(0, 1),
                    b2=(1, 2),
                    swap_type=0,
                    action_score=10.0,
                ),
                Candidate(
                    is_over=False,
                    b1=(0, 1),
                    b2=(0, 1),
                    swap_type=0,
                    action_score=5.0,
                ),
            ][:top_k]
        return [Candidate(is_over=True, action_score=0.0)]

    def fake_apply_step(state, b1, b2, swap_type):
        new_state = SimpleNamespace(name="high" if b2 == (1, 2) else "low")
        product = "CC" if new_state.name == "high" else "CCC"
        return StepResult(product, new_state, True)

    evaluator.forward_from_state = fake_forward_from_state
    evaluator.score_best_pair = fake_score_best_pair
    evaluator.apply_step = fake_apply_step

    matching_target = evaluator.run_iteration(
        _dummy_state(), ev.canonicalize_smiles("CC"), "C", False, history_cache=[]
    )
    different_target = evaluator.run_iteration(
        _dummy_state(), ev.canonicalize_smiles("CCC"), "C", False, history_cache=[]
    )

    assert (
        matching_target.product_smi,
        matching_target.predicted_steps,
        matching_target.stop_reason,
    ) == (
        different_target.product_smi,
        different_target.predicted_steps,
        different_target.stop_reason,
    )
    assert matching_target.product_smi == "CC"
    assert matching_target.is_match != different_target.is_match
