"""Tests for Evaluator.run_iteration match-state preservation and
stop reason correctness across 1-4 step scenarios with all termination modes.

Mocks isolate forward_from_state, score_best_pair, and apply_step so each
termination path can be tested deterministically without a trained model.
"""

import numpy as np
import pytest
import torch
from dataclasses import dataclass
from itertools import product as iter_product
from unittest.mock import patch, MagicMock

from bond_match.evaluate_ranking import (
    BEState,
    Candidate,
    Evaluator,
    ForwardResult,
    IterationResult,
    StepResult,
    StopReason,
    canonicalize_smiles,
    check_smiles_match,
)


def make_dummy_state():
    """Minimal BEState for tests that mock apply_step."""
    return BEState(
        be_matrix=np.zeros((2, 2)),
        atoms_identity=np.zeros((2, 2)),
    )


def make_dummy_forward_result(state=None):
    """Minimal ForwardResult for mocking forward_from_state."""
    return ForwardResult(
        out={},
        bond_indices=torch.zeros(2, 2, dtype=torch.long),
        state=state or make_dummy_state(),
        G=None,
    )


def make_candidate(virtual_prob=0.9, max_bond_prob=0.95, pair_score=1.0, is_over=None):
    """Pair-action candidate for run_iteration tests."""
    if is_over is None:
        is_over = False
    return Candidate(
        is_over=is_over,
        b1=(0, 1),
        b2=(2, 3),
        swap_type=0,
        virtual_prob=virtual_prob,
        embeddings=torch.zeros(4, 8),
        max_bond_prob=max_bond_prob,
        pair_score=pair_score,
    )


def make_stop_candidate(pair_score=1.0):
    """Explicit stop-action candidate."""
    return Candidate(
        is_over=True,
        embeddings=torch.zeros(4, 8),
        action_score=pair_score,
        action_prob=1.0,
        pair_score=pair_score,
        pair_prob=1.0,
    )


@dataclass
class MockConfig:
    """Minimal config with the fields run_iteration reads."""

    eval_beam_width: int = 1
    enable_history_context: bool = False
    max_eval_steps: int = 5
    virtual_prob_threshold: float = 0.5
    bond_prob_threshold_multi: float = 0.8
    ranking_loss_type: str = "softmax"
    extended_atom_features: bool = False


def make_evaluator(config=None):
    """Create an Evaluator with a minimal mock model and the given config."""
    model = MagicMock()
    model.use_pair_bond_type = False
    cfg = config or MockConfig()
    return Evaluator(model=model, config=cfg, device=torch.device("cpu"))


class TestOneStepStopImmediately:
    """Step 1 already produced a result. run_iteration should preserve it
    when the model says stop on the very first check."""

    @patch.object(
        Evaluator, "score_best_pair", return_value=[make_candidate(is_over=True)]
    )
    @patch.object(
        Evaluator, "forward_from_state", return_value=make_dummy_forward_result()
    )
    def test_low_virtual_prob_preserves_match(self, mock_fwd, mock_score):
        """Immediate stop prediction preserves a matching initial state."""
        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.3,
            initial_smi="CCO",
            initial_is_match=True,
        )
        assert result.is_match is True
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER
        assert result.predicted_steps == 1

    @patch.object(
        Evaluator, "score_best_pair", return_value=[make_candidate(is_over=True)]
    )
    @patch.object(
        Evaluator, "forward_from_state", return_value=make_dummy_forward_result()
    )
    def test_low_virtual_prob_preserves_no_match(self, mock_fwd, mock_score):
        """Immediate stop prediction preserves a non-matching initial state."""
        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.3,
            initial_smi="CC",
            initial_is_match=False,
        )
        assert result.is_match is False
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER
        assert result.predicted_steps == 1

    def test_none_smi_gives_no_product(self):
        """A missing product SMILES yields NO_PRODUCT and preserves is_match."""
        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi=None,
            initial_is_match=False,
        )
        assert result.is_match is False
        assert result.stop_reason == StopReason.NO_PRODUCT


class TestOneStepForwardFails:
    @patch.object(Evaluator, "forward_from_state", return_value=None)
    def test_forward_fails_preserves_match(self, mock_fwd):
        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CCO",
            initial_is_match=True,
        )
        assert result.is_match is True
        assert result.stop_reason == StopReason.FORWARD_FAILED
        assert result.predicted_steps == 1

    @patch.object(Evaluator, "forward_from_state", return_value=None)
    def test_forward_fails_preserves_no_match(self, mock_fwd):
        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CC",
            initial_is_match=False,
        )
        assert result.is_match is False
        assert result.stop_reason == StopReason.FORWARD_FAILED


class TestOneStepNoCandidates:
    @patch.object(Evaluator, "score_best_pair", return_value=None)
    @patch.object(Evaluator, "forward_from_state")
    def test_no_candidates_preserves_match(self, mock_fwd, mock_score):
        mock_fwd.return_value = make_dummy_forward_result()
        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CCO",
            initial_is_match=True,
        )
        assert result.is_match is True
        assert result.stop_reason == StopReason.NO_CANDIDATES


class TestOneStepLowBondProb:
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_low_bond_prob_preserves_match(self, mock_fwd, mock_score):
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.return_value = [make_stop_candidate()]
        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CCO",
            initial_is_match=True,
        )
        assert result.is_match is True
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER
        assert result.predicted_steps == 1


class TestTwoSteps:
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_step2_produces_match_then_stops_low_vprob(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Step 1: no match. Step 2: match. Step 3 predicts is_over.
        Result should be is_match=True from step 2."""
        state2 = make_dummy_state()
        mock_fwd.return_value = make_dummy_forward_result(state=state2)
        mock_score.side_effect = [
            [make_candidate(max_bond_prob=0.95)],
            [make_stop_candidate()],
        ]
        mock_apply.return_value = StepResult(
            product_smi="CCO", state=state2, success=True
        )

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CC",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 2
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_step2_no_match_then_stops(self, mock_fwd, mock_score, mock_apply):
        """A non-matching second step resets a previously matching state."""
        state2 = make_dummy_state()
        mock_fwd.return_value = make_dummy_forward_result(state=state2)
        mock_score.side_effect = [[make_candidate()], [make_stop_candidate()]]
        mock_apply.return_value = StepResult(
            product_smi="CCCC", state=state2, success=True
        )

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CCO",
            initial_is_match=True,
        )
        # Step 2 produces a non-matching product, so is_match resets to False.
        assert result.is_match is False
        assert result.predicted_steps == 2

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_step2_apply_fails_preserves_step1_match(
        self, mock_fwd, mock_score, mock_apply
    ):
        """An apply_step failure preserves the matching state from step 1."""
        state2 = make_dummy_state()
        mock_fwd.return_value = make_dummy_forward_result(state=state2)
        mock_score.return_value = [make_candidate()]
        mock_apply.return_value = StepResult(product_smi="", state=None, success=False)

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CCO",
            initial_is_match=True,
        )
        # The failed candidate leaves the current beam as the terminal fallback.
        assert result.is_match is True
        assert result.stop_reason == StopReason.MAX_STEPS_REACHED


class TestThreeSteps:
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_three_steps_final_match(self, mock_fwd, mock_score, mock_apply):
        """Step 1: no match. Step 2: no match. Step 3: match. Then predict is_over."""
        call_count = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return StepResult("CC", make_dummy_state(), True)  # step 2: wrong
            else:
                return StepResult("CCO", make_dummy_state(), True)  # step 3: match

        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate()],
            [make_candidate()],
            [make_stop_candidate()],
        ]
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 3
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_three_steps_match_at_step2_lost_at_step3(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Step 1: no match. Step 2: match. Step 3: wrong product. Stop.
        is_match should be False because step 3 overwrote it."""
        call_count = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return StepResult("CCO", make_dummy_state(), True)  # step 2: match
            else:
                return StepResult("CCCC", make_dummy_state(), True)  # step 3: wrong

        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate()],
            [make_candidate()],
            [make_stop_candidate()],
        ]
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is False
        assert result.predicted_steps == 3

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_step2_match_preserved_when_step3_low_bond_prob(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Step 1: no match. Step 2: match. Step 3 predicts is_over.
        is_match should be True -- step 2 match preserved because model stopped."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate(max_bond_prob=0.95)],
            [make_stop_candidate()],
        ]
        mock_apply.return_value = StepResult("CCO", make_dummy_state(), True)

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 2
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_step2_match_preserved_when_step3_forward_fails(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Step 2: match. Step 3: forward fails. is_match=True preserved."""
        fwd_call = 0

        def fwd_side_effect(state, history_cache=None, step_index=0):
            nonlocal fwd_call
            fwd_call += 1
            if fwd_call == 1:
                return make_dummy_forward_result()
            return None  # step 3 forward fails

        mock_fwd.side_effect = fwd_side_effect
        mock_score.return_value = [make_candidate()]
        mock_apply.return_value = StepResult("CCO", make_dummy_state(), True)

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 2
        assert result.stop_reason == StopReason.FORWARD_FAILED


class TestFourSteps:
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_four_steps_match_at_end(self, mock_fwd, mock_score, mock_apply):
        """Steps 2,3,4 all succeed. Step 4 produces match. Then predict is_over."""
        call_count = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal call_count
            call_count += 1
            smi = "CCO" if call_count == 3 else f"wrong_{call_count}"
            return StepResult(smi, make_dummy_state(), True)

        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate()],
            [make_candidate()],
            [make_candidate()],
            [make_stop_candidate()],
        ]
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 4
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_four_steps_match_at_step3_lost_at_step4(
        self, mock_fwd, mock_score, mock_apply
    ):
        """A non-matching fourth step resets a match from step 3."""
        call_count = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return StepResult("CCO", make_dummy_state(), True)  # step 3: match
            return StepResult(f"wrong_{call_count}", make_dummy_state(), True)

        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate()],
            [make_candidate()],
            [make_candidate()],
            [make_stop_candidate()],
        ]
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is False
        assert result.predicted_steps == 4

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_four_steps_match_at_step3_preserved_when_step4_stops(
        self, mock_fwd, mock_score, mock_apply
    ):
        """A stop prediction preserves the match from step 3."""
        call_count = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return StepResult("CCO", make_dummy_state(), True)  # step 3
            return StepResult(f"wrong_{call_count}", make_dummy_state(), True)

        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate()],
            [make_candidate()],
            [make_stop_candidate()],
        ]
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 3
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER


class TestMaxStepsReached:
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_max_steps_reached(self, mock_fwd, mock_score, mock_apply):
        """All steps succeed with high v_prob. Hits max_eval_steps limit."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.return_value = [make_candidate()]
        mock_apply.return_value = StepResult("CC", make_dummy_state(), True)

        ev = make_evaluator(MockConfig(max_eval_steps=3))  # steps 1,2 in range(1,3)
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is False
        assert result.stop_reason == StopReason.MAX_STEPS_REACHED
        assert result.predicted_steps == 3  # 1 initial + 2 iteration steps


class TestBeamSearch:
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_beam_picks_best_scoring_terminal(self, mock_fwd, mock_score, mock_apply):
        """Beam search selects the highest-scoring matching terminal candidate."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate(pair_score=5.0), make_candidate(pair_score=1.0)],
            [make_stop_candidate()],
            [make_stop_candidate()],
        ]
        apply_count = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_count
            apply_count += 1
            if apply_count == 1:
                return StepResult("CCO", make_dummy_state(), True)
            return StepResult("CCCC", make_dummy_state(), True)

        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(eval_beam_width=2))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 2

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_beam_match_from_lower_score_not_selected(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Beam search selects a higher-scoring non-matching terminal candidate."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate(pair_score=5.0), make_candidate(pair_score=1.0)],
            [make_stop_candidate()],
            [make_stop_candidate()],
        ]
        apply_count = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_count
            apply_count += 1
            if apply_count == 1:
                return StepResult(
                    "CCCC", make_dummy_state(), True
                )
            return StepResult("CCO", make_dummy_state(), True)

        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(eval_beam_width=2))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is False
        assert result.predicted_steps == 2

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_beam_multi_step_diverge_and_converge(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Beam width=2, 3 steps. Two paths diverge at step 2, both continue
        at step 3. The lower-scoring path at step 2 accumulates more score
        at step 3 and wins overall with a match."""
        mock_fwd.return_value = make_dummy_forward_result()

        score_call = 0

        def score_side_effect(out, bond_indices, top_k=1):
            nonlocal score_call
            score_call += 1
            if score_call == 1:
                return [
                    make_candidate(pair_score=3.0),
                    make_candidate(pair_score=1.0),
                ]
            if score_call in (2, 3):
                return [
                    make_candidate(pair_score=8.0),
                ]
            return [make_stop_candidate()]

        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            if apply_call <= 2:
                return StepResult(f"wrong_{apply_call}", make_dummy_state(), True)
            elif apply_call == 3:
                return StepResult("CCCC", make_dummy_state(), True)
            else:
                return StepResult("CCO", make_dummy_state(), True)

        mock_score.side_effect = score_side_effect
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(eval_beam_width=2))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        # The higher-scoring terminal path is non-matching.
        assert result.is_match is False
        assert result.predicted_steps == 3

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_beam_multi_step_best_path_matches(self, mock_fwd, mock_score, mock_apply):
        """Beam width=2, 3 steps. Highest cumulative score path also matches."""
        mock_fwd.return_value = make_dummy_forward_result()

        score_call = 0

        def score_side_effect(out, bond_indices, top_k=1):
            nonlocal score_call
            score_call += 1
            if score_call == 1:
                return [
                    make_candidate(pair_score=5.0),
                    make_candidate(pair_score=1.0),
                ]
            if score_call in (2, 3):
                return [make_candidate(pair_score=3.0)]
            return [make_stop_candidate()]

        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            if apply_call <= 2:
                return StepResult(f"wrong_{apply_call}", make_dummy_state(), True)
            elif apply_call == 3:
                return StepResult("CCO", make_dummy_state(), True)
            else:
                return StepResult("CCCC", make_dummy_state(), True)

        mock_score.side_effect = score_side_effect
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(eval_beam_width=2))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 3

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_beam_one_path_terminates_early_other_continues(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Beam width=2. At step 2, path A gets low v_prob (terminates),
        path B continues to step 3 where it finds a match with higher score."""
        mock_fwd.return_value = make_dummy_forward_result()

        score_call = 0

        def score_side_effect(out, bond_indices, top_k=1):
            nonlocal score_call
            score_call += 1
            if score_call == 1:
                return [
                    make_candidate(pair_score=2.0),
                    make_candidate(pair_score=1.0),
                ]
            if score_call == 2:
                return [make_stop_candidate()]
            if score_call == 3:
                return [make_candidate(pair_score=5.0)]
            return [make_stop_candidate()]

        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            if apply_call == 1:
                return StepResult("wrong_A", make_dummy_state(), True)
            elif apply_call == 2:
                return StepResult("wrong_B", make_dummy_state(), True)
            else:
                return StepResult("CCO", make_dummy_state(), True)

        mock_score.side_effect = score_side_effect
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(eval_beam_width=2))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        # The continued path has the highest terminal score and matches.
        assert result.is_match is True
        assert result.predicted_steps == 3

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_beam_pruning_keeps_top_k(self, mock_fwd, mock_score, mock_apply):
        """Beam pruning retains two expanded candidates before forward failure."""
        fwd_call = 0

        def fwd_side_effect(state, history_cache=None, step_index=0):
            nonlocal fwd_call
            fwd_call += 1
            if fwd_call == 1:
                return make_dummy_forward_result()
            return None

        mock_fwd.side_effect = fwd_side_effect

        mock_score.return_value = [
            make_candidate(pair_score=10.0),
            make_candidate(pair_score=1.0),
        ]

        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            return StepResult(f"prod_{apply_call}", make_dummy_state(), True)

        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(eval_beam_width=2))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        # The two expanded beam entries both terminate on forward failure.
        assert result.stop_reason == StopReason.FORWARD_FAILED
        assert result.predicted_steps == 2
        assert apply_call == 2


class TestGreedyMultiStep:
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_greedy_processes_one_candidate_per_step(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Greedy mode requests one candidate and keeps one path per step."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [
            [make_candidate(pair_score=1.0)],
            [make_stop_candidate()],
        ]
        mock_apply.return_value = StepResult("CCO", make_dummy_state(), True)

        ev = make_evaluator(MockConfig(eval_beam_width=1))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 2
        assert mock_score.call_count == 2
        for call in mock_score.call_args_list:
            _, kwargs = call
            assert kwargs.get("top_k", 1) == 1

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_greedy_three_steps_sequential(self, mock_fwd, mock_score, mock_apply):
        """Greedy 3-step chain: each step has exactly one beam entry."""
        mock_fwd.return_value = make_dummy_forward_result()
        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            return StepResult(f"prod_{apply_call}", make_dummy_state(), True)

        mock_score.side_effect = [
            [make_candidate(pair_score=2.0)],
            [make_candidate(pair_score=3.0)],
            [make_candidate(pair_score=1.0)],
            [make_stop_candidate()],
        ]
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(eval_beam_width=1))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert apply_call == 3
        assert result.predicted_steps == 4
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_greedy_is_match_tracks_latest_step_only(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Greedy: is_match flips each step depending on the latest product.
        Step 2: match. Step 3: no match. Step 4: match again. Result = match."""
        mock_fwd.return_value = make_dummy_forward_result()
        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            if apply_call in (1, 3):
                return StepResult("CCO", make_dummy_state(), True)
            return StepResult("CCCC", make_dummy_state(), True)

        mock_score.side_effect = [
            [make_candidate()],
            [make_candidate()],
            [make_candidate()],
            [make_stop_candidate()],
        ]
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(eval_beam_width=1))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        assert result.is_match is True
        assert result.predicted_steps == 4


class TestAllApplyFail:
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_all_apply_fail_preserves_initial(self, mock_fwd, mock_score, mock_apply):
        """All apply_step failures leave the initial beam entry as terminal."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.return_value = [make_candidate()]
        mock_apply.return_value = StepResult("", None, False)

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CCO",
            initial_is_match=True,
        )
        assert result.is_match is True
        # The initial beam entry is collected when no candidate advances.
        assert result.stop_reason == StopReason.MAX_STEPS_REACHED
        assert result.predicted_steps == 1


class TestNoTargetLeakage:
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_target_does_not_affect_stopping_decisions(
        self, mock_fwd, mock_score, mock_apply
    ):
        """Changing only the target changes only the final match status."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [[make_candidate()], [make_stop_candidate()]]
        mock_apply.return_value = StepResult("CCO", make_dummy_state(), True)

        ev = make_evaluator()

        result_match = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [[make_candidate()], [make_stop_candidate()]]
        mock_apply.return_value = StepResult("CCO", make_dummy_state(), True)

        result_no_match = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCCCC",
            initial_v_prob=0.9,
            initial_smi="C",
            initial_is_match=False,
        )

        assert result_match.predicted_steps == result_no_match.predicted_steps
        assert result_match.stop_reason == result_no_match.stop_reason
        assert result_match.is_match is True
        assert result_no_match.is_match is False


STOP_CONDITIONS = [
    "low_vprob",
    "forward_fail",
    "no_candidates",
    "low_bond_prob",
    "apply_fail",
]


class TestCombinatorial:
    """Systematically test all combinations of:
    - Number of successful steps before stopping: 0 to 3 (on top of step 1)
    - Stop condition at the stopping step
    - Whether the LAST successful step matches the target
    - Whether a PREVIOUS step matched the target (to verify overwrite)
    """

    @pytest.mark.parametrize(
        "n_successful_iter_steps, stop_condition",
        list(iter_product(range(4), STOP_CONDITIONS)),
    )
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_is_match_reflects_last_successful_step(
        self,
        mock_fwd,
        mock_score,
        mock_apply,
        n_successful_iter_steps,
        stop_condition,
    ):
        """For N successful iteration steps (steps 2..N+1), then a stop at
        step N+2 with the given condition: verify is_match reflects step N+1's
        product (the last successful step)."""

        target = "CCO"
        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            if apply_call == n_successful_iter_steps:
                return StepResult(target, make_dummy_state(), True)
            return StepResult(f"wrong_{apply_call}", make_dummy_state(), True)

        fwd_calls = 0

        def fwd_side_effect(state, history_cache=None, step_index=0):
            nonlocal fwd_calls
            fwd_calls += 1
            if fwd_calls <= n_successful_iter_steps:
                return make_dummy_forward_result()
            if stop_condition == "forward_fail":
                return None
            return make_dummy_forward_result()

        mock_fwd.side_effect = fwd_side_effect

        score_calls = 0

        def score_side_effect(out, bond_indices, top_k=1):
            nonlocal score_calls
            score_calls += 1
            if score_calls <= n_successful_iter_steps:
                return [make_candidate()]
            if stop_condition == "no_candidates":
                return None
            if stop_condition == "low_bond_prob":
                return [make_stop_candidate()]
            if stop_condition == "low_vprob":
                return [make_stop_candidate()]
            if stop_condition == "apply_fail":
                return [make_candidate()]
            return [make_candidate()]

        mock_score.side_effect = score_side_effect

        apply_call_for_stop = [0]

        def apply_side_effect_with_stop(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            step_num = apply_call
            if step_num <= n_successful_iter_steps:
                if step_num == n_successful_iter_steps:
                    return StepResult(target, make_dummy_state(), True)
                return StepResult(f"wrong_{step_num}", make_dummy_state(), True)
            if stop_condition == "apply_fail":
                return StepResult("", None, False)
            return StepResult(f"extra_{step_num}", make_dummy_state(), True)

        apply_call = 0
        mock_apply.side_effect = apply_side_effect_with_stop

        ev = make_evaluator(MockConfig(max_eval_steps=6))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon=target,
            initial_v_prob=0.9,
            initial_smi="initial_wrong",
            initial_is_match=False,
        )

        if n_successful_iter_steps == 0:
            if stop_condition in (
                "forward_fail",
                "no_candidates",
                "low_bond_prob",
                "low_vprob",
            ):
                assert result.is_match is False, (
                    f"n_steps=0, stop={stop_condition}: expected is_match=False "
                    f"(initial preserved), got True"
                )
                assert result.predicted_steps == 1
            elif stop_condition == "apply_fail":
                assert result.is_match is False
                assert result.predicted_steps == 1
        else:
            if stop_condition in (
                "forward_fail",
                "no_candidates",
                "low_bond_prob",
                "low_vprob",
            ):
                assert result.is_match is True, (
                    f"n_steps={n_successful_iter_steps}, stop={stop_condition}: "
                    f"expected is_match=True (last step was match), got False"
                )
                assert result.predicted_steps == n_successful_iter_steps + 1
            elif stop_condition == "apply_fail":
                assert result.is_match is True, (
                    f"n_steps={n_successful_iter_steps}, stop={stop_condition}: "
                    f"expected is_match=True, got False"
                )


class TestCombinatorialMatchPreservation:
    """Specifically test: match at step N, then model stops at step N+1.
    The match from step N is preserved for all stop conditions
    that prevent step N+1 from producing a new product."""

    PRESERVING_STOPS = ["forward_fail", "no_candidates", "low_bond_prob", "apply_fail"]

    @pytest.mark.parametrize(
        "match_at_step, stop_condition",
        list(iter_product([2, 3, 4], PRESERVING_STOPS)),
    )
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_match_preserved_on_stop(
        self,
        mock_fwd,
        mock_score,
        mock_apply,
        match_at_step,
        stop_condition,
    ):
        """A stopped next step preserves the previous matching product."""
        target = "CCO"
        n_iter_steps = match_at_step - 1  # steps 2..match_at_step in iteration

        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            if apply_call == n_iter_steps:
                return StepResult(target, make_dummy_state(), True)
            if stop_condition == "apply_fail" and apply_call == n_iter_steps + 1:
                return StepResult("", None, False)
            return StepResult(f"wrong_{apply_call}", make_dummy_state(), True)

        fwd_call = 0

        def fwd_side_effect(state, history_cache=None, step_index=0):
            nonlocal fwd_call
            fwd_call += 1
            if fwd_call <= n_iter_steps:
                return make_dummy_forward_result()
            if stop_condition == "forward_fail":
                return None
            return make_dummy_forward_result()

        score_call = 0

        def score_side_effect(out, bond_indices, top_k=1):
            nonlocal score_call
            score_call += 1
            if score_call <= n_iter_steps:
                return [make_candidate()]
            if stop_condition == "no_candidates":
                return None
            if stop_condition == "low_bond_prob":
                return [make_stop_candidate()]
            return [make_candidate()]

        mock_fwd.side_effect = fwd_side_effect
        mock_score.side_effect = score_side_effect
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(max_eval_steps=6))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon=target,
            initial_v_prob=0.9,
            initial_smi="wrong_initial",
            initial_is_match=False,
        )

        assert result.is_match is True, (
            f"Match at step {match_at_step}, stopped by {stop_condition}: "
            f"expected is_match=True but got False. "
            f"stop_reason={result.stop_reason}, steps={result.predicted_steps}"
        )
        assert result.predicted_steps == match_at_step


class TestCombinatorialMatchOverwrite:
    """Specifically test: match at step N, then step N+1 SUCCEEDS with wrong
    product. The later product overwrites the previous match state."""

    @pytest.mark.parametrize("match_at_step", [2, 3, 4])
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_match_overwritten_by_successful_wrong_step(
        self,
        mock_fwd,
        mock_score,
        mock_apply,
        match_at_step,
    ):
        """A successful non-matching step clears a previous match."""
        target = "CCO"
        n_iter_steps = match_at_step - 1

        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            if apply_call == n_iter_steps:
                return StepResult(target, make_dummy_state(), True)
            return StepResult(f"wrong_{apply_call}", make_dummy_state(), True)

        mock_fwd.return_value = make_dummy_forward_result()
        scores = []
        for i in range(n_iter_steps + 2):
            if i <= n_iter_steps:
                scores.append([make_candidate()])
            else:
                scores.append([make_stop_candidate()])
        mock_score.side_effect = scores
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(max_eval_steps=6))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon=target,
            initial_v_prob=0.9,
            initial_smi="wrong_initial",
            initial_is_match=False,
        )

        assert result.is_match is False, (
            f"Match at step {match_at_step}, then step {match_at_step + 1} "
            f"produced wrong product: expected is_match=False but got True"
        )
        assert result.predicted_steps == match_at_step + 1


class TestCombinatorialInitialMatch:
    """Test: step 1 already matched (initial_is_match=True).
    If the model continues and produces a wrong product, the match is lost.
    If the model stops, the match is preserved."""

    PRESERVING_STOPS = ["forward_fail", "no_candidates", "low_bond_prob", "apply_fail"]

    @pytest.mark.parametrize("stop_condition", PRESERVING_STOPS)
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_initial_match_preserved_on_stop(
        self,
        mock_fwd,
        mock_score,
        mock_apply,
        stop_condition,
    ):
        """A stop before step 2 preserves an initial match."""
        if stop_condition == "forward_fail":
            mock_fwd.return_value = None
        else:
            mock_fwd.return_value = make_dummy_forward_result()

        if stop_condition == "no_candidates":
            mock_score.return_value = None
        elif stop_condition == "low_bond_prob":
            mock_score.return_value = [make_stop_candidate()]
        else:
            mock_score.return_value = [make_candidate()]

        if stop_condition == "apply_fail":
            mock_apply.return_value = StepResult("", None, False)
        else:
            mock_apply.return_value = StepResult(
                "doesnt_matter", make_dummy_state(), True
            )

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CCO",
            initial_is_match=True,
        )

        assert result.is_match is True, (
            f"Initial match, stopped by {stop_condition}: "
            f"expected is_match=True, got False"
        )
        assert result.predicted_steps == 1

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_initial_match_lost_when_step2_succeeds_with_wrong(
        self,
        mock_fwd,
        mock_score,
        mock_apply,
    ):
        """A successful non-matching step 2 clears an initial match."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.side_effect = [[make_candidate()], [make_stop_candidate()]]
        mock_apply.return_value = StepResult("WRONG", make_dummy_state(), True)

        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.9,
            initial_smi="CCO",
            initial_is_match=True,
        )
        assert result.is_match is False
        assert result.predicted_steps == 2

    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_initial_match_preserved_on_low_vprob(
        self,
        mock_fwd,
        mock_score,
        mock_apply,
    ):
        """An explicit stop prediction preserves an initial match."""
        mock_fwd.return_value = make_dummy_forward_result()
        mock_score.return_value = [make_candidate(is_over=True)]
        ev = make_evaluator()
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon="CCO",
            initial_v_prob=0.3,
            initial_smi="CCO",
            initial_is_match=True,
        )
        assert result.is_match is True
        assert result.predicted_steps == 1
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER


class TestMatchAtFinalStep:
    """Tests match preservation when the matching product occurs at the final step."""

    @pytest.mark.parametrize("match_step", [2, 3, 4])
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_match_at_step_N_with_low_vprob_stop(
        self,
        mock_fwd,
        mock_score,
        mock_apply,
        match_step,
    ):
        """An explicit stop after step N preserves that step's match status."""
        target = "CCO"

        mock_fwd.return_value = make_dummy_forward_result()
        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            iter_step = apply_call
            if iter_step == match_step - 1:
                return StepResult(target, make_dummy_state(), True)
            return StepResult(f"wrong_{iter_step}", make_dummy_state(), True)

        scores = []
        for i in range(match_step):
            if i < match_step - 1:
                scores.append([make_candidate()])
            else:
                scores.append([make_stop_candidate()])
        mock_score.side_effect = scores
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(max_eval_steps=6))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon=target,
            initial_v_prob=0.9,
            initial_smi="wrong_init",
            initial_is_match=False,
        )

        assert result.is_match is True, (
            f"Match at step {match_step} with low v_prob stop: "
            f"expected True, got False. steps={result.predicted_steps}, "
            f"reason={result.stop_reason}"
        )
        assert result.predicted_steps == match_step
        assert result.stop_reason == StopReason.PREDICTED_IS_OVER

    @pytest.mark.parametrize("match_step", [2, 3, 4])
    @patch.object(Evaluator, "apply_step")
    @patch.object(Evaluator, "score_best_pair")
    @patch.object(Evaluator, "forward_from_state")
    def test_match_at_step_N_with_max_steps_stop(
        self,
        mock_fwd,
        mock_score,
        mock_apply,
        match_step,
    ):
        """The last allowed step preserves its match status."""
        target = "CCO"
        max_steps = match_step

        mock_fwd.return_value = make_dummy_forward_result()
        apply_call = 0

        def apply_side_effect(state, b1, b2, swap_type):
            nonlocal apply_call
            apply_call += 1
            if apply_call == match_step - 1:
                return StepResult(target, make_dummy_state(), True)
            return StepResult(f"wrong_{apply_call}", make_dummy_state(), True)

        mock_score.return_value = [make_candidate()]
        mock_apply.side_effect = apply_side_effect

        ev = make_evaluator(MockConfig(max_eval_steps=max_steps))
        result = ev.run_iteration(
            state=make_dummy_state(),
            target_canon=target,
            initial_v_prob=0.9,
            initial_smi="wrong_init",
            initial_is_match=False,
        )

        assert result.is_match is True, (
            f"Match at step {match_step} with max_steps={max_steps}: "
            f"expected True, got False. steps={result.predicted_steps}, "
            f"reason={result.stop_reason}"
        )
        assert result.predicted_steps == match_step


class TestCanonicalizeAndMatch:
    """Test that canonicalize_smiles and check_smiles_match handle the
    kinds of SMILES that be_result_to_smiles can produce."""

    def test_mapped_vs_unmapped_same_molecule(self):
        mapped = "[CH3:1]O"
        unmapped = "CO"
        assert canonicalize_smiles(mapped) == canonicalize_smiles(unmapped)

    def test_multi_fragment_sorted(self):
        """Fragments are sorted alphabetically after canonicalization."""
        assert canonicalize_smiles("O.CC") == canonicalize_smiles("CC.O")
        assert canonicalize_smiles("O.CC") == "CC.O"

    def test_subset_match_with_extra_fragments(self):
        """Target fragments match when predicted fragments include a catalyst."""
        target = canonicalize_smiles("CCO")
        predicted = canonicalize_smiles("CCO.c1ccccc1")
        assert check_smiles_match(predicted, target)

    def test_no_match_when_target_fragment_missing(self):
        """Missing target fragments fail the match check."""
        target = canonicalize_smiles("CCO.CC")
        predicted = canonicalize_smiles("CCO")
        assert not check_smiles_match(predicted, target)

    def test_empty_smiles(self):
        assert canonicalize_smiles("") == ""
        assert canonicalize_smiles(None) == ""
        assert not check_smiles_match("", "CCO")
        assert not check_smiles_match("CCO", "")
        assert not check_smiles_match("", "")

    def test_invalid_smiles_passthrough(self):
        """Invalid SMILES strings are returned unchanged."""
        canon = canonicalize_smiles("INVALID_SMILES")
        assert canon == "INVALID_SMILES"

    def test_explicit_h_removed(self):
        """Removing hydrogens normalizes explicit-H representations."""
        assert canonicalize_smiles("[CH4]") == canonicalize_smiles("C")

    def test_charged_species(self):
        """Charged species canonicalize consistently."""
        assert canonicalize_smiles("[Na+].[Cl-]") == canonicalize_smiles("[Cl-].[Na+]")

    def test_complex_reaction_product(self):
        """Atom-mapped and unmapped forms of a complex product match."""
        target = (
            "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9]"
            "[CH2:10][CH:11]([NH:22][C:29]([c:28]2[cH:23][cH:24][c:25]([Br:31])"
            "[cH:26][cH:27]2)=[O:30])[CH:12]([c:14]2[cH:15][c:16]([Cl:21])"
            "[c:17]([Cl:20])[cH:18][cH:19]2)[CH2:13]1"
        )
        target_canon = canonicalize_smiles(target)
        unmapped = "CC(C)(C)OC(=O)N1CCC(NC(=O)c2ccc(Br)cc2)C(c2ccc(Cl)c(Cl)c2)C1"
        unmapped_canon = canonicalize_smiles(unmapped)

        assert target_canon == unmapped_canon
        assert check_smiles_match(unmapped_canon, target_canon)
        assert check_smiles_match(target_canon, unmapped_canon)

    def test_product_with_reagent_still_matches(self):
        """If predicted product includes reagent fragments, subset check passes."""
        target_canon = canonicalize_smiles(
            "CC(C)(C)OC(=O)N1CCC(NC(=O)c2ccc(Br)cc2)C(c2ccc(Cl)c(Cl)c2)C1"
        )
        predicted_canon = canonicalize_smiles(
            "CC(C)(C)OC(=O)N1CCC(NC(=O)c2ccc(Br)cc2)C(c2ccc(Cl)c(Cl)c2)C1"
            ".Cc1ccc(S(=O)(=O)O)cc1"
        )
        assert check_smiles_match(predicted_canon, target_canon)
