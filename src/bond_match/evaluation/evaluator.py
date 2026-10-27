import csv
import json
import random
from collections import defaultdict
from typing import Any, NamedTuple, Optional
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
from bond_match.dataset import (
    _be_to_nx,
    augment_with_lp_edges,
    collate_fn,
    get_graph_features_nx,
    nx_to_mol,
    trajectory_collate_fn,
)

from bond_match.evaluation.be_ops import (
    apply_predicted_step_be,
    apply_step_with_lp,
    be_result_to_smiles,
    decode_swap_type,
    score_actions,
)
from bond_match.evaluation.constants import (
    logger,
)
from bond_match.evaluation.data import (
    InferenceDataset,
    _load_all_eval_entries,
    _load_eval_data,
    _sample_one_jsonl_entry,
    inference_collate,
)
from bond_match.evaluation.results import (
    ActiveMolecule,
    BEState,
    BeamEntry,
    Candidate,
    DisplayResult,
    EvalResult,
    ForwardResult,
    IterationResult,
    SampleMeta,
    StepResult,
    StopReason,
    TerminalEntry,
)
from bond_match.evaluation.utils import (
    _build_yield_metrics_summary,
    _is_no_reaction_target,
    _maybe_rewrite_yield0_entry,
    canonicalize_smiles,
    check_smiles_match,
    is_entry_successful,
    is_no_reaction_match,
    strip_atom_mappings_from_data,
)

class _MolSlice(NamedTuple):
    """Per-molecule views into a batched model output."""

    embeddings: torch.Tensor
    history_embeddings: torch.Tensor
    binary_logits: torch.Tensor
    stop_logit: torch.Tensor
    local_bond_indices: torch.Tensor
    bond_atom_indices: torch.Tensor | None


class _DatasetStats:
    """Accumulates per-sample outcomes during ``Evaluator.evaluate_dataset``."""

    def __init__(self) -> None:
        self.correct_top1 = 0
        self.total_samples = 0
        self.step_count_total: defaultdict = defaultdict(int)
        self.step_count_correct: defaultdict = defaultdict(int)
        self.yield_bucket_total: defaultdict = defaultdict(int)
        self.yield_bucket_correct: defaultdict = defaultdict(int)
        self.yield_zero_confusion: defaultdict = defaultdict(int)
        self.step1_apply_fail = 0

    def update(self, meta: SampleMeta, eval_result: EvalResult) -> None:
        """Record one evaluated sample's match, step-count, and yield outcome."""
        self.total_samples += 1
        num_steps = meta.num_steps
        self.step_count_total[num_steps] += 1
        yield_bucket = None
        if meta.yield_value is not None:
            yield_bucket = "yield_0" if meta.yield_value == 0 else "yield_gt_0"
            self.yield_bucket_total[yield_bucket] += 1

        if eval_result.stop_reason == StopReason.STEP1_FAILED:
            self.step1_apply_fail += 1

        if yield_bucket is not None:
            actual_yield_zero = yield_bucket == "yield_0"
            predicted_yield_zero = (
                eval_result.stop_reason == StopReason.PREDICTED_IS_OVER
                and eval_result.predicted_steps == 0
            )
            if actual_yield_zero and predicted_yield_zero:
                self.yield_zero_confusion["tp"] += 1
            elif actual_yield_zero:
                self.yield_zero_confusion["fn"] += 1
            elif predicted_yield_zero:
                self.yield_zero_confusion["fp"] += 1
            else:
                self.yield_zero_confusion["tn"] += 1

        if eval_result.is_match:
            self.correct_top1 += 1
            self.step_count_correct[num_steps] += 1
            if meta.yield_value is not None:
                self.yield_bucket_correct[yield_bucket] += 1


class Evaluator:
    """V3 blockspace evaluator. Greedy and beam search via unified iteration."""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        enable_yield0_negatives: bool = False,
        eval_seed: int = 42,
    ):
        """Store model, config, and device, and seed the sampling RNGs.

        Args:
            model: Trained BondGAT model in eval-ready state.
            config: Ranking config providing eval settings (beam width,
                selection mode, history context, ...).
            device: Device for model forwards.
            enable_yield0_negatives: Rewrite yield-0 entries into no-reaction
                targets and report detection metrics for them.
            eval_seed: Seed for data subsampling and random-sample picking.
        """
        self.model = model
        self.config = config
        self.device = device
        self.use_ebm = config.ranking_loss_type == "ebm"
        self.enable_yield0_negatives = enable_yield0_negatives
        self.eval_seed = eval_seed
        self.sample_rng = random.Random(eval_seed + 1)

    def uses_trajectory_mode(self) -> bool:
        """Whether samples are evaluated in trajectory mode (vs. step mode)."""
        return getattr(self.config, "sample_mode", "step") == "trajectory"

    def should_use_batched_greedy_eval(self) -> bool:
        """Whether the fast batched greedy path applies (no beam, no trajectory)."""
        return self.config.eval_beam_width == 1 and not self.uses_trajectory_mode()

    def _use_pair_bond_type(self) -> bool:
        """Whether the model provides a pair swap-type classifier."""
        return self.model.use_pair_bond_type and hasattr(
            self.model, "pair_bond_type_classifier"
        )

    def _decode_swap_or_default(
        self,
        local_b1: int,
        local_b2: int,
        node_embeddings: torch.Tensor | None,
        bond_atom_indices: torch.Tensor | None,
        bond_embeddings: torch.Tensor,
    ) -> int:
        """Swap type for a bond pair, or 0 when no swap classifier is configured."""
        if not self._use_pair_bond_type():
            return 0
        return decode_swap_type(
            self.model,
            local_b1,
            local_b2,
            node_embeddings=node_embeddings,
            bond_atom_indices=bond_atom_indices,
            bond_embeddings=bond_embeddings,
        )[0]

    def _score_actions_for_mol(
        self,
        embeddings: torch.Tensor,
        binary_logits: torch.Tensor,
        stop_logit: torch.Tensor,
        top_k: int = 1,
    ) -> list:
        """Score pair/stop actions for one molecule with the configured selection settings."""
        return score_actions(
            self.model,
            embeddings,
            binary_logits,
            stop_logit,
            self.device,
            top_k=top_k,
            selection_mode=self.config.pair_selection_mode,
            temperature=self.config.pair_selection_temperature,
            softmax_topk=self.config.pair_softmax_topk,
            pair_scoring_mode=self.config.pair_scoring_mode,
        )

    def _model_forward(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        bond_indices: torch.Tensor,
        history_context: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        step_index: torch.Tensor,
        trajectory_ids: torch.Tensor | None = None,
    ) -> dict:
        """Run the model under ``no_grad`` with the standard argument set."""
        with torch.no_grad():
            return self.model(
                x,
                batch_idx,
                edge_index,
                edge_attr,
                bond_indices,
                history_context=history_context,
                history_mask=history_mask,
                step_index=step_index,
                trajectory_ids=trajectory_ids,
            )

    def _collated_to_device(
        self, collated: tuple
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Move the standard collated tensors to the eval device.

        Returns:
            Tuple of (x, batch_idx, edge_index, edge_attr, bond_indices,
            step_index).
        """
        return (
            collated[0].to(self.device),
            collated[2].to(self.device),
            collated[3].to(self.device),
            collated[4].to(self.device),
            collated[5].to(self.device),
            collated[10].to(self.device),
        )

    def _segment_offsets(
        self, segment_ids: torch.Tensor, n_segments: int
    ) -> torch.Tensor:
        """Exclusive prefix offsets of shape ``(n_segments + 1,)`` from segment ids.

        ``offsets[i]:offsets[i+1]`` then slices segment ``i`` of any tensor
        ordered by ``segment_ids``.
        """
        counts = torch.bincount(segment_ids, minlength=n_segments)
        offsets = torch.zeros(n_segments + 1, dtype=torch.long, device=self.device)
        torch.cumsum(counts, 0, out=offsets[1:])
        return offsets

    @staticmethod
    def _state_with_lp_map(state: BEState, reverse_lp_map: dict) -> BEState:
        """Copy a BEState, replacing only its reverse lone-pair map."""
        return BEState(
            be_matrix=state.be_matrix,
            atoms_identity=state.atoms_identity,
            reverse_lp_map=reverse_lp_map,
            has_catalyst=state.has_catalyst,
            n_reactant_heavy=state.n_reactant_heavy,
            catalyst_flags=state.catalyst_flags,
        )

    def initial_state(self, meta: SampleMeta) -> BEState:
        """Build the step-0 BEState (BE matrix + atom identities) from sample metadata."""
        catalyst_flags = None
        if self.config.extended_atom_features:
            G = meta.G
            if G:
                catalyst_flags = {
                    node: G.nodes[node].get("is_catalyst", 0) for node in G.nodes()
                }

        return BEState(
            be_matrix=meta.be_matrix,
            atoms_identity=meta.atoms_identity,
            reverse_lp_map=meta.reverse_lp_map,
            has_catalyst=meta.has_catalyst,
            n_reactant_heavy=meta.n_reactant_heavy,
            catalyst_flags=catalyst_flags,
        )

    def apply_step(
        self, state: BEState, b1: tuple, b2: tuple, swap_type: int
    ) -> StepResult:
        """Apply one predicted bond-pair step to a BE state.

        Selects the lone-pair-aware BE update when either bond touches a
        lone-pair pseudo-atom, validates the resulting BE matrix, and renders
        the product SMILES (catalyst atoms are excluded for catalytic samples).

        Args:
            state: Current BE state.
            b1: Atom index pair of the first bond.
            b2: Atom index pair of the second bond.
            swap_type: Swap variant predicted by the bond-type head.

        Returns:
            StepResult with the product SMILES and the successor state;
            ``success=False`` when the state is incomplete or the BE update
            produced negative bond orders.
        """
        if state.be_matrix is None or state.atoms_identity is None:
            return StepResult("", None, False)

        has_lp = any(a in state.reverse_lp_map for a in (*b1, *b2))
        if has_lp and state.reverse_lp_map:
            be_result = apply_step_with_lp(
                state.be_matrix,
                b1,
                b2,
                swap_type=swap_type,
                reverse_lp_map=state.reverse_lp_map,
            )
        else:
            be_result = apply_predicted_step_be(
                state.be_matrix, b1, b2, swap_type=swap_type
            )

        if np.any(be_result < -0.01):
            logger.debug("Negative entries in BE result")
            return StepResult("", None, False)

        if state.has_catalyst and state.n_reactant_heavy > 0:
            product_smi = self._reactant_only_smiles(state, be_result)
        else:
            product_smi = be_result_to_smiles(be_result, state.atoms_identity)

        new_state = BEState(
            be_matrix=be_result,
            atoms_identity=state.atoms_identity,
            reverse_lp_map={},
            has_catalyst=state.has_catalyst,
            n_reactant_heavy=state.n_reactant_heavy,
            catalyst_flags=state.catalyst_flags,
        )
        return StepResult(product_smi, new_state, True)

    @staticmethod
    def _catalyst_reactant_mask(state: BEState, be_result: np.ndarray) -> np.ndarray:
        """Mark atoms belonging to the reactant (vs. catalyst) molecule.

        Heavy atoms are classified by their position relative to
        ``n_reactant_heavy``; hydrogens follow whichever heavy atom they are
        bonded to in ``be_result``.
        """
        n = len(state.be_matrix)
        reactant_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            atomic_num = (
                int(state.atoms_identity[i, 1]) if i < len(state.atoms_identity) else 0
            )
            if atomic_num == 0:
                continue
            elif atomic_num > 1:
                reactant_mask[i] = i < state.n_reactant_heavy
            else:
                for j in range(n):
                    if (
                        be_result[i, j] > 0
                        and int(state.atoms_identity[j, 1]) > 1
                        and j < state.n_reactant_heavy
                    ):
                        reactant_mask[i] = True
                        break
        return reactant_mask

    def _reactant_only_smiles(self, state: BEState, be_result: np.ndarray) -> str:
        """Render the product SMILES restricted to reactant atoms.

        Bond order toward excluded catalyst atoms is folded into the diagonal
        (free valence) of the bonded reactant atom so valences stay correct.
        """
        reactant_mask = self._catalyst_reactant_mask(state, be_result)
        react_idx = np.where(reactant_mask)[0]
        cat_idx = np.where(~reactant_mask)[0]

        be_for_smi = be_result.copy()
        for i in react_idx:
            cat_bond_sum = sum(be_result[i, j] for j in cat_idx if be_result[i, j] > 0)
            if cat_bond_sum > 0:
                be_for_smi[i, i] += cat_bond_sum

        be_react = be_for_smi[np.ix_(react_idx, react_idx)]
        atoms_react = state.atoms_identity[react_idx]
        return be_result_to_smiles(be_react, atoms_react)

    def _collate_and_forward(
        self,
        sample: dict,
        history_cache: list | None = None,
    ) -> tuple[dict, torch.Tensor]:
        """Collate a single sample, move it to device, and run a model forward.

        In trajectory mode the history samples are collated together with the
        current sample and the outputs are sliced back down to the current
        state afterwards.

        Args:
            sample: Collate-ready sample dict.
            history_cache: Per-step history; (embeddings, step) tuples in step
                mode, (sample dict, step) tuples in trajectory mode.

        Returns:
            Tuple of (model output dict, bond atom-index pairs local to the
            current molecule).
        """
        trajectory_mode = getattr(self.config, "sample_mode", "step") == "trajectory"
        if trajectory_mode:
            history_states = [h[0] for h in (history_cache or []) if isinstance(h[0], dict)]
            states = history_states + [sample]
            batch = trajectory_collate_fn([{"states": states}], enable_matrix_output=False)
        else:
            batch = collate_fn([sample], enable_matrix_output=False)

        x_b, batch_idx, ei_b, ea_b, bond_idx, step_idx_b = self._collated_to_device(batch)
        trajectory_ids = batch[11].to(self.device) if len(batch) > 11 else None

        history_context = None
        history_mask = None
        if self.config.enable_history_context:
            if history_cache:
                # history_cache is list of (embeddings, step_index) tuples
                all_hist = torch.cat([emb for emb, _ in history_cache], dim=0)
                history_context = all_hist.unsqueeze(0).to(
                    self.device
                )  # (1, total_tokens, d_model)
                history_mask = torch.zeros(
                    1, all_hist.size(0), dtype=torch.bool, device=self.device
                )
            else:
                history_context, history_mask = self.model.empty_history(1, self.device)

        out = self._model_forward(
            x_b,
            batch_idx,
            ei_b,
            ea_b,
            bond_idx,
            history_context,
            history_mask,
            step_idx_b,
            trajectory_ids=trajectory_ids,
        )

        out_bond_idx = out.get("bond_indices_out", bond_idx)
        if trajectory_mode:
            out, out_bond_idx = self._slice_trajectory_output(
                out, out_bond_idx, batch_idx, step_idx_b
            )
        out["_eval_sample"] = sample
        return out, out_bond_idx

    @staticmethod
    def _slice_trajectory_output(
        out: dict,
        out_bond_idx: torch.Tensor,
        batch_idx: torch.Tensor,
        step_idx_b: torch.Tensor,
    ) -> tuple[dict, torch.Tensor]:
        """Restrict trajectory-mode outputs to the last (current) state.

        Trajectory collation concatenates history states before the current
        one; only the final state's bonds and stop logit are scored.
        """
        current_state = step_idx_b.numel() - 1
        bond_batch_vec = batch_idx[out_bond_idx[:, 0]]
        current_mask = bond_batch_vec == current_state
        atom_mask = batch_idx == current_state
        current_atom_offset = torch.where(atom_mask)[0].min()
        current_bond_idx_global = out_bond_idx[current_mask]
        out = dict(out)
        out["embeddings"] = out["embeddings"][current_mask]
        out["logits_binary"] = out["logits_binary"][current_mask]
        out["logits_stop"] = out["logits_stop"][current_state : current_state + 1]
        if "bond_indices_out" in out:
            out["bond_indices_out"] = current_bond_idx_global
        return out, current_bond_idx_global - current_atom_offset

    def forward_from_state(
        self, state: BEState, history_cache: list | None = None, step_index: int = 0
    ) -> Optional[ForwardResult]:
        """Run model forward from BE matrix state.

        Rebuilds the graph via ``_build_sample_from_state`` (identical to
        training's _process_v3_mechanism) and collates + forwards it.

        Args:
            state: BE state to score.
            history_cache: Optional history for the forward (see
                ``_collate_and_forward``).
            step_index: Step index fed to the model.

        Returns:
            ForwardResult, or None when graph construction or the forward
            fails.
        """
        built = self._build_sample_from_state(state, step_index)
        if built is None:
            return None
        sample, reverse_lp_map, G = built

        try:
            out, bond_idx = self._collate_and_forward(
                sample,
                history_cache,
            )
            new_state = self._state_with_lp_map(state, reverse_lp_map)
            return ForwardResult(
                out=out, bond_indices=bond_idx, state=new_state, G=G, sample=sample
            )
        except Exception:
            logger.debug("forward_from_state exception", exc_info=True)
            return None

    def score_actions(
        self, out: dict, bond_indices: torch.Tensor, top_k: int = 1
    ) -> list[Candidate]:
        """Score actions from model output.

        Args:
            out: Model output dict for a single molecule.
            bond_indices: Bond atom-index pairs local to the molecule.
            top_k: Number of top actions to return.

        Returns:
            Stop/pair candidates ordered by score.
        """
        embeddings = out["embeddings"]
        # Use self-attention-only tokens for history (matches training's encode_history_tokens)
        history_emb = out.get("history_embeddings", embeddings)
        node_embeddings = out.get("node_embeddings")
        bond_atom_indices = out.get("bond_indices_out")
        if bond_atom_indices is None:
            bond_atom_indices = bond_indices

        pair_results = self._score_actions_for_mol(
            embeddings, out["logits_binary"], out["logits_stop"][0], top_k=top_k
        )

        candidates = []
        for pr in pair_results:
            if pr.is_over:
                candidates.append(
                    Candidate(
                        is_over=True,
                        b1=None,
                        b2=None,
                        swap_type=0,
                        embeddings=history_emb.detach(),
                        action_score=pr.score,
                        action_prob=pr.action_prob,
                        virtual_prob=0.0,
                        max_bond_prob=0.0,
                    )
                )
                continue
            swap = self._decode_swap_or_default(
                pr.local_b1,
                pr.local_b2,
                node_embeddings,
                bond_atom_indices,
                embeddings,
            )
            candidates.append(
                Candidate(
                    is_over=False,
                    b1=tuple(bond_indices[pr.local_b1].tolist()),
                    b2=tuple(bond_indices[pr.local_b2].tolist()),
                    swap_type=swap,
                    embeddings=history_emb.detach(),
                    action_score=pr.score,
                    action_prob=pr.action_prob,
                    virtual_prob=1.0,
                    max_bond_prob=1.0,
                )
            )
        return candidates

    def score_best_pair(
        self, out: dict, bond_indices: torch.Tensor, top_k: int = 1
    ) -> list[Candidate]:
        """Return top action candidates through the current scoring path.

        The returned candidates are action candidates and may include `is_over`.
        """
        return self.score_actions(out, bond_indices, top_k=top_k)

    def _evaluate_mol(
        self,
        embeddings: torch.Tensor,
        binary_logits: torch.Tensor,
        stop_logit: torch.Tensor,
        bond_indices: torch.Tensor,
        meta: SampleMeta,
        target_canon: str,
        history_embeddings: torch.Tensor | None = None,
        node_embeddings: torch.Tensor | None = None,
        bond_atom_indices: torch.Tensor | None = None,
    ) -> EvalResult:
        """Core per-molecule evaluation logic. Both evaluate_dataset and
        evaluate_single call this so the decision flow is identical.

        Args:
            embeddings: bond embeddings for this molecule (n_bonds, d) — post-cross-attention
            binary_logits: raw per-bond logits for this molecule (n_bonds, 1)
            stop_logit: scalar stop-action logit for this molecule
            bond_indices: bond atom index pairs for this molecule (n_bonds, 2)
            meta: sample metadata
            target_canon: canonicalized target SMILES for comparison
            history_embeddings: self-attention-only tokens for history storage (matches training)
            node_embeddings: (num_nodes, d_model) node embeddings from GAT for AtomSwapTypeHead
            bond_atom_indices: (num_bonds, 2) atom index pairs for each bond (for AtomSwapTypeHead)

        Returns:
            EvalResult for the full (step 1 + iteration) prediction.
        """
        if history_embeddings is None:
            history_embeddings = embeddings

        # Score best action for step 1.
        action_results = self._score_actions_for_mol(
            embeddings, binary_logits, stop_logit
        )
        if not action_results:
            return EvalResult(None, 0, False, StopReason.NO_CANDIDATES)
        best_action = action_results[0]
        meta.root_stop_prob = best_action.stop_prob

        if best_action.is_over:
            input_canon = canonicalize_smiles(meta.reactant_smiles)
            is_match = is_no_reaction_match(input_canon, target_canon)
            return EvalResult(
                predicted_product=meta.reactant_smiles,
                predicted_steps=0,
                is_match=is_match,
                stop_reason=StopReason.PREDICTED_IS_OVER,
            )

        b1_atoms = tuple(bond_indices[best_action.local_b1].tolist())
        b2_atoms = tuple(bond_indices[best_action.local_b2].tolist())

        swap_type = self._decode_swap_or_default(
            best_action.local_b1,
            best_action.local_b2,
            node_embeddings,
            bond_atom_indices,
            embeddings,
        )

        be_state = self.initial_state(meta)

        try:
            result = self.apply_step(be_state, b1_atoms, b2_atoms, swap_type)
            if not result.success:
                return EvalResult(None, 0, False, StopReason.STEP1_FAILED)
        except Exception as e:
            logger.debug("Step 1 exception: %s", e)
            return EvalResult(None, 0, False, StopReason.STEP1_FAILED)

        current_smi = result.product_smi
        be_state = result.state
        canon_1 = canonicalize_smiles(current_smi)
        is_match = check_smiles_match(canon_1, target_canon)

        trajectory_mode = getattr(self.config, "sample_mode", "step") == "trajectory"
        history_cache = []
        if trajectory_mode:
            initial_sample = self._build_sample_from_state(self.initial_state(meta), 0)
            if initial_sample is not None:
                history_cache.append((initial_sample[0], 0))
        elif self.config.enable_history_context:
            history_cache.append((history_embeddings.detach(), 0))

        # Run iteration (steps 2+)
        iter_result = self.run_iteration(
            be_state,
            target_canon,
            current_smi,
            is_match,
            history_cache=history_cache,
        )
        return EvalResult(
            predicted_product=iter_result.product_smi,
            predicted_steps=iter_result.predicted_steps,
            is_match=iter_result.is_match,
            stop_reason=iter_result.stop_reason,
        )

    def _expand_entry_candidates(
        self,
        entry: BeamEntry,
        fwd: ForwardResult,
        candidates: list[Candidate],
        step_idx: int,
        target_canon: str,
        trajectory_mode: bool,
        enable_history: bool,
    ) -> tuple[list[BeamEntry], list[TerminalEntry]]:
        """Expand one beam entry into successor entries and terminals.

        ``is_over`` candidates terminate the entry with its current product;
        pair candidates are applied to the BE state and become new beam
        entries (failed applications are dropped silently).
        """
        new_entries: list[BeamEntry] = []
        new_terminals: list[TerminalEntry] = []

        for cand in candidates:
            if cand.is_over:
                new_terminals.append(
                    TerminalEntry(
                        entry.score + cand.action_score,
                        entry.smi,
                        entry.n_steps,
                        entry.is_match,
                        StopReason.PREDICTED_IS_OVER,
                    )
                )
                continue
            try:
                step_result = self.apply_step(
                    fwd.state, cand.b1, cand.b2, cand.swap_type
                )
                if not step_result.success:
                    continue
                new_hist = list(entry.history)
                if trajectory_mode:
                    new_hist.append((fwd.sample, step_idx))
                elif enable_history:
                    new_hist.append((cand.embeddings, step_idx))
                new_smi = step_result.product_smi
                new_is_match = check_smiles_match(
                    canonicalize_smiles(new_smi), target_canon
                )
                new_entries.append(
                    BeamEntry(
                        score=entry.score + cand.action_score,
                        smi=new_smi,
                        state=step_result.state,
                        history=new_hist,
                        n_steps=entry.n_steps + 1,
                        is_match=new_is_match,
                    )
                )
            except Exception:
                logger.debug("Beam step exception", exc_info=True)
                continue

        return new_entries, new_terminals

    def run_iteration(
        self,
        state: BEState,
        target_canon: str,
        initial_smi: str,
        initial_is_match: bool,
        history_cache: list | None = None,
        initial_v_prob: float | None = None,
    ) -> IterationResult:
        """Run iterative steps. beam_width=1 is greedy, >1 is beam search.

        Returns IterationResult with best terminal candidate.

        ``is_match`` is reset only after applying a new pair step. If the model
        selects ``is_over``, the previous step's match status is preserved.
        """
        beam_width = self.config.eval_beam_width
        trajectory_mode = getattr(self.config, "sample_mode", "step") == "trajectory"
        enable_history = self.config.enable_history_context or trajectory_mode

        if history_cache is None:
            history_cache = []

        beam = [
            BeamEntry(
                score=0.0,
                smi=initial_smi,
                state=state,
                history=list(history_cache),
                n_steps=1,
                is_match=initial_is_match,
            )
        ]
        terminals: list[TerminalEntry] = []

        for step_idx in range(1, self.config.max_eval_steps):
            next_beam = []
            for entry in beam:
                if entry.smi is None:
                    terminals.append(
                        TerminalEntry(
                            entry.score,
                            entry.smi,
                            entry.n_steps,
                            entry.is_match,
                            StopReason.NO_PRODUCT,
                        )
                    )
                    continue

                fwd = self.forward_from_state(
                    entry.state,
                    history_cache=entry.history if enable_history else None,
                    step_index=step_idx,
                )
                if fwd is None:
                    terminals.append(
                        TerminalEntry(
                            entry.score,
                            entry.smi,
                            entry.n_steps,
                            entry.is_match,
                            StopReason.FORWARD_FAILED,
                        )
                    )
                    continue

                candidates = self.score_best_pair(
                    fwd.out, fwd.bond_indices, top_k=beam_width
                )
                if not candidates:
                    terminals.append(
                        TerminalEntry(
                            entry.score,
                            entry.smi,
                            entry.n_steps,
                            entry.is_match,
                            StopReason.NO_CANDIDATES,
                        )
                    )
                    continue

                new_entries, new_terminals = self._expand_entry_candidates(
                    entry,
                    fwd,
                    candidates,
                    step_idx,
                    target_canon,
                    trajectory_mode,
                    enable_history,
                )
                next_beam.extend(new_entries)
                terminals.extend(new_terminals)

            if not next_beam:
                break
            next_beam.sort(key=lambda e: e.score, reverse=True)
            beam = next_beam[:beam_width]

        # Collect remaining beam entries as terminals
        for entry in beam:
            terminals.append(
                TerminalEntry(
                    entry.score,
                    entry.smi,
                    entry.n_steps,
                    entry.is_match,
                    StopReason.MAX_STEPS_REACHED,
                )
            )

        if not terminals:
            return IterationResult(
                initial_smi, 1, initial_is_match, StopReason.MAX_STEPS_REACHED
            )

        terminals.sort(key=lambda t: t.score, reverse=True)
        best = terminals[0]
        return IterationResult(best.smi, best.n_steps, best.is_match, best.stop_reason)

    def _build_sample_from_state(
        self, state: BEState, step_index: int
    ) -> Optional[tuple[dict, dict, object]]:
        """Build a collate-ready sample dict from a BEState.

        Returns (sample_dict, reverse_lp_map, G) or None on failure.
        Extracted from forward_from_state to allow batching multiple samples.
        """
        try:
            G = _be_to_nx(state.be_matrix, state.atoms_identity, state.catalyst_flags)
            x, edge_index, edge_attr, bonds, edge_map = get_graph_features_nx(
                G,
                extended_features=self.config.extended_atom_features,
                canonical_rank=getattr(self.config, "use_canonical_rank", False),
            )
            if len(bonds) == 0:
                return None

            node_indices = sorted(list(G.nodes()))
            node_map_inv = {i: node_idx for i, node_idx in enumerate(node_indices)}
            x, edge_index, edge_attr, bonds, edge_map, lp_node_map = (
                augment_with_lp_edges(
                    x,
                    edge_index,
                    edge_attr,
                    bonds,
                    edge_map,
                    state.be_matrix,
                    node_map_inv,
                )
            )
            reverse_lp_map = {
                lp_idx: atom_idx for atom_idx, lp_idx in lp_node_map.items()
            }
            bonds_arr = np.array([sorted(b) for b in bonds])
            sample = {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "bonds": bonds_arr,
                "y": np.zeros(len(bonds_arr)),
                "is_over_target": 0,
                "step_index": step_index,
            }
            return sample, reverse_lp_map, G
        except Exception:
            logger.debug("_build_sample_from_state exception", exc_info=True)
            return None

    def _build_batched_history(
        self, history_caches: list[list], device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build padded history context and mask for a batch of molecules.

        Args:
            history_caches: Per-molecule lists of (embeddings, step) tuples.
            device: Target device.

        Returns:
            Tuple of (context ``[B, T, D]``, key-padding mask ``[B, T]`` with
            True marking padding), or (None, None) when history is disabled.
        """
        if not self.config.enable_history_context:
            return None, None

        B = len(history_caches)
        all_histories = []
        for cache in history_caches:
            if cache:
                hist = torch.cat([emb for emb, _ in cache], dim=0)
                all_histories.append(hist)
            else:
                all_histories.append(None)

        max_len = max((h.size(0) if h is not None else 0) for h in all_histories)
        if max_len == 0:
            return self.model.empty_history(B, device)

        d_model = next(h.size(1) for h in all_histories if h is not None)
        context = torch.zeros(B, max_len, d_model, device=device)
        mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
        for i, h in enumerate(all_histories):
            if h is not None:
                L = h.size(0)
                context[i, :L] = h.to(device)
                mask[i, :L] = False
        return context, mask

    def _batched_forward(
        self, samples: list[dict], history_caches: list[list]
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Run batched model forward for multiple sample dicts.

        Returns:
            Tuple of (model output dict, bond atom-index pairs, atom batch
            index vector).
        """
        batch = collate_fn(samples, enable_matrix_output=False)
        x_b, batch_idx, ei_b, ea_b, bond_idx, step_idx_b = self._collated_to_device(
            batch
        )

        history_context, history_mask = self._build_batched_history(
            history_caches, self.device
        )

        out = self._model_forward(
            x_b,
            batch_idx,
            ei_b,
            ea_b,
            bond_idx,
            history_context,
            history_mask,
            step_idx_b,
        )
        return out, out.get("bond_indices_out", bond_idx), batch_idx

    @staticmethod
    def _slice_molecule(
        out: dict,
        bond_index_source: torch.Tensor,
        bond_offsets: torch.Tensor,
        atom_offsets: torch.Tensor,
        k: int,
    ) -> _MolSlice:
        """Slice molecule ``k``'s tensors out of a batched model output."""
        embeddings = out["embeddings"]
        history_embeddings = out.get("history_embeddings", embeddings)
        bi_out = out.get("bond_indices_out")
        b_start, b_end = bond_offsets[k].item(), bond_offsets[k + 1].item()
        a_start = atom_offsets[k].item()
        return _MolSlice(
            embeddings=embeddings[b_start:b_end],
            history_embeddings=history_embeddings[b_start:b_end],
            binary_logits=out["logits_binary"][b_start:b_end],
            stop_logit=out["logits_stop"].squeeze(-1)[k],
            local_bond_indices=bond_index_source[b_start:b_end] - a_start,
            bond_atom_indices=bi_out[b_start:b_end] if bi_out is not None else None,
        )

    @staticmethod
    def _bond_atom_tuples(
        local_bond_indices: torch.Tensor, local_b1: int, local_b2: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Atom index tuples of the two selected bonds, local to the molecule."""
        b1 = (
            local_bond_indices[local_b1, 0].item(),
            local_bond_indices[local_b1, 1].item(),
        )
        b2 = (
            local_bond_indices[local_b2, 0].item(),
            local_bond_indices[local_b2, 1].item(),
        )
        return b1, b2

    @torch.no_grad()
    def _evaluate_batch_greedy(
        self,
        out: dict,
        metadata: list,
        collated_bond_indices: torch.Tensor,
        collated_batch_idx: torch.Tensor,
    ) -> list[tuple[int, EvalResult]]:
        """Batched greedy evaluation for all molecules in one DataLoader batch.

        Step 0 forward is already done. Scores and applies step 1 for every
        molecule, advances the surviving molecules in batched iterations for
        steps 2+, and finally re-checks each finished product against its
        target.

        Returns:
            (batch index, EvalResult) pairs, one per molecule.
        """
        results: list[tuple[int, EvalResult]] = []
        active = self._start_active_molecules(
            out, metadata, collated_bond_indices, collated_batch_idx, results
        )

        for step_idx in range(1, self.config.max_eval_steps):
            if not active:
                break
            active = self._advance_active_molecules(active, step_idx, results)

        for am in active:
            results.append(
                (
                    am.batch_idx,
                    EvalResult(
                        am.current_smi,
                        am.n_steps,
                        am.is_match,
                        StopReason.MAX_STEPS_REACHED,
                    ),
                )
            )

        self._finalize_pending_matches(results, metadata)
        return results

    def _start_active_molecules(
        self,
        out: dict,
        metadata: list,
        collated_bond_indices: torch.Tensor,
        collated_batch_idx: torch.Tensor,
        results: list[tuple[int, EvalResult]],
    ) -> list[ActiveMolecule]:
        """Score and apply step 1 for every molecule in the batch.

        Molecules that stop, fail, or have no candidates get their EvalResult
        appended to ``results``; the rest become ActiveMolecule entries for
        the iterative steps.
        """
        bi_out = out.get("bond_indices_out")
        bond_index_source = bi_out if bi_out is not None else collated_bond_indices
        # Pre-compute per-molecule boundary indices (avoids repeated == i / torch.where)
        bond_batch_mapping = collated_batch_idx[bond_index_source[:, 0]]
        n_mols = len(metadata)
        bond_offsets = self._segment_offsets(bond_batch_mapping, n_mols)
        atom_offsets = self._segment_offsets(collated_batch_idx, n_mols)
        node_embeddings = out.get("node_embeddings")
        enable_history = self.config.enable_history_context

        active: list[ActiveMolecule] = []
        for i, meta in enumerate(metadata):
            mol = self._slice_molecule(
                out, bond_index_source, bond_offsets, atom_offsets, i
            )
            ar = self._score_actions_for_mol(
                mol.embeddings, mol.binary_logits, mol.stop_logit
            )
            if not ar:
                results.append(
                    (i, EvalResult(None, 0, False, StopReason.NO_CANDIDATES))
                )
                continue

            best = ar[0]
            meta.root_stop_prob = best.stop_prob
            target_canon = canonicalize_smiles(meta.target_smiles)
            if best.is_over:
                input_canon = canonicalize_smiles(meta.reactant_smiles)
                results.append(
                    (
                        i,
                        EvalResult(
                            meta.reactant_smiles,
                            0,
                            is_no_reaction_match(input_canon, target_canon),
                            StopReason.PREDICTED_IS_OVER,
                        ),
                    )
                )
                continue

            b1_atoms, b2_atoms = self._bond_atom_tuples(
                mol.local_bond_indices, best.local_b1, best.local_b2
            )
            swap_type = self._decode_swap_or_default(
                best.local_b1,
                best.local_b2,
                node_embeddings,
                mol.bond_atom_indices,
                mol.embeddings,
            )

            be_state = self.initial_state(meta)
            try:
                sr = self.apply_step(be_state, b1_atoms, b2_atoms, swap_type)
            except Exception:
                sr = None
            if sr is None or not sr.success:
                results.append((i, EvalResult(None, 0, False, StopReason.STEP1_FAILED)))
                continue

            hcache = []
            if enable_history:
                hcache.append((mol.history_embeddings.detach(), 0))

            step1_is_match = check_smiles_match(
                canonicalize_smiles(sr.product_smi), target_canon
            )

            active.append(
                ActiveMolecule(
                    batch_idx=i,
                    meta=meta,
                    target_canon=target_canon,
                    state=sr.state,
                    current_smi=sr.product_smi,
                    is_match=step1_is_match,
                    history_cache=hcache,
                    n_steps=1,
                    score=best.score,
                )
            )

        return active

    def _advance_active_molecules(
        self,
        active: list[ActiveMolecule],
        step_idx: int,
        results: list[tuple[int, EvalResult]],
    ) -> list[ActiveMolecule]:
        """Run one batched iteration step over all still-active molecules.

        Builds samples from each BE state, forwards them in one batch, then
        scores and applies the best action per molecule. Molecules that stop,
        fail, or exhaust their candidates move to ``results``; the rest are
        returned as the next round's active set.
        """
        samples: list[dict] = []
        build_data: list[tuple] = []
        valid: list[int] = []

        for j, am in enumerate(active):
            built = self._build_sample_from_state(am.state, step_idx)
            if built is None:
                results.append(
                    (
                        am.batch_idx,
                        EvalResult(
                            am.current_smi,
                            am.n_steps,
                            am.is_match,
                            StopReason.FORWARD_FAILED,
                        ),
                    )
                )
            else:
                samples.append(built[0])
                build_data.append(built)
                valid.append(j)

        if not samples:
            return []

        fwd_out, fwd_bi, fwd_bidx = self._batched_forward(
            samples,
            [active[j].history_cache for j in valid],
        )

        fwd_bio = fwd_out.get("bond_indices_out")
        bond_index_source = fwd_bio if fwd_bio is not None else fwd_bi
        bond_batch_mapping = fwd_bidx[bond_index_source[:, 0]]
        n_fwd = len(valid)
        bond_offsets = self._segment_offsets(bond_batch_mapping, n_fwd)
        atom_offsets = self._segment_offsets(fwd_bidx, n_fwd)
        node_embeddings = fwd_out.get("node_embeddings")
        enable_history = self.config.enable_history_context

        next_active: list[ActiveMolecule] = []
        for k, j in enumerate(valid):
            am = active[j]
            _, reverse_lp_map, _ = build_data[k]
            mol = self._slice_molecule(
                fwd_out, bond_index_source, bond_offsets, atom_offsets, k
            )

            ar = self._score_actions_for_mol(
                mol.embeddings, mol.binary_logits, mol.stop_logit
            )
            if not ar:
                results.append(
                    (
                        am.batch_idx,
                        EvalResult(
                            am.current_smi,
                            am.n_steps,
                            am.is_match,
                            StopReason.NO_CANDIDATES,
                        ),
                    )
                )
                continue

            cand = ar[0]
            if cand.is_over:
                results.append(
                    (
                        am.batch_idx,
                        EvalResult(
                            am.current_smi,
                            am.n_steps,
                            am.is_match,
                            StopReason.PREDICTED_IS_OVER,
                        ),
                    )
                )
                continue

            b1, b2 = self._bond_atom_tuples(
                mol.local_bond_indices, cand.local_b1, cand.local_b2
            )
            swap_type = self._decode_swap_or_default(
                cand.local_b1,
                cand.local_b2,
                node_embeddings,
                mol.bond_atom_indices,
                mol.embeddings,
            )

            new_state = self._state_with_lp_map(am.state, reverse_lp_map)
            try:
                sr = self.apply_step(new_state, b1, b2, swap_type)
            except Exception:
                sr = None
            if sr is None or not sr.success:
                results.append(
                    (
                        am.batch_idx,
                        EvalResult(
                            am.current_smi,
                            am.n_steps,
                            am.is_match,
                            StopReason.MAX_STEPS_REACHED,
                        ),
                    )
                )
                continue

            new_history = list(am.history_cache)
            if enable_history:
                new_history.append((mol.history_embeddings.detach(), step_idx))

            next_active.append(
                ActiveMolecule(
                    batch_idx=am.batch_idx,
                    meta=am.meta,
                    target_canon=am.target_canon,
                    state=sr.state,
                    current_smi=sr.product_smi,
                    is_match=check_smiles_match(
                        canonicalize_smiles(sr.product_smi), am.target_canon
                    ),
                    history_cache=new_history,
                    n_steps=am.n_steps + 1,
                    score=am.score + cand.score,
                )
            )

        return next_active

    @staticmethod
    def _finalize_pending_matches(
        results: list[tuple[int, EvalResult]], metadata: list
    ) -> None:
        """Re-check every finished product against its target, in place.

        Deferred match check — one canonicalization per molecule at
        finalization upgrades ``is_match`` for products that match after all.
        """
        for idx in range(len(results)):
            mol_i, er = results[idx]
            if er.predicted_product is not None:
                target_canon = canonicalize_smiles(metadata[mol_i].target_smiles)
                pred_canon = canonicalize_smiles(er.predicted_product)
                if check_smiles_match(pred_canon, target_canon):
                    results[idx] = (
                        mol_i,
                        EvalResult(
                            er.predicted_product,
                            er.predicted_steps,
                            True,
                            er.stop_reason,
                        ),
                    )

    def evaluate_dataset(
        self,
        json_path: str,
        save_csv: str | None = None,
        require_success: bool = True,
        all_entries: bool = False,
    ) -> float:
        """Evaluate the full dataset and print the accuracy summary.

        Loads the eval data, runs the (batched greedy or sequential)
        evaluation over all samples, prints step-count/yield breakdowns, and
        optionally writes per-sample CSV results.

        Args:
            json_path: Path to the JSON/JSONL eval data.
            save_csv: Optional CSV output path for per-sample results.
            require_success: Only evaluate entries marked successful.
            all_entries: Evaluate every entry of the file (see
                ``_load_all_eval_entries``); overrides ``require_success`` and
                ``test_limit``.

        Returns:
            Top-1 accuracy over all samples.
        """
        print(f"\nLoading test data from {json_path}...")
        eval_rng = random.Random(self.eval_seed)
        if all_entries:
            data = _load_all_eval_entries(json_path)
        else:
            data = _load_eval_data(
                json_path,
                self.config.test_limit,
                require_success=require_success,
                enable_yield0_negatives=self.enable_yield0_negatives,
                rng=eval_rng,
            )
        data = strip_atom_mappings_from_data(data)
        print(f"Processing {len(data)} samples...")

        dataset = InferenceDataset(
            data,
            extended_features=self.config.extended_atom_features,
            canonical_rank=getattr(self.config, "use_canonical_rank", False),
            enable_yield0_negatives=self.enable_yield0_negatives,
        )

        print(
            f"Eval settings: max_eval_steps={self.config.max_eval_steps}, "
            f"beam_width={self.config.eval_beam_width}, "
            f"sample_mode={getattr(self.config, 'sample_mode', 'step')}, "
            f"pair_scoring_mode={self.config.pair_scoring_mode}, "
            f"pair_selection={self.config.pair_selection_mode.value}, "
            f"temperature={self.config.pair_selection_temperature}, "
            f"softmax_topk={self.config.pair_softmax_topk}, "
            f"yield0_negatives={self.enable_yield0_negatives}"
        )

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=inference_collate,
            num_workers=4,
            pin_memory=True,
        )

        self.model.eval()

        stats = _DatasetStats()
        csv_rows: list[dict] = []
        unknown_mech_rows: list[dict] = []  # predictions for entries without mechanisms

        for batch_tuple in tqdm(loader, desc="Evaluating (Batched)"):
            if batch_tuple is None:
                continue

            collated_tensors, metadata = batch_tuple
            out, batch_idx, bond_indices = self._forward_collated_batch(
                collated_tensors, metadata
            )

            if self.should_use_batched_greedy_eval():
                batch_results = self._evaluate_batch_greedy(
                    out,
                    metadata,
                    bond_indices,
                    batch_idx,
                )
            else:
                batch_results = self._evaluate_batch_sequential(
                    out, metadata, bond_indices, batch_idx
                )

            for i, eval_result in batch_results:
                meta = metadata[i]
                stats.update(meta, eval_result)

                if meta.num_steps == 0:
                    unknown_mech_rows.append(
                        {
                            "reactant": meta.reactant_smiles,
                            "target": meta.target_smiles or "",
                            "prediction": eval_result.predicted_product or "",
                            "match": eval_result.is_match,
                            "predicted_steps": eval_result.predicted_steps,
                            "stop_reason": eval_result.stop_reason.name
                            if eval_result.stop_reason
                            else "",
                        }
                    )

                if save_csv:
                    csv_rows.append(
                        self._build_csv_row(
                            meta,
                            data,
                            eval_result.predicted_product,
                            eval_result.is_match,
                            meta.num_steps,
                            eval_result.predicted_steps,
                        )
                    )

        acc = self._print_dataset_summary(stats)
        if all_entries:
            acc = stats.correct_top1 / max(len(data), 1)
            print(
                f"Top-1 Accuracy over all {len(data)} entries "
                f"(entries that cannot be encoded count as wrong): {acc:.4f}"
            )
        self._write_results_csv(save_csv, csv_rows)
        self._write_unknown_mech_csv(save_csv, unknown_mech_rows)
        return acc

    def _forward_collated_batch(
        self, collated_tensors: tuple, metadata: list
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Run the step-0 model forward for one collated DataLoader batch.

        Returns:
            Tuple of (model output dict, atom batch index vector, bond
            atom-index pairs).
        """
        (
            x_batch,
            batch_idx,
            edge_idx,
            edge_attr,
            bond_indices,
            step_index,
        ) = self._collated_to_device(collated_tensors)

        # Build empty history context for step-0 batch forward
        history_context = None
        history_mask = None
        if self.config.enable_history_context:
            batch_size = int(batch_idx.max().item()) + 1
            history_context, history_mask = self.model.empty_history(
                batch_size, self.device
            )

        trajectory_ids = None
        if self.uses_trajectory_mode():
            trajectory_ids = torch.arange(
                len(metadata), dtype=torch.long, device=self.device
            )

        out = self._model_forward(
            x_batch,
            batch_idx,
            edge_idx,
            edge_attr,
            bond_indices,
            history_context,
            history_mask,
            step_index,
            trajectory_ids=trajectory_ids,
        )
        return out, batch_idx, bond_indices

    def _evaluate_batch_sequential(
        self,
        out: dict,
        metadata: list,
        bond_indices: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> list[tuple[int, EvalResult]]:
        """Per-molecule (non-batched) evaluation via ``_evaluate_mol``.

        Used for beam search and trajectory mode, where the batched greedy
        path does not apply.
        """
        embeddings = out["embeddings"]
        history_embeddings = out.get("history_embeddings", embeddings)
        binary_logits = out["logits_binary"]
        stop_logits = out["logits_stop"].squeeze(-1)
        bi_out = out.get("bond_indices_out")
        bond_index_source = bi_out if bi_out is not None else bond_indices
        bond_batch_mapping = batch_idx[bond_index_source[:, 0]]

        batch_results: list[tuple[int, EvalResult]] = []
        for i, meta in enumerate(metadata):
            target_canon = canonicalize_smiles(meta.target_smiles)
            mol_mask = bond_batch_mapping == i
            mol_bonds_global_idx = torch.where(mol_mask)[0]
            min_atom_idx = torch.where(batch_idx == i)[0].min().item()
            eval_result = self._evaluate_mol(
                embeddings[mol_mask],
                binary_logits[mol_mask],
                stop_logits[i],
                bond_index_source[mol_bonds_global_idx] - min_atom_idx,
                meta,
                target_canon,
                history_embeddings=history_embeddings[mol_mask],
                node_embeddings=out.get("node_embeddings"),
                bond_atom_indices=bond_index_source[mol_bonds_global_idx],
            )
            batch_results.append((i, eval_result))
        return batch_results

    def _print_dataset_summary(self, stats: _DatasetStats) -> float:
        """Print accuracy breakdowns and return the overall top-1 accuracy."""
        # Separate known mechanisms from entries with no recorded mechanism.
        unknown_total = stats.step_count_total.get(0, 0)
        unknown_correct = stats.step_count_correct.get(0, 0)
        known_total = stats.total_samples - unknown_total
        known_correct = stats.correct_top1 - unknown_correct

        acc = stats.correct_top1 / max(stats.total_samples, 1)
        print(f"\n--- Ranking Evaluation ---")
        print(f"Total Samples: {stats.total_samples}")
        print(
            f"Top-1 Accuracy (w/ Iteration): {acc:.4f} ({stats.correct_top1}/{stats.total_samples})"
        )

        if unknown_total > 0:
            known_acc = known_correct / max(known_total, 1)
            unknown_acc = unknown_correct / max(unknown_total, 1)
            print(
                f"  Known Mechanism:   {known_acc:.4f} ({known_correct}/{known_total})"
            )
            print(
                f"  Unknown Mechanism: {unknown_acc:.4f} ({unknown_correct}/{unknown_total})"
            )

        if self.enable_yield0_negatives and stats.yield_bucket_total:
            self._print_yield0_metrics(stats)

        single_total = stats.step_count_total.get(1, 0)
        single_correct = stats.step_count_correct.get(1, 0)
        single_acc = single_correct / max(single_total, 1)
        print(f"  1-Step Accuracy: {single_acc:.4f} ({single_correct}/{single_total})")

        for n in sorted(k for k in stats.step_count_total if k >= 2):
            n_total = stats.step_count_total[n]
            n_correct = stats.step_count_correct.get(n, 0)
            n_acc = n_correct / max(n_total, 1)
            print(f"  {n}-Step Accuracy: {n_acc:.4f} ({n_correct}/{n_total})")

        multi_total = sum(v for k, v in stats.step_count_total.items() if k >= 2)
        multi_correct = sum(v for k, v in stats.step_count_correct.items() if k >= 2)
        if multi_total > 0:
            multi_acc = multi_correct / max(multi_total, 1)
            print(
                f"  Multi-Step Total: {multi_acc:.4f} ({multi_correct}/{multi_total})"
            )

        if stats.step1_apply_fail > 0:
            print(f"  Step-1 Apply Failures: {stats.step1_apply_fail}")

        return acc

    def _print_yield0_metrics(self, stats: _DatasetStats) -> None:
        """Print yield-0 detection metrics (confusion counts, precision/recall/F1)."""
        tp = stats.yield_zero_confusion.get("tp", 0)
        fp = stats.yield_zero_confusion.get("fp", 0)
        tn = stats.yield_zero_confusion.get("tn", 0)
        fn = stats.yield_zero_confusion.get("fn", 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        false_positive_rate = fp / max(fp + tn, 1)
        false_negative_rate = fn / max(fn + tp, 1)

        for line in _build_yield_metrics_summary(
            self.enable_yield0_negatives,
            stats.yield_bucket_total,
            stats.yield_bucket_correct,
            stats.yield_zero_confusion,
        ):
            print(line)

        print("  Yield=0 Detection (predicted as no reaction):")
        print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    FPR:       {false_positive_rate:.4f}")
        print(f"    FNR:       {false_negative_rate:.4f}")

    @staticmethod
    def _write_results_csv(save_csv: str | None, csv_rows: list[dict]) -> None:
        """Write per-sample prediction rows to ``save_csv`` when requested."""
        if not (save_csv and csv_rows):
            return
        fieldnames = [
            "entry_idx",
            "reaction_smiles",
            "reaction_smiles_mapped",
            "target_product",
            "predicted_product",
            "match",
            "root_stop_prob",
            "num_steps_true",
            "num_steps_predicted",
            "original_mechanisms",
        ]
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nResults saved to {save_csv} ({len(csv_rows)} rows)")

    @staticmethod
    def _write_unknown_mech_csv(
        save_csv: str | None, unknown_mech_rows: list[dict]
    ) -> None:
        """Write predictions for entries without recorded mechanisms to CSV."""
        if not unknown_mech_rows:
            return
        unknown_path = (
            save_csv.replace(".csv", "_unknown_mech.csv")
            if save_csv
            else "unknown_mechanism_predictions.csv"
        )
        fieldnames = [
            "reactant",
            "target",
            "prediction",
            "match",
            "predicted_steps",
            "stop_reason",
        ]
        with open(unknown_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unknown_mech_rows)
        print(
            f"Unknown mechanism predictions saved to {unknown_path} ({len(unknown_mech_rows)} rows)"
        )

    @staticmethod
    def _build_csv_row(
        meta: SampleMeta,
        data: list[dict],
        predicted_product: str | None,
        is_match: bool,
        num_steps: int,
        predicted_steps: int,
    ) -> dict:
        """Build one per-sample CSV result row."""
        orig_entry = data[meta.entry_idx] if meta.entry_idx is not None else {}
        return {
            "entry_idx": meta.entry_idx,
            "reaction_smiles": orig_entry.get("reaction_smiles", ""),
            "reaction_smiles_mapped": orig_entry.get("reaction_smiles_mapped", ""),
            "target_product": meta.target_smiles,
            "predicted_product": predicted_product or "",
            "match": int(is_match),
            "root_stop_prob": meta.root_stop_prob,
            "num_steps_true": num_steps,
            "num_steps_predicted": predicted_steps,
            "original_mechanisms": json.dumps(orig_entry.get("mechanisms", [])),
        }

    def evaluate_single(
        self,
        entry: dict | None = None,
        smiles: str | None = None,
        target_product: str | None = None,
        top_k: int = 5,
        quiet: bool = False,
    ):
        """Full single-sample evaluation.

        Runs _evaluate_mol for the actual decision, then displays top-K
        pair rankings and iterative step details (unless quiet=True).

        Args:
            entry: Eval data entry; built from ``smiles`` when omitted.
            smiles: Reaction SMILES (with or without ``>>``) used when no
                entry is given.
            target_product: Target product SMILES; defaults to the entry's
                product side.
            top_k: Number of top actions shown per step.
            quiet: Suppress all printing and return the bare EvalResult.

        Returns:
            EvalResult when quiet=True, else (mapped_educt, product_smi, is_correct).
        """

        def _fail(message: str):
            if not quiet:
                print(message)
            return None if quiet else (None, None, False)

        if entry is None and smiles is not None:
            if ">>" in smiles:
                entry = {"reaction_smiles": smiles, "success": True}
            else:
                entry = {"reaction_smiles": smiles + ">>", "success": True}
        if entry is None:
            return _fail("No entry or SMILES provided.")

        entry = _maybe_rewrite_yield0_entry(
            entry, enable_yield0_negatives=self.enable_yield0_negatives
        )

        rxn = entry.get("reaction_smiles", "")
        if ">>" not in rxn:
            return _fail("Invalid reaction SMILES format.")
        reactant_part, entry_product = rxn.split(">>", 1)
        if target_product is None and entry_product:
            target_product = entry_product

        dataset = InferenceDataset(
            [entry],
            extended_features=self.config.extended_atom_features,
            canonical_rank=getattr(self.config, "use_canonical_rank", False),
            enable_yield0_negatives=self.enable_yield0_negatives,
        )
        sample = dataset[0]
        if sample is None:
            return _fail("Failed to process sample.")

        batch_tuple = inference_collate([sample])
        if batch_tuple is None:
            return _fail("Failed to collate sample.")

        collated_tensors, metadata = batch_tuple
        out, _, bond_indices = self._forward_collated_batch(
            collated_tensors, metadata
        )

        meta = metadata[0]
        mol = meta.mol
        target_canon = canonicalize_smiles(target_product) if target_product else None
        is_no_reaction_target = (
            self.enable_yield0_negatives
            and _is_no_reaction_target(meta.reactant_smiles, target_product)
        )

        display_mol = Chem.Mol(mol)
        has_mapping = any(a.GetAtomMapNum() != 0 for a in mol.GetAtoms())
        if not has_mapping:
            for atom in display_mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())

        embeddings = out["embeddings"]
        history_embeddings = out.get("history_embeddings", embeddings)
        binary_logits = out["logits_binary"]
        stop_logit = out["logits_stop"][0]

        eval_result = self._evaluate_mol(
            embeddings,
            binary_logits,
            stop_logit,
            out.get("bond_indices_out", bond_indices),
            meta,
            target_canon,
            history_embeddings=history_embeddings,
            node_embeddings=out.get("node_embeddings"),
            bond_atom_indices=out.get("bond_indices_out"),
        )

        if quiet:
            return eval_result

        be_state = self.initial_state(meta)
        initial_actions = self.score_actions(
            out, out.get("bond_indices_out", bond_indices), top_k=max(top_k, 1)
        )

        print(f"\n--- Initial Step Analysis for: {meta.reactant_smiles} ---")
        if initial_actions and initial_actions[0].is_over:
            print(
                f"Top action: is_over (prob={initial_actions[0].action_prob * 100:.2f}%, "
                f"score={initial_actions[0].action_score:.4f})"
            )
            if is_no_reaction_target:
                print(f"Top action matches no-reaction target? {eval_result.is_match}")

        display = self._display_top_k(
            out,
            out.get("bond_indices_out", bond_indices),
            mol,
            be_state,
            target_canon,
            top_k=top_k,
        )

        stopped = eval_result.stop_reason == StopReason.PREDICTED_IS_OVER
        if eval_result.predicted_steps + stopped > 1 and display.product_smi is not None:
            self._display_iterative_steps(
                eval_result, display, meta, mol, target_canon, history_embeddings, top_k
            )

        print(f"\n--- Final Result ---")
        print(f"Steps applied: {eval_result.predicted_steps}")
        print(f"Stop reason: {eval_result.stop_reason.value}")
        final_product = eval_result.predicted_product
        if final_product is not None:
            print(
                f"Pred Product: {Chem.MolToSmiles(display_mol, includeDativeBonds=False)}"
                f">>{final_product.replace('->', '-').replace('<-', '-')}"
            )
        else:
            print("Pred Product: (No valid product)")
        if is_no_reaction_target:
            print(f"No-Reaction Target? {is_no_reaction_target}")
            print(
                "Immediate is_over correct? "
                f"{eval_result.stop_reason == StopReason.PREDICTED_IS_OVER and eval_result.predicted_steps == 0 and eval_result.is_match}"
            )
        print(f"Top-1 Correct? {eval_result.is_match}")

        return Chem.MolToSmiles(display_mol), final_product, eval_result.is_match

    def _display_iterative_steps(
        self,
        eval_result: EvalResult,
        display: DisplayResult,
        meta: SampleMeta,
        mol: Chem.Mol,
        target_canon: str | None,
        history_embeddings: torch.Tensor,
        top_k: int,
    ) -> None:
        """Print top-K analysis for each iterative step beyond step 1.

        Replays the prediction from the step-1 display state, printing the
        per-step rankings until the model stops or a step fails. Display
        only — the actual decision was already made by ``_evaluate_mol``.
        """
        trajectory_mode = self.uses_trajectory_mode()
        history_cache = []
        if trajectory_mode:
            initial_sample = self._build_sample_from_state(self.initial_state(meta), 0)
            if initial_sample is not None:
                history_cache.append((initial_sample[0], 0))
        elif self.config.enable_history_context:
            history_cache.append((history_embeddings.detach(), 0))

        current_state = display.state
        current_smi = display.product_smi

        # One step past the applied ones shows the ranking in which is_over won.
        stopped = eval_result.stop_reason == StopReason.PREDICTED_IS_OVER
        for step in range(1, eval_result.predicted_steps + stopped):
            print(
                f"\n>>> Continuing to step {step + 1} from current product: {current_smi}"
            )

            fwd = self.forward_from_state(
                current_state,
                history_cache=history_cache
                if (trajectory_mode or self.config.enable_history_context)
                else None,
                step_index=step,
            )
            if fwd is None:
                print(
                    f"\n>>> Not continuing to step {step + 1}: Forward pass failed"
                )
                break

            step_mol = nx_to_mol(fwd.G, remove_virtual=True)
            if step_mol:
                step_mol = Chem.AddHs(step_mol)
            else:
                step_mol = mol

            indent = "   " * step
            print(f"\n{indent}--- Iterative Step {step + 1} ---")

            step_display = self._display_top_k(
                fwd.out,
                fwd.bond_indices,
                step_mol,
                fwd.state,
                target_canon,
                top_k=top_k,
                indent=indent,
            )

            if step_display.product_smi is None:
                print(f"{indent}>>> Not continuing: No valid product")
                break

            if step_display.is_over:
                print(f"{indent}>>> Stopping: Model selected is_over")
                break

            if trajectory_mode:
                history_cache.append((fwd.sample, step))
            elif self.config.enable_history_context:
                step_hist_emb = fwd.out.get(
                    "history_embeddings", fwd.out["embeddings"]
                )
                history_cache.append((step_hist_emb.detach(), step))

            current_smi = step_display.product_smi
            current_state = step_display.state

            print(f"{indent}Product: {current_smi}")
            if step_display.is_correct:
                print(f"{indent}\033[92mMatch found!\033[0m")

    def _display_top_k(
        self,
        out: dict,
        bond_indices: torch.Tensor,
        mol: Chem.Mol,
        state: BEState,
        target_canon: str | None,
        top_k: int = 5,
        indent: str = "",
    ) -> DisplayResult:
        """Score top actions and display them with match indicators."""
        embeddings = out["embeddings"]
        action_results = self._score_actions_for_mol(
            embeddings,
            out["logits_binary"],
            out["logits_stop"][0],
            top_k=max(top_k, 1),
        )

        use_pair_bt = self._use_pair_bond_type()

        def get_lbl(idx):
            if idx >= mol.GetNumAtoms():
                return "LP" if idx > mol.GetNumAtoms() else "HEAD"
            atom = mol.GetAtomWithIdx(idx)
            map_num = atom.GetAtomMapNum()
            if map_num:
                return f"{atom.GetSymbol()}:{map_num}"
            return f"{atom.GetSymbol()}#{idx}"

        print(f"{indent}{'Rank':<5} {'Score':<8} {'Prob':<8} {'Action':<25} {'Match?'}")
        print(f"{indent}" + "-" * 70)

        final_product = None
        final_state = state
        is_correct_top1 = False
        current_smi = canonicalize_smiles(
            Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
        )

        for rank_pos, action in enumerate(action_results[:top_k]):
            score = action.score

            if action.is_over:
                match = (
                    check_smiles_match(current_smi, target_canon)
                    if target_canon
                    else False
                )
                match_str = "\u2705" if match else "\u274c"
                color_code = "\033[92m" if match and rank_pos == 0 else ""
                reset_code = "\033[0m"
                if rank_pos == 0:
                    final_product = current_smi
                    is_correct_top1 = match
                row = (
                    f"{rank_pos + 1:<5} {score:<8.4f} {action.action_prob * 100.0:<8.2f} "
                    f"{'is_over':<25} {match_str}"
                )
                print(f"{indent}{color_code}{row}{reset_code}")
                continue

            i_idx = action.local_b1
            j_idx = action.local_b2
            b1_atoms = tuple(bond_indices[i_idx].tolist())
            b2_atoms = tuple(bond_indices[j_idx].tolist())

            if use_pair_bt:
                swap_type, swap_dist = decode_swap_type(
                    self.model,
                    i_idx,
                    j_idx,
                    node_embeddings=out.get("node_embeddings"),
                    bond_atom_indices=out.get("bond_indices_out"),
                    bond_embeddings=embeddings,
                )
            else:
                swap_type = 0
                swap_dist = ""

            try:
                result = self.apply_step(state, b1_atoms, b2_atoms, swap_type)
                if not result.success:
                    continue

                match_str = ""
                color_code = ""
                reset_code = "\033[0m"

                if target_canon:
                    pred_canon = canonicalize_smiles(result.product_smi)
                    match = check_smiles_match(pred_canon, target_canon)
                    match_str = "\u2705" if match else "\u274c"
                    if match:
                        color_code = "\033[92m"
                        if rank_pos == 0:
                            is_correct_top1 = True
                    elif rank_pos == 0:
                        color_code = "\033[91m"

                if rank_pos == 0:
                    final_product = result.product_smi
                    final_state = result.state

                b1_str = f"{get_lbl(b1_atoms[0])}-{get_lbl(b1_atoms[1])}"
                b2_str = f"{get_lbl(b2_atoms[0])}-{get_lbl(b2_atoms[1])}"
                row = (
                    f"{rank_pos + 1:<5} {score:<8.4f} {action.action_prob * 100.0:<8.2f} "
                    f"{(b1_str + ' | ' + b2_str):<25} {match_str} {swap_type} {swap_dist}"
                )
                print(f"{indent}{color_code}{row}{reset_code}")
            except Exception as e:
                logger.debug("Display pair error: %s", e)

        return DisplayResult(
            product_smi=final_product,
            state=final_state,
            is_correct=is_correct_top1,
            is_over=action_results[0].is_over if action_results else False,
        )


    def evaluate_random_sample(
        self, test_path: str, test_entries: list[dict] | None = None
    ) -> list[dict] | None:
        """Pick one random valid entry from the test data and evaluate it verbosely.

        Args:
            test_path: Path to the JSON/JSONL test data.
            test_entries: Already-loaded entries; avoids re-reading the file.

        Returns:
            The loaded test entries (for reuse on subsequent calls), or None
            for JSONL sampling.
        """
        if test_path.endswith(".jsonl") and test_entries is None:
            # Read a single random valid entry without loading entire file
            entry = _sample_one_jsonl_entry(
                test_path,
                enable_yield0_negatives=self.enable_yield0_negatives,
                rng=self.sample_rng,
            )
        else:
            if test_entries is None:
                with open(test_path, "r") as f:
                    test_entries = json.load(f)
            while True:
                entry = _maybe_rewrite_yield0_entry(
                    self.sample_rng.choice(test_entries),
                    enable_yield0_negatives=self.enable_yield0_negatives,
                )
                if is_entry_successful(entry):
                    break

        if entry is None:
            print("Could not find a valid entry.")
            return test_entries

        print(f"\nChosen Reaction: {entry.get('reaction_smiles', '')}")
        self.evaluate_single(entry=entry)
        return test_entries
