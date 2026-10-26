"""Evaluation package for BondGAT ranking models.

Re-exports the public API of the evaluation submodules: constants, results,
utils, be_ops, model_io, data, evaluator, and cli.
"""

from bond_match.evaluation.constants import (
    logger,
    PT,
    _original_mol_to_smiles,
    _uncharger,
    _normalizer,
    _tautomer_enumerator,
    _normalize_cache,
    custom_mol_to_smiles,
)
from bond_match.evaluation.results import (
    StopReason,
    BEState,
    SampleMeta,
    StepResult,
    ForwardResult,
    Candidate,
    PairResult,
    BeamEntry,
    TerminalEntry,
    IterationResult,
    EvalResult,
    ActiveMolecule,
    DisplayResult,
)
from bond_match.evaluation.utils import (
    seed_everything,
    same_molecule_different_protonation_or_tautomer,
    remove_atom_mapping,
    canonicalize_smiles,
    strip_atom_mappings_from_data,
    _normalize_smiles,
    check_smiles_match,
    _is_no_reaction_target,
    _rewrite_product_as_reactant,
    _maybe_rewrite_yield0_entry,
    is_entry_successful,
    _build_yield_metrics_summary,
)
from bond_match.evaluation.be_ops import (
    apply_predicted_step_be,
    apply_step_with_lp,
    be_result_to_smiles,
    score_actions,
    decode_swap_type,
)
from bond_match.evaluation.model_io import (
    load_eval_model,
)
from bond_match.evaluation.data import (
    InferenceDataset,
    inference_collate,
    _load_eval_data,
    _sample_one_jsonl_entry,
)
from bond_match.evaluation.evaluator import (
    Evaluator,
)
from bond_match.evaluation.cli import (
    main,
)

__all__ = [
    'logger',
    'PT',
    '_original_mol_to_smiles',
    '_uncharger',
    '_normalizer',
    '_tautomer_enumerator',
    'custom_mol_to_smiles',
    '_normalize_cache',
    'StopReason',
    'BEState',
    'SampleMeta',
    'StepResult',
    'ForwardResult',
    'Candidate',
    'PairResult',
    'BeamEntry',
    'TerminalEntry',
    'IterationResult',
    'EvalResult',
    'ActiveMolecule',
    'DisplayResult',
    'seed_everything',
    'same_molecule_different_protonation_or_tautomer',
    'remove_atom_mapping',
    'canonicalize_smiles',
    'strip_atom_mappings_from_data',
    '_normalize_smiles',
    'check_smiles_match',
    '_is_no_reaction_target',
    '_rewrite_product_as_reactant',
    '_maybe_rewrite_yield0_entry',
    'is_entry_successful',
    '_build_yield_metrics_summary',
    'apply_predicted_step_be',
    'apply_step_with_lp',
    'be_result_to_smiles',
    'score_actions',
    'decode_swap_type',
    'load_eval_model',
    'InferenceDataset',
    'inference_collate',
    '_load_eval_data',
    '_sample_one_jsonl_entry',
    'Evaluator',
    'main',
]
