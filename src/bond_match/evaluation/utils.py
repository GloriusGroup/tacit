"""SMILES canonicalization, matching, and miscellaneous evaluation utilities.

Includes random-seed control, atom-map removal, SMILES canonicalization and
normalization, prediction-target matching, yield-0 entry rewriting, and
yield-metric summary formatting.
"""

import os
import random
from typing import Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from bond_match.dataset import (
    _extract_entry_yield,
)

from bond_match.evaluation.constants import (
    _normalize_cache,
    _normalizer,
    _tautomer_enumerator,
    _uncharger,
)

def same_molecule_different_protonation_or_tautomer(smiles1: str, smiles2: str) -> bool:
    """Check if two SMILES are the same molecule differing only by protonation state or tautomer."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string")

    uncharger = rdMolStandardize.Uncharger()
    normalizer = rdMolStandardize.Normalizer()
    te = rdMolStandardize.TautomerEnumerator()

    def canonical_smiles(mol):
        """Return the canonical SMILES after normalization, uncharging, and tautomer canonicalization."""
        mol = normalizer.normalize(mol)
        mol = uncharger.uncharge(mol)
        mol = te.Canonicalize(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)

    return canonical_smiles(mol1) == canonical_smiles(mol2)

def seed_everything(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def remove_atom_mapping(smiles: str) -> str:
    """Remove atom-map numbers from a SMILES string and return its canonical form.

    Parsing falls back to unsanitized mode if sanitized parsing fails.
    Raises ``ValueError`` if the SMILES cannot be parsed at all.

    Args:
        smiles: Input SMILES, possibly with atom-map numbers.

    Returns:
        Canonical SMILES without atom-map numbers, with dative bonds included.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("Invalid SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(
        mol, canonical=True, ignoreAtomMapNumbers=True, includeDativeBonds=True
    )

def canonicalize_smiles(smi):
    """Canonicalize a possibly multi-component SMILES string for comparison.

    Each ``.``-separated component is parsed, stripped of atom-map numbers
    and explicit hydrogens, and written as canonical non-isomeric SMILES;
    unparseable components are kept verbatim and empty ones dropped. Components are sorted before
    joining so the result is order-independent.

    Args:
        smi: Input SMILES string; may contain multiple components.

    Returns:
        Sorted, dot-joined canonical SMILES, or an empty string for empty input.
    """
    if not smi:
        return ""
    smiles_list = [s for s in smi.split(".") if s]
    out_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            out_smiles.append(s)
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol = Chem.RemoveHs(mol)
        out_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        out_smiles.append(out_smi)
    return ".".join(sorted(out_smiles))

def strip_atom_mappings_from_data(data):
    """Canonicalize reactant and product SMILES of each entry's ``reaction_smiles`` in place.

    Args:
        data: List of entry dictionaries.

    Returns:
        The same list with rewritten ``reaction_smiles`` fields.
    """
    for entry in data:
        rxn = entry.get("reaction_smiles", "")
        if ">>" in rxn:
            reactant, product = rxn.split(">>")
            entry["reaction_smiles"] = (
                canonicalize_smiles(reactant) + ">>" + canonicalize_smiles(product)
            )
    return data

def _rewrite_product_as_reactant(rxn_smiles: str) -> str:
    """Rewrite a reaction SMILES so the product equals the reactant (reactant>>reactant)."""
    if not rxn_smiles or ">>" not in rxn_smiles:
        return rxn_smiles
    reactant, _ = rxn_smiles.split(">>", 1)
    return f"{reactant}>>{reactant}"

def _maybe_rewrite_yield0_entry(entry: dict, enable_yield0_negatives: bool) -> dict:
    """Rewrite a yield-0 entry so its targets become reactant>>reactant.

    Applies to ``reaction_smiles``, ``reaction_smiles_mapped``, and the
    ``Reaction SMILES`` metadata field.

    Args:
        entry: Reaction entry dictionary.
        enable_yield0_negatives: If False, the entry is returned unchanged.

    Returns:
        A rewritten copy of the entry, or the original entry if no rewrite
        is required.
    """
    if not enable_yield0_negatives or _extract_entry_yield(entry) != 0:
        return entry

    updated_entry = dict(entry)
    for field in ("reaction_smiles", "reaction_smiles_mapped"):
        rxn_smiles = entry.get(field, "")
        if rxn_smiles:
            updated_entry[field] = _rewrite_product_as_reactant(rxn_smiles)

    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        reaction_smiles = metadata.get("Reaction SMILES", "")
        if reaction_smiles:
            updated_metadata = dict(metadata)
            updated_metadata["Reaction SMILES"] = _rewrite_product_as_reactant(
                reaction_smiles
            )
            updated_entry["metadata"] = updated_metadata

    return updated_entry

def _build_yield_metrics_summary(
    enable_yield0_negatives: bool,
    yield_bucket_total: dict,
    yield_bucket_correct: dict,
    yield_zero_confusion: dict,
) -> list[str]:
    """Format yield-0 negative metrics as printable summary lines.

    Computes top-1 accuracy for yield-0 and yield-greater-than-0 buckets,
    their macro average, and precision, recall, and F1 of no-reaction
    detection from the confusion counts.

    Args:
        enable_yield0_negatives: If False, an empty list is returned.
        yield_bucket_total: Sample counts per yield bucket.
        yield_bucket_correct: Correct-prediction counts per yield bucket.
        yield_zero_confusion: Confusion counts with keys ``tp``, ``fp``, ``fn``.

    Returns:
        List of formatted summary lines (empty if disabled or no data).
    """
    if not enable_yield0_negatives or not yield_bucket_total:
        return []

    neg_total = yield_bucket_total.get("yield_0", 0)
    neg_correct = yield_bucket_correct.get("yield_0", 0)
    pos_total = yield_bucket_total.get("yield_gt_0", 0)
    pos_correct = yield_bucket_correct.get("yield_gt_0", 0)

    tp = yield_zero_confusion.get("tp", 0)
    fp = yield_zero_confusion.get("fp", 0)
    fn = yield_zero_confusion.get("fn", 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (
        0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )

    lines = [
        "  Yield-0 negatives enabled (yield=0 targets rewritten to reactant>>reactant)"
    ]
    if neg_total > 0:
        lines.append(
            f"  No-Reaction Top-1: {neg_correct / neg_total:.4f} ({neg_correct}/{neg_total})"
        )
    if pos_total > 0:
        lines.append(
            f"  Reaction Top-1:    {pos_correct / pos_total:.4f} ({pos_correct}/{pos_total})"
        )
    if neg_total > 0 and pos_total > 0:
        macro_top1 = ((neg_correct / neg_total) + (pos_correct / pos_total)) / 2.0
        lines.append(f"  Macro Top-1:       {macro_top1:.4f}")
    lines.append(f"  No-Reaction F1:    {f1:.4f}")
    lines.append(f"  No-Reaction Prec:  {precision:.4f}")
    lines.append(f"  No-Reaction Recall:{recall:.4f}")
    return lines

def _is_no_reaction_target(reactant_smiles: str, target_smiles: Optional[str]) -> bool:
    if not target_smiles:
        return False
    reactant_canon = canonicalize_smiles(reactant_smiles)
    target_canon = canonicalize_smiles(target_smiles)
    return check_smiles_match(reactant_canon, target_canon)

def _normalize_smiles(smi: str) -> str:
    """Normalize a single-component SMILES via charge neutralization and tautomer canonicalization."""
    cached = _normalize_cache.get(smi)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _normalize_cache[smi] = smi
        return smi
    mol = _normalizer.normalize(mol)
    mol = _uncharger.uncharge(mol)
    mol = _tautomer_enumerator.Canonicalize(mol)
    Chem.SanitizeMol(mol)
    result = Chem.MolToSmiles(mol, canonical=True)
    _normalize_cache[smi] = result
    return result

def check_smiles_match(predicted_canon, target_canon):
    if not predicted_canon or not target_canon:
        return False
    target_frags = set(target_canon.split("."))
    pred_frags = set(predicted_canon.split("."))
    if predicted_canon == target_canon or target_frags <= pred_frags:
        return True
    try:
        target_norm = {_normalize_smiles(s) for s in target_frags}
        pred_norm = {_normalize_smiles(s) for s in pred_frags}
        return target_norm <= pred_norm
    except Exception:
        return False

def is_no_reaction_match(input_canon, target_canon):
    """Whether stopping before the first step matches the target.

    Only a no-reaction target (the unchanged reactant set) counts. The fragment
    subset rule of ``check_smiles_match`` would otherwise score a first-step stop
    as correct whenever the product already appears among the reactants.
    """
    return bool(target_canon) and input_canon == target_canon

def is_entry_successful(entry):
    mechanisms = entry.get("mechanisms", [])
    if mechanisms and mechanisms[0].get("steps"):
        num_steps = len(mechanisms[0]["steps"])
    else:
        num_steps = entry.get("num_steps", 1)
    if num_steps > 4:
        print(
            f"Warning: Entry has {num_steps} steps, which is unusually high. Skipping."
        )
        return False
    return entry.get("success", True) and num_steps >= 1

