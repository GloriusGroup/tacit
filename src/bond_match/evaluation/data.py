"""Dataset, collation, and data-loading utilities for evaluation.

Contains the ``InferenceDataset`` used for test-set evaluation, the matching
collate function, and JSON/JSONL loaders that support random subsampling
without reading entire files.
"""

import contextlib
import io
import json
import random
import numpy as np
import torch
from rdkit import Chem
from bond_match.dataset import (
    _extract_entry_yield,
    augment_with_lp_edges,
    collate_fn,
    get_graph_features_nx,
    prepare_v3_reactant,
)

from bond_match.evaluation.constants import (
    logger,
)
from bond_match.evaluation.results import (
    SampleMeta,
)
from bond_match.evaluation.utils import (
    _maybe_rewrite_yield0_entry,
    is_entry_successful,
    remove_atom_mapping,
)

class InferenceDataset(torch.utils.data.Dataset):
    """Dataset that converts raw reaction entries into model-ready samples for inference."""

    def __init__(
        self,
        entries,
        extended_features=False,
        canonical_rank=False,
        enable_yield0_negatives=False,
    ):
        """Store entries and feature options.

        Args:
            entries: List of reaction entry dictionaries.
            extended_features: Whether to compute extended atom features.
            canonical_rank: Whether to include canonical rank features.
            enable_yield0_negatives: Whether to rewrite yield-0 entries as
                reactant>>reactant targets.
        """
        self.entries = entries
        self.extended_features = extended_features
        self.canonical_rank = canonical_rank
        self.enable_yield0_negatives = enable_yield0_negatives

    def __len__(self):
        """Return the number of entries."""
        return len(self.entries)

    def __getitem__(self, idx):
        """Process the entry at ``idx`` into a sample dictionary.

        Returns:
            A sample dictionary with ``_entry_idx`` set, or None if the
            entry could not be processed.
        """
        entry = _maybe_rewrite_yield0_entry(
            self.entries[idx],
            enable_yield0_negatives=self.enable_yield0_negatives,
        )
        sample = self._process_v3(entry)
        if sample is not None:
            sample["_entry_idx"] = idx
        return sample

    def _process_v3(self, entry):
        """Build a model-ready sample dictionary from a raw reaction entry.

        Prepares the reactant (BE matrix, graph, atom identities), extracts
        graph features, augments the graph with lone-pair edges, and collects
        target SMILES and step-count metadata.

        Args:
            entry: Raw reaction entry dictionary.

        Returns:
            Sample dictionary with tensors and metadata, or None if the
            entry cannot be parsed or has no bonds.
        """
        try:
            prep = prepare_v3_reactant(entry, extended_features=self.extended_features)
            if prep is None:
                return None

            mol = prep["mol"]
            G = prep["G"]
            be_matrix = prep["be_matrix"]
            atoms_identity = prep["atoms_array"]
            has_catalyst = prep["has_catalyst"]
            n_reactant_heavy = prep["n_reactant_heavy"]

            x, edge_index, edge_attr, bonds, edge_map = get_graph_features_nx(
                G,
                extended_features=self.extended_features,
                canonical_rank=self.canonical_rank,
            )
            if len(bonds) == 0:
                return None

            node_indices = sorted(list(G.nodes()))
            node_map_inv = {i: node_idx for i, node_idx in enumerate(node_indices)}

            x, edge_index, edge_attr, bonds, edge_map, lp_node_map = (
                augment_with_lp_edges(
                    x, edge_index, edge_attr, bonds, edge_map, be_matrix, node_map_inv
                )
            )
            reverse_lp_map = {
                lp_idx: atom_idx for atom_idx, lp_idx in lp_node_map.items()
            }

            bonds = np.array([sorted(b) for b in bonds])

            rxn_smiles = entry.get("reaction_smiles", "")
            if ">>" in rxn_smiles:
                _, product_smi_raw = rxn_smiles.split(">>")
                product_smi = remove_atom_mapping(product_smi_raw)
            else:
                product_smi = ""

            clean_mol = Chem.RWMol(mol)
            for atom in clean_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            clean_reactant = Chem.MolToSmiles(Chem.RemoveHs(clean_mol))

            mechanisms = entry.get("mechanisms", [])
            if mechanisms and mechanisms[0].get("steps"):
                num_steps = len(mechanisms[0]["steps"])
            else:
                num_steps = entry.get("num_steps", 1)

            return {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "bonds": bonds,
                "y": np.zeros(len(bonds)),
                "is_over_target": 0,
                "step_index": 0,
                "reactant_smiles": clean_reactant,
                "target_smiles": product_smi,
                "yield_value": _extract_entry_yield(entry),
                "G": G,
                "mol": mol,
                "be_matrix": be_matrix,
                "atoms_identity": atoms_identity,
                "lp_node_map": lp_node_map,
                "reverse_lp_map": reverse_lp_map,
                "has_catalyst": has_catalyst,
                "n_reactant_heavy": n_reactant_heavy,
                "num_steps": num_steps,
            }
        except Exception as e:
            logger.debug("Could not parse v3 entry: %s", e)
            return None

def inference_collate(batch):
    """Collate inference samples, separating tensor data from metadata.

    Args:
        batch: List of sample dictionaries (None entries are dropped).

    Returns:
        Tuple ``(collated_tensors, metadata)`` where ``metadata`` is a list
        of ``SampleMeta`` objects, or None if the batch is empty.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    metadata = []
    tensor_batch = []

    for item in batch:
        meta = SampleMeta(
            reactant_smiles=item.pop("reactant_smiles"),
            target_smiles=item.pop("target_smiles"),
            yield_value=item.pop("yield_value", None),
            G=item.pop("G"),
            mol=item.pop("mol"),
            be_matrix=item.pop("be_matrix", None),
            atoms_identity=item.pop("atoms_identity", None),
            lp_node_map=item.pop("lp_node_map", {}),
            reverse_lp_map=item.pop("reverse_lp_map", {}),
            has_catalyst=item.pop("has_catalyst", False),
            n_reactant_heavy=item.pop("n_reactant_heavy", 0),
            num_steps=item.pop("num_steps", 1),
            step_index=item.get("step_index", 0),
            entry_idx=item.pop("_entry_idx", None),
        )
        metadata.append(meta)
        tensor_batch.append(item)

    collated_tensors = collate_fn(
        tensor_batch, enable_matrix_output=False, enable_history=False
    )
    return collated_tensors, metadata

def _load_eval_data(
    json_path: str,
    limit: int | None,
    require_success: bool = True,
    enable_yield0_negatives: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """Load evaluation entries from a JSON or JSONL file.

    For JSON files, or when ``limit`` is None, the whole file is loaded,
    filtered, and optionally subsampled. For JSONL files with a limit, byte
    offsets of all lines are built first and random lines are read until
    ``limit`` valid entries are collected, so the full file is never parsed.

    Filtering: when ``require_success`` is True, entries must pass
    ``is_entry_successful``; otherwise entries with more than 4 steps are
    dropped. Yield-0 entries are rewritten when ``enable_yield0_negatives``
    is set.

    Args:
        json_path: Path to the JSON or JSONL file.
        limit: Maximum number of entries to return, or None for all.
        require_success: Whether to keep only successful entries.
        enable_yield0_negatives: Whether to rewrite yield-0 entries as
            reactant>>reactant targets.
        rng: Random number generator; the ``random`` module is used if None.

    Returns:
        List of entry dictionaries.
    """
    if rng is None:
        rng = random

    if not json_path.endswith(".jsonl") or limit is None:
        if json_path.endswith(".jsonl"):
            with open(json_path, "r") as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(json_path, "r") as f:
                data = json.load(f)
        if require_success:
            data = [e for e in data if is_entry_successful(e)]
        else:
            data = [e for e in data if e.get("num_steps", 1) <= 4]
        data = [
            _maybe_rewrite_yield0_entry(
                e, enable_yield0_negatives=enable_yield0_negatives
            )
            for e in data
        ]
        if limit is not None and limit < len(data):
            indices = rng.sample(range(len(data)), limit)
            data = [data[i] for i in sorted(indices)]
        return data

    with open(json_path, "r") as f:
        total_lines = sum(1 for _ in f)

    print(f"  File has {total_lines} lines, sampling {limit} entries...")

    offsets = []
    with open(json_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)

    available = list(range(len(offsets)))
    rng.shuffle(available)

    data = []
    idx_pos = 0
    with open(json_path, "r") as f:
        while len(data) < limit and idx_pos < len(available):
            line_idx = available[idx_pos]
            idx_pos += 1
            f.seek(offsets[line_idx])
            line = f.readline()
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if require_success:
                if is_entry_successful(entry):
                    data.append(
                        _maybe_rewrite_yield0_entry(
                            entry,
                            enable_yield0_negatives=enable_yield0_negatives,
                        )
                    )
            else:
                if entry.get("num_steps", 1) <= 4:
                    data.append(
                        _maybe_rewrite_yield0_entry(
                            entry,
                            enable_yield0_negatives=enable_yield0_negatives,
                        )
                    )

    print(f"  Loaded {len(data)} valid entries (checked {idx_pos} lines)")
    return data

def _load_all_eval_entries(json_path: str) -> list[dict]:
    """Load every entry of a JSONL file, without the blockspace filter.

    Entries that fail ``is_entry_successful`` keep only their reaction SMILES
    and metadata, so the model is evaluated on them from the reaction SMILES
    alone. Used for the full USPTO-MIT test set and the HTE datasets.
    """
    with open(json_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    with contextlib.redirect_stdout(io.StringIO()):
        keep = [is_entry_successful(e) for e in data]
    return [
        e if ok else {"reaction_smiles": e["reaction_smiles"], "metadata": e.get("metadata", {})}
        for e, ok in zip(data, keep)
    ]

def _sample_one_jsonl_entry(
    json_path: str,
    enable_yield0_negatives: bool = False,
    rng: random.Random | None = None,
) -> dict | None:
    """Sample a single random valid entry from a JSONL file.

    Builds byte offsets for all non-empty lines, then reads lines in random
    order until one passes ``is_entry_successful``.

    Args:
        json_path: Path to the JSONL file.
        enable_yield0_negatives: Whether to rewrite a yield-0 entry as a
            reactant>>reactant target.
        rng: Random number generator; the ``random`` module is used if None.

    Returns:
        A valid entry dictionary, or None if no line qualifies.
    """
    if rng is None:
        rng = random

    offsets = []
    with open(json_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)

    indices = list(range(len(offsets)))
    rng.shuffle(indices)

    with open(json_path, "r") as f:
        for idx in indices:
            f.seek(offsets[idx])
            line = f.readline()
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_entry_successful(entry):
                return _maybe_rewrite_yield0_entry(
                    entry, enable_yield0_negatives=enable_yield0_negatives
                )
    return None

