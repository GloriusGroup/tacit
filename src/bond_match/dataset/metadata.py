"""Sample metadata container and tensor/array conversion helpers."""

from dataclasses import dataclass
import os
import torch
import numpy as np
import torch


@dataclass(slots=True)
class SampleMetadata:
    """Compact per-sample metadata kept in memory for sharded datasets."""

    type: str
    has_catalyst: bool
    step_index: int
    reactant_smi: str = ""

    def __getitem__(self, key):
        """Return the attribute named ``key`` (dict-style access)."""
        return getattr(self, key)

    def get(self, key, default=None):
        """Return the attribute named ``key`` or ``default`` if absent."""
        return getattr(self, key, default)

def _coerce_sample_metadata(item) -> SampleMetadata:
    """Convert a metadata dict to ``SampleMetadata``.

    Args:
        item: A ``SampleMetadata`` instance (returned unchanged) or a dict
            with at least a ``type`` key.

    Returns:
        A ``SampleMetadata`` instance.
    """
    if isinstance(item, SampleMetadata):
        return item
    return SampleMetadata(
        type=item["type"],
        has_catalyst=item.get("has_catalyst", False),
        step_index=item.get("step_index", 0),
        reactant_smi=item.get("reactant_smi", ""),
    )

def _dataset_cache_prefix(json_path: str) -> str:
    """Build a cache-file prefix from the last four path components.

    Args:
        json_path: Path to the source dataset file.

    Returns:
        Underscore-joined identifier derived from the resolved path with the
        file extension removed.
    """
    resolved_path = os.path.realpath(json_path)
    path_parts = list(os.path.normpath(resolved_path).split(os.sep))
    path_parts[-1] = os.path.splitext(path_parts[-1])[0]
    meaningful_parts = path_parts[-4:]
    return "_".join(part.replace(os.sep, "_") for part in meaningful_parts if part)

def _sample_tensors_to_numpy(sample: dict) -> dict:
    """Convert the torch tensors of a sample (and its history) to numpy arrays for pickle-safe IPC."""
    for key in ("x", "edge_index", "edge_attr"):
        if key in sample and isinstance(sample[key], torch.Tensor):
            sample[key] = sample[key].numpy()
    if "history" in sample:
        for h in sample["history"]:
            for key in ("x", "edge_index", "edge_attr"):
                if key in h and isinstance(h[key], torch.Tensor):
                    h[key] = h[key].numpy()
    return sample

def _sample_numpy_to_tensors(sample: dict) -> dict:
    """Convert the numpy arrays of a sample (and its history) back to torch tensors."""
    for key, dtype in (
        ("x", torch.float32),
        ("edge_index", torch.long),
        ("edge_attr", torch.long),
    ):
        if key in sample and isinstance(sample[key], np.ndarray):
            sample[key] = torch.from_numpy(sample[key]).to(dtype)
    if "history" in sample:
        for h in sample["history"]:
            for key, dtype in (
                ("x", torch.float32),
                ("edge_index", torch.long),
                ("edge_attr", torch.long),
            ):
                if key in h and isinstance(h[key], np.ndarray):
                    h[key] = torch.from_numpy(h[key]).to(dtype)
    return sample

def _trajectory_numpy_to_tensors(trajectory: dict) -> dict:
    """Convert all states of a trajectory dict back to torch tensors.

    Single-state samples (no ``states`` key) are converted directly.
    """
    if "states" not in trajectory:
        return _sample_numpy_to_tensors(trajectory)
    trajectory = dict(trajectory)
    trajectory["states"] = [
        _sample_numpy_to_tensors(dict(state)) for state in trajectory["states"]
    ]
    return trajectory

def _extract_entry_yield(entry):
    """Extract the reaction yield from an entry's metadata.

    Searches ``metadata`` and ``metadata.metadata`` for a ``Yield``/``yield``
    key, converting string values to float.

    Args:
        entry: Raw reaction entry dict.

    Returns:
        The yield value, or None if absent, NaN, or unparseable.
    """
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return None

    for container in (metadata, metadata.get("metadata")):
        if not isinstance(container, dict):
            continue
        for key in ("Yield", "yield"):
            if key not in container:
                continue
            value = container[key]
            if value is None:
                return None
            if isinstance(value, float) and np.isnan(value):
                return None
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    return None
                if np.isnan(value):
                    return None
            return value

    return None
