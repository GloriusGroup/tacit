"""Shared fixtures for bond_match test suite."""

import json
from pathlib import Path

import pytest
from rdkit import Chem

USPTO_TRAIN = Path(__file__).resolve().parents[1] / "data" / "uspto_mit" / "train.jsonl"


def load_uspto_entries(n, require_steps=True):
    """First ``n`` successful entries of ``data/uspto_mit/train.jsonl``; skip the test if the file is missing."""
    if not USPTO_TRAIN.exists():
        pytest.skip(f"Dataset not found: {USPTO_TRAIN}")
    entries = []
    with USPTO_TRAIN.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("success") is False:
                continue
            mechs = entry.get("mechanisms", [])
            if require_steps and not (mechs and mechs[0].get("steps")):
                continue
            entries.append(entry)
            if len(entries) >= n:
                break
    return entries


@pytest.fixture
def simple_mol():
    """Water molecule with explicit hydrogens."""
    mol = Chem.MolFromSmiles("O")
    return Chem.AddHs(mol)


@pytest.fixture
def mapped_mol():
    """Methanol with atom-map numbers and explicit hydrogens."""
    mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
    return Chem.AddHs(mol)


@pytest.fixture
def sample_v3_entry():
    """Return the first valid v3 blockspace entry (successful, with mechanism steps) from the USPTO dataset; skip if unavailable."""
    entries = load_uspto_entries(1)
    if not entries:
        pytest.skip("No valid v3 entry found in dataset")
    return entries[0]
