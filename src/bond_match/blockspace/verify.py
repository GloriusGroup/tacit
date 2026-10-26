#!/usr/bin/env python
"""
Verify serialized mechanism JSON files.

Reads a mechanism JSON, reconstructs each step from the serialized atom labels,
applies them to the substrate BE matrix, and checks that the final result
matches the product.

Usage:
    python verify.py mechanism.json
"""

import argparse
import json
import re
import sys

from rdkit import Chem

from be_components import decompose_molecules
from be_core import matrices_to_smiles, smiles_to_matrices
from be_hash import is_subset
from mech_steps import apply_step, is_valid_be

PT = Chem.GetPeriodicTable()


def parse_atom_label(label: str):
    """
    Parse an atom label string.

    Returns:
        (element, map_num, parent_element, parent_map)
        parent fields are None for non-H or mapped atoms.

    Examples:
        "C:1"       -> ("C", 1, None, None)
        "H:0~C:7"   -> ("H", 0, "C", 7)
        "H:0"       -> ("H", 0, None, None)
    """
    m = re.match(r"^(\w+):(\d+)~(\w+):(\d+)$", label)
    if m:
        return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    m = re.match(r"^(\w+):(\d+)$", label)
    if m:
        return m.group(1), int(m.group(2)), None, None
    raise ValueError(f"Cannot parse atom label: {label}")


def resolve_atom_index(label: str, atoms, be_matrix):
    """
    Resolve a serialized atom label to a BE matrix index.

    For mapped atoms (map > 0): find by map number.
    For unmapped H (H:0~Element:map): find H bonded to the parent atom.
    """
    elem, map_num, parent_elem, parent_map = parse_atom_label(label)
    z = PT.GetAtomicNumber(elem)

    if map_num > 0:
        # Mapped atom: find by map number
        for i in range(len(atoms)):
            if int(atoms[i][0]) == map_num and int(atoms[i][1]) == z:
                return i
        raise ValueError(f"No atom found for {label}")

    if elem == "H" and parent_elem is not None:
        # Unmapped H bonded to parent
        parent_z = PT.GetAtomicNumber(parent_elem)
        parent_idx = None
        for i in range(len(atoms)):
            if int(atoms[i][0]) == parent_map and int(atoms[i][1]) == parent_z:
                parent_idx = i
                break
        if parent_idx is None:
            raise ValueError(
                f"Parent atom {parent_elem}:{parent_map} not found for {label}"
            )
        for i in range(len(atoms)):
            if int(atoms[i][1]) == 1 and int(atoms[i][0]) == 0:
                if be_matrix[i, parent_idx] > 0:
                    return i
        raise ValueError(f"No unmapped H bonded to {parent_elem}:{parent_map}")

    for i in range(len(atoms)):
        if int(atoms[i][0]) == 0 and int(atoms[i][1]) == z:
            return i
    raise ValueError(f"No atom found for {label}")


def reconstruct_step(step_record, atoms, be_matrix):
    """
    Reconstruct a step (list of (i, j, val) tuples) from a serialized step record.
    """
    changes = []

    for a1_label, a2_label in step_record.get("break", []):
        i = resolve_atom_index(a1_label, atoms, be_matrix)
        j = resolve_atom_index(a2_label, atoms, be_matrix)
        if i > j:
            i, j = j, i
        changes.append((i, j, -1.0))

    for a1_label, a2_label in step_record.get("form", []):
        i = resolve_atom_index(a1_label, atoms, be_matrix)
        j = resolve_atom_index(a2_label, atoms, be_matrix)
        if i > j:
            i, j = j, i
        changes.append((i, j, 1.0))

    for lp_str in step_record.get("lp", []):
        # Parse "O:3 LP -2" or "Br:2 LP +2"
        m = re.match(r"^(.+?) LP ([+-]?\d+)$", lp_str)
        if not m:
            raise ValueError(f"Cannot parse LP change: {lp_str}")
        label, val = m.group(1), int(m.group(2))
        i = resolve_atom_index(label, atoms, be_matrix)
        changes.append((i, i, float(val)))

    return changes


def verify_mechanism(data):
    """
    Verify a single mechanism JSON dict. Returns (success, messages).
    """
    msgs = []
    ok = True

    rxn_mapped = data["reaction_smiles_mapped"]
    sub_smi, prod_smi = rxn_mapped.split(">>")

    be_sub, atoms_sub = smiles_to_matrices(
        sub_smi, kekulize=True, include_hydrogens=True
    )
    be_prod, atoms_prod = smiles_to_matrices(
        prod_smi, kekulize=True, include_hydrogens=True
    )

    num_mechs = data.get("num_mechanisms", len(data.get("mechanisms", [])))
    msgs.append(f"Verifying {num_mechs} mechanism(s), {data['num_steps']} step(s) each")

    for m_idx, mech in enumerate(data["mechanisms"]):
        be_current = be_sub.copy()
        mech_ok = True

        for s_idx, step_record in enumerate(mech["steps"]):
            try:
                step = reconstruct_step(step_record, atoms_sub, be_current)
            except ValueError as e:
                msgs.append(
                    f"  Mechanism {m_idx+1}, Step {s_idx+1}: FAIL reconstruct - {e}"
                )
                mech_ok = False
                ok = False
                break

            be_current = apply_step(be_current, step)

            # Check intermediate validity
            if not is_valid_be(be_current):
                msgs.append(
                    f"  Mechanism {m_idx+1}, Step {s_idx+1}: FAIL - negative BE entries"
                )
                mech_ok = False
                ok = False
                break

            # Check intermediate SMILES matches (canonical comparison)
            expected_smiles = step_record.get("smiles", "")
            if expected_smiles:
                int_mols = decompose_molecules(be_current, atoms_sub)
                actual_smiles = ".".join(matrices_to_smiles(m, a) for m, a in int_mols)
                if actual_smiles != expected_smiles:
                    # Compare canonical SMILES after removing atom maps and explicit hydrogens.
                    def canonical_unmap(smi):
                        mol = Chem.MolFromSmiles(smi, sanitize=False)
                        if mol is None:
                            return None
                        for a in mol.GetAtoms():
                            a.SetAtomMapNum(0)
                        try:
                            Chem.SanitizeMol(mol)
                            mol = Chem.RemoveHs(mol)
                        except Exception:
                            pass
                        return Chem.MolToSmiles(mol)

                    exp_can = canonical_unmap(expected_smiles)
                    act_can = canonical_unmap(actual_smiles)
                    if exp_can is None or act_can is None:
                        msgs.append(
                            f"  Mechanism {m_idx+1}, Step {s_idx+1}: WARN - SMILES unparseable"
                        )
                    elif exp_can != act_can:
                        msgs.append(
                            f"  Mechanism {m_idx+1}, Step {s_idx+1}: WARN - SMILES mismatch"
                        )
                        msgs.append(f"    Expected: {exp_can}")
                        msgs.append(f"    Got:      {act_can}")

        if not mech_ok:
            continue

        # Check final product is subset
        if is_subset(be_prod, atoms_prod, be_current, atoms_sub):
            msgs.append(f"  Mechanism {m_idx+1}: OK")
        else:
            msgs.append(
                f"  Mechanism {m_idx+1}: FAIL - product not subset of final state"
            )
            ok = False

    return ok, msgs


def main():
    """CLI entry point: verify one or more mechanism JSON files."""
    parser = argparse.ArgumentParser(description="Verify mechanism JSON files")
    parser.add_argument("files", nargs="+", help="JSON file(s) to verify")
    args = parser.parse_args()

    all_ok = True
    for path in args.files:
        print(f"\n{'='*60}")
        print(f"File: {path}")
        print(f"{'='*60}")

        with open(path) as f:
            data = json.load(f)

        if not data.get("success", False):
            print("  Skipped (success=false)")
            continue

        ok, msgs = verify_mechanism(data)
        for msg in msgs:
            print(msg)

        if ok:
            print("PASSED")
        else:
            print("FAILED")
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
