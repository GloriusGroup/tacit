"""
Atom and step label formatting.

Human-readable / serializable labels for atoms ("C:12", "H:0~C:7"), single
changes ("C:1—O:3 +1", "O:3 LP -2"), and whole elemental steps.  The label
grammar here is parsed back by ``verify.parse_atom_label`` /
``verify.resolve_atom_index``.
"""

import numpy as np

from be_core import PT
from mech_steps import Change, Step


def atom_label(idx: int, atoms: np.ndarray) -> str:
    """
    Format an atom label for serialization.

    Returns 'Element:map' for mapped atoms and 'H:0' for unmapped hydrogens;
    the parent annotation ('H:0~Element:map') is added by
    ``atom_label_with_context``.

    Args:
        idx: Atom index.
        atoms: (n, 2) atom identity array with [map_num, atomic_num].

    Returns:
        Label string.
    """
    map_num, z = int(atoms[idx][0]), int(atoms[idx][1])
    elem = PT.GetElementSymbol(z)
    if z == 1 and map_num == 0:
        return f"H:0"
    return f"{elem}:{map_num}"


def atom_label_with_context(idx: int, atoms: np.ndarray, be_matrix: np.ndarray) -> str:
    """
    Format an atom label with parent context for unmapped hydrogens.

    Unmapped hydrogens are identified by the first bonded non-H atom found in
    the BE matrix.

    Args:
        idx: Atom index.
        atoms: (n, 2) atom identity array with [map_num, atomic_num].
        be_matrix: BE matrix used to locate the parent heavy atom.

    Returns:
        'Element:map' for mapped atoms, 'H:0~Element:map' for unmapped H
        bonded to a heavy atom, or 'H:0' if no parent is found.
    """
    map_num, z = int(atoms[idx][0]), int(atoms[idx][1])
    elem = PT.GetElementSymbol(z)
    if z == 1 and map_num == 0:
        for j in range(len(be_matrix)):
            if j != idx and be_matrix[idx, j] > 0 and int(atoms[j][1]) != 1:
                parent_map = int(atoms[j][0])
                parent_elem = PT.GetElementSymbol(int(atoms[j][1]))
                return f"H:0~{parent_elem}:{parent_map}"
        return "H:0"
    return f"{elem}:{map_num}"


def format_change(change: Change, atoms: np.ndarray) -> str:
    """
    Format a single change for display.

    Args:
        change: (i, j, value) change tuple.
        atoms: (n, 2) atom identity array.

    Returns:
        'Element:map LP ±v' for diagonal (lone pair) changes,
        'Element:map—Element:map ±v' for bond changes.
    """
    i, j, val = change

    def atom_label(idx):
        """Format 'Element:map' for the atom at idx ('Element:?' if unmapped)."""
        map_num, z = int(atoms[idx][0]), int(atoms[idx][1])
        elem = PT.GetElementSymbol(z)
        return f"{elem}:{map_num}" if map_num > 0 else f"{elem}:?"

    sign = "+" if val > 0 else ""

    if i == j:
        return f"{atom_label(i)} LP {sign}{val:.0f}"
    else:
        return f"{atom_label(i)}—{atom_label(j)} {sign}{val:.0f}"


def format_step(step: Step, atoms: np.ndarray) -> str:
    """
    Format an elemental step for display.

    Changes are sorted with negative (bond-breaking) changes before positive
    ones.

    Args:
        step: List of change tuples.
        atoms: (n, 2) atom identity array.

    Returns:
        Comma-separated formatted changes.
    """
    sorted_step = sorted(step, key=lambda c: (c[2] > 0, c[0], c[1]))
    return ", ".join(format_change(c, atoms) for c in sorted_step)


def print_step_details(step: Step, atoms: np.ndarray, step_num: int):
    """
    Print a detailed source/sink breakdown of a step.

    Args:
        step: List of change tuples.
        atoms: (n, 2) atom identity array.
        step_num: Step number used in the printed header.
    """
    sources = [c for c in step if c[2] < 0]
    sinks = [c for c in step if c[2] > 0]

    print(f"  Step {step_num}:")
    print(f"    Sources (-4): {', '.join(format_change(c, atoms) for c in sources)}")
    print(f"    Sinks   (+4): {', '.join(format_change(c, atoms) for c in sinks)}")
