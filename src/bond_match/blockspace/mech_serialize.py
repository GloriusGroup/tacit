"""
Mechanism serialization to a JSON-compatible format.

Each mechanism is a list of steps; each step records broken bonds, formed
bonds, lone pair changes (as atom labels from ``atom_labels``), and the
intermediate SMILES after applying the step.  ``verify.py`` consumes this
format and checks it against the reaction.
"""

from typing import List, Optional, Tuple

import numpy as np

from atom_labels import atom_label_with_context
from be_components import decompose_molecules
from be_core import matrices_to_smiles
from map_assignment import assign_full_mapping
from mech_steps import Step, apply_step


def serialize_mechanism(
    rxn_smiles: str,
    all_mechanisms: List[List[Step]],
    be_sub: np.ndarray,
    atoms_sub: np.ndarray,
    intermediates: Optional[List[Tuple[int, int]]] = None,
    mapped_rxn: Optional[str] = None,
) -> dict:
    """
    Serialize all mechanisms to a JSON-compatible dict.

    When ``mapped_rxn`` is given, ``atoms_sub`` is assumed to be fully mapped
    and index-consistent with the mechanism and is used directly, avoiding
    index misalignment from re-parsing. Otherwise, a full mapping is assigned
    from scratch via ``assign_full_mapping``.

    Args:
        rxn_smiles: Original reaction SMILES.
        all_mechanisms: Mechanisms to serialize (lists of steps).
        be_sub: Substrate BE matrix.
        atoms_sub: Substrate atom identity array.
        intermediates: Optional intermediate bonds (unused in serialization).
        mapped_rxn: Optional fully mapped reaction SMILES consistent with
            ``atoms_sub``.

    Returns:
        JSON-compatible dict in the following format:
    {
        "reaction_smiles": "original",
        "reaction_smiles_mapped": "fully mapped",
        "success": true,
        "num_steps": N,
        "num_mechanisms": M,
        "mechanisms": [
            { "steps": [
                {
                    "break": [["C:15","C:12"], ...],
                    "form":  [["C:15","H:0~C:12"], ...],
                    "smiles": "intermediate SMILES"
                }, ...
            ] }, ...
        ]
    }
    """
    if mapped_rxn:
        mapped_sub, mapped_prod = mapped_rxn.split(">>")
        atoms_mapped = atoms_sub
    else:
        sub_smi, prod_smi = rxn_smiles.split(">>")
        mapped_sub, mapped_prod, atoms_mapped = assign_full_mapping(
            sub_smi, prod_smi, atoms_sub
        )

    num_steps = len(all_mechanisms[0]) if all_mechanisms else 0
    mechanisms = []

    for mech in all_mechanisms:
        be_current = be_sub.copy()
        step_records = []

        for i, step in enumerate(mech):
            breaks = []
            forms = []
            lp_changes = []

            for si, sj, val in step:
                v = float(val)
                if si == sj:
                    label = atom_label_with_context(si, atoms_mapped, be_current)
                    sign = "+" if v > 0 else ""
                    lp_changes.append(f"{label} LP {sign}{int(v)}")
                elif v < 0:
                    a1 = atom_label_with_context(si, atoms_mapped, be_current)
                    a2 = atom_label_with_context(sj, atoms_mapped, be_current)
                    breaks.append([a1, a2])
                else:
                    a1 = atom_label_with_context(si, atoms_mapped, be_current)
                    a2 = atom_label_with_context(sj, atoms_mapped, be_current)
                    forms.append([a1, a2])

            be_current = apply_step(be_current, step)

            int_mols = decompose_molecules(be_current, atoms_mapped)
            int_smiles = ".".join(matrices_to_smiles(m, a) for m, a in int_mols)

            record = {"break": breaks, "form": forms}
            if lp_changes:
                record["lp"] = lp_changes
            record["smiles"] = int_smiles

            step_records.append(record)

        mechanisms.append({"steps": step_records})

    return {
        "reaction_smiles": rxn_smiles,
        "reaction_smiles_mapped": f"{mapped_sub}>>{mapped_prod}",
        "success": num_steps > 0,
        "num_steps": num_steps,
        "num_mechanisms": len(mechanisms),
        "mechanisms": mechanisms,
    }
