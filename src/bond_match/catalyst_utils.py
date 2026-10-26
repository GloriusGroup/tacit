"""Utility functions for repairing catalyst SMILES strings.

Metal-organic photocatalysts (Ir and Ru complexes) are often recorded with
dative bonds or charge annotations such as ``[C-]`` and ``[B+3]`` that RDKit
cannot parse directly. This module provides ``fix_catalyst_smiles``, which
applies counter-ion rewrites, metal-coordination surgery, and fragment-level
repair to produce a SMILES that ``Chem.MolFromSmiles`` accepts.

``METAL_ATOMIC_NUMS`` lists the atomic numbers of metals commonly found in
photocatalysts and organometallic catalysts (Al, Co, Ni, Ru, Rh, Pd, Ir, Pt).
"""

import re
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import RWMol

METAL_ATOMIC_NUMS = {13, 27, 28, 44, 45, 46, 77, 78}


def _fix_broken_salts(smiles: str) -> str:
    """Rewrite mis-charged counter-ion notation into RDKit-parseable form.

    Handles hexafluorophosphate written with ``[P+5]``, tetrafluoroborate
    written with ``[B+3]``, and tetraarylborate (BArF-type) fragments that
    combine ``[B+3]`` with ``[C-]`` aryl carbons.

    Args:
        smiles: Raw SMILES string.

    Returns:
        SMILES string with salt patterns rewritten; returned unchanged when
        no pattern matches.
    """
    if not smiles:
        return smiles

    if "P+5" in smiles:
        pf6_pattern = r"\[F-\]\[P\+5\]\(\[F-\]\)\(\[F-\]\)\(\[F-\]\)\(\[F-\]\)\[F-\]"
        smiles = re.sub(pf6_pattern, "F[P-](F)(F)(F)(F)F", smiles)

    if "B+3" in smiles and "[F-]" in smiles:
        bf4_pattern = r"\[F-\]\[B\+3\]\(\[F-\]\)\(\[F-\]\)\[F-\]"
        smiles = re.sub(bf4_pattern, "F[B-](F)(F)F", smiles)

    if "[B+3]" in smiles:
        fragments = smiles.split(".")
        fixed_frags = []
        for frag in fragments:
            if "[B+3]" in frag:
                frag = frag.replace("[B+3]", "[B-]").replace("[C-]", "C")
            fixed_frags.append(frag)
        smiles = ".".join(fixed_frags)

    return smiles


def _has_metal(mol) -> bool:
    """Return True if the molecule contains an atom from METAL_ATOMIC_NUMS."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in METAL_ATOMIC_NUMS:
            return True
    return False


def _metal_surgery(emol: RWMol):
    """Edit an RWMol in place to make metal coordination compatible with RDKit.

    For every metal atom, clears aromaticity and explicit/implicit hydrogens,
    converts all bonds to the metal into single bonds, neutralizes carbanion
    neighbors, and clears radical electrons on aromatic nitrogen neighbors.
    Property caches are updated non-strictly afterwards.

    Args:
        emol: Molecule modified in place.
    """
    for atom in emol.GetAtoms():
        if atom.GetAtomicNum() not in METAL_ATOMIC_NUMS:
            continue

        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)

        for bond in atom.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)

            other_idx = bond.GetOtherAtomIdx(atom.GetIdx())
            other = emol.GetAtomWithIdx(other_idx)

            if other.GetAtomicNum() == 6 and other.GetFormalCharge() == -1:
                other.SetFormalCharge(0)
            if other.GetAtomicNum() == 7 and other.GetIsAromatic():
                other.SetNumRadicalElectrons(0)

    for atom in emol.GetAtoms():
        try:
            atom.UpdatePropertyCache(strict=False)
        except Exception:
            pass


def _try_sanitize_and_export(mol) -> str | None:
    """Sanitize a molecule under progressively relaxed flags and export SMILES.

    Attempts full sanitization first, then variants that skip property
    calculation and kekulization. For each successful sanitization, tries
    kekulized output before aromatic output and accepts the first SMILES that
    round-trips through ``Chem.MolFromSmiles``.

    Args:
        mol: Molecule to sanitize and export.

    Returns:
        A SMILES string parseable by ``Chem.MolFromSmiles``, or None if all
        attempts fail.
    """
    sanitize_flag_combos = [
        Chem.SanitizeFlags.SANITIZE_ALL,
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        (Chem.SanitizeFlags.SANITIZE_ALL
         ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
         ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE),
    ]

    for flags in sanitize_flag_combos:
        try:
            test_mol = RWMol(mol)
            Chem.SanitizeMol(test_mol, flags)
            test_mol.UpdatePropertyCache(strict=False)
        except Exception:
            continue

        for kekule in [True, False]:
            try:
                if kekule:
                    Chem.Kekulize(test_mol, clearAromaticFlags=True)
                    smi = Chem.MolToSmiles(test_mol, kekuleSmiles=True)
                else:
                    smi = Chem.MolToSmiles(test_mol)

                if Chem.MolFromSmiles(smi) is not None:
                    return smi
            except Exception:
                continue

    return None


def _disconnect_metal(smi: str) -> str | None:
    """Remove all metal-ligand bonds and export the disconnected structure.

    Before deleting the bonds, neutralizes charged carbon and oxygen
    neighbors of metal atoms and clears radical electrons on nitrogen
    neighbors. If exporting the whole disconnected molecule fails, repairs
    the dot-separated fragments individually and recombines the valid ones.

    Args:
        smi: SMILES string of a metal-containing structure.

    Returns:
        A SMILES string parseable by ``Chem.MolFromSmiles``, or None if the
        structure could not be exported.
    """
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if not mol:
        return None

    emol = RWMol(mol)

    metal_idxs = set()
    bonds_to_remove = []
    for atom in emol.GetAtoms():
        if atom.GetAtomicNum() in METAL_ATOMIC_NUMS:
            metal_idxs.add(atom.GetIdx())
            atom.SetIsAromatic(False)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)

    for bond in emol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in metal_idxs or end_idx in metal_idxs:
            bonds_to_remove.append((begin_idx, end_idx))

            non_metal_idx = end_idx if begin_idx in metal_idxs else begin_idx
            other = emol.GetAtomWithIdx(non_metal_idx)

            if other.GetAtomicNum() == 6 and other.GetFormalCharge() == -1:
                other.SetFormalCharge(0)
            if other.GetAtomicNum() == 7:
                other.SetNumRadicalElectrons(0)
            if other.GetAtomicNum() == 8 and other.GetFormalCharge() == -1:
                other.SetFormalCharge(0)

    for begin_idx, end_idx in bonds_to_remove:
        emol.RemoveBond(begin_idx, end_idx)

    for atom in emol.GetAtoms():
        try:
            atom.UpdatePropertyCache(strict=False)
        except Exception:
            pass

    result = _try_sanitize_and_export(emol)
    if result:
        return result

    try:
        Chem.SanitizeMol(
            emol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        emol.UpdatePropertyCache(strict=False)
        raw_smi = Chem.MolToSmiles(emol)
    except Exception:
        return None

    valid_frags = []
    for frag in raw_smi.split("."):
        if Chem.MolFromSmiles(frag) is not None:
            valid_frags.append(Chem.MolToSmiles(Chem.MolFromSmiles(frag)))
        else:
            m = Chem.MolFromSmiles(frag, sanitize=False)
            if m:
                fixed = _try_sanitize_and_export(RWMol(m))
                if fixed:
                    valid_frags.append(fixed)

    if valid_frags:
        combined = ".".join(valid_frags)
        if Chem.MolFromSmiles(combined) is not None:
            return combined

    return None


def _fix_single_fragment(frag_smi: str) -> str | None:
    """Repair a single dot-separated SMILES fragment.

    Returns the canonical SMILES directly if the fragment already parses.
    Otherwise applies relaxed sanitization; for metal-containing fragments,
    additionally applies metal surgery and, as a last resort, metal
    disconnection.

    Args:
        frag_smi: SMILES of one fragment.

    Returns:
        A parseable SMILES for the fragment, or None if it could not be
        repaired.
    """
    mol = Chem.MolFromSmiles(frag_smi)
    if mol:
        return Chem.MolToSmiles(mol)

    mol = Chem.MolFromSmiles(frag_smi, sanitize=False)
    if not mol:
        return None

    emol = RWMol(mol)

    if not _has_metal(emol):
        return _try_sanitize_and_export(emol)

    _metal_surgery(emol)
    result = _try_sanitize_and_export(emol)
    if result:
        return result

    return _disconnect_metal(frag_smi)


def _verify_as_catalyst(smi: str) -> bool:
    """Check that a SMILES remains parseable when combined with another component.

    Prepends ``"C."`` to verify the string is valid as part of a
    multi-component mixture.

    Args:
        smi: Candidate catalyst SMILES.

    Returns:
        True if ``"C." + smi`` parses, False otherwise.
    """
    if not smi:
        return False
    return Chem.MolFromSmiles("C." + smi) is not None


@lru_cache(maxsize=256)
def fix_catalyst_smiles(smiles: str) -> str | None:
    """Repair a catalyst SMILES so that RDKit can parse it.

    Already valid input is returned unchanged. Otherwise, repairs are
    attempted in order: counter-ion rewrites, fragment-by-fragment repair,
    and whole-string metal surgery. Results are cached because the same
    catalyst appears across many dataset entries.

    Args:
        smiles: Catalyst SMILES string.

    Returns:
        A SMILES accepted by ``Chem.MolFromSmiles`` (the input itself if
        already valid, otherwise a repaired form), or None if unfixable.
    """
    if not smiles:
        return None

    if Chem.MolFromSmiles(smiles) is not None:
        return smiles

    clean = _fix_broken_salts(smiles)

    mol = Chem.MolFromSmiles(clean)
    if mol:
        return Chem.MolToSmiles(mol)

    fragments = [f for f in clean.split(".") if f]
    fixed_fragments = []

    for frag in fragments:
        fixed = _fix_single_fragment(frag)
        if fixed:
            fixed_fragments.append(fixed)

    if fixed_fragments:
        result = ".".join(fixed_fragments)
        if _verify_as_catalyst(result):
            return result

    mol = Chem.MolFromSmiles(clean, sanitize=False)
    if mol:
        emol = RWMol(mol)
        if _has_metal(emol):
            _metal_surgery(emol)
            result = _try_sanitize_and_export(emol)
            if result and _verify_as_catalyst(result):
                return result

    return None
