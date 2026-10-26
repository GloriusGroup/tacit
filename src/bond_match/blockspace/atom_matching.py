"""
Substrate <-> product atom matching.

Pairs substrate atoms with product atoms by atom-map number, then resolves
unmapped hydrogens heuristically: first by shared parent heavy atom, then by
reassigning hydrogens between heavy atoms when the donor and acceptor differ,
and finally by sourcing hydrogens from byproduct molecules (preferring
hydrogens on unmapped parents such as solvents/reagents, with special
handling for H2).
"""

from typing import List, Tuple

import numpy as np


def match_atoms_by_map(
    atoms_substrate: np.ndarray,
    atoms_product: np.ndarray,
    be_substrate: np.ndarray = None,
    be_product: np.ndarray = None,
) -> Tuple[dict, List[int], List[int]]:
    """
    Match atoms between substrate and product by atom-map number.

    If BE matrices are provided, unmapped hydrogens are additionally matched
    in three phases:

    1. Per matched heavy atom: pair unmapped hydrogens bonded to the same
       heavy atom on both sides; record excess substrate hydrogens (lost) and
       excess product hydrogens (gained).
    2. Pair lost hydrogens with gained hydrogens (hydrogens reassigned
       between heavy atoms when the donor and acceptor differ).
    3. Source remaining gained hydrogens from byproduct atoms, preferring
       hydrogens whose parent is unmapped (solvents/reagents) over hydrogens
       on mapped-but-unmatched parents, since the latter creates extra bond
       deficits on mapped byproducts.  For H2 molecules at most one hydrogen
       per H-H pair is taken first; the partner is deferred so it retains a
       broken bond and can bond with heavy byproducts.

    Args:
        atoms_substrate: (n, 2) with [map_num, atomic_num]
        atoms_product: (m, 2) with [map_num, atomic_num]
        be_substrate: Optional BE matrix for hydrogen parent matching
        be_product: Optional BE matrix for hydrogen parent matching

    Returns:
        substrate_to_product: dict mapping substrate idx -> product idx (for matched atoms)
        byproduct_indices: substrate indices not in product (leaving groups, etc.)
        new_product_indices: product indices not in substrate (e.g., from reagents)
    """
    sub_map_to_idx = {}
    for i, (map_num, _) in enumerate(atoms_substrate):
        if map_num > 0:
            sub_map_to_idx[map_num] = i

    prod_map_to_idx = {}
    for i, (map_num, _) in enumerate(atoms_product):
        if map_num > 0:
            prod_map_to_idx[map_num] = i

    substrate_to_product = {}
    matched_sub = set()
    matched_prod = set()

    for map_num, sub_idx in sub_map_to_idx.items():
        if map_num in prod_map_to_idx:
            prod_idx = prod_map_to_idx[map_num]
            substrate_to_product[sub_idx] = prod_idx
            matched_sub.add(sub_idx)
            matched_prod.add(prod_idx)

    if be_substrate is not None and be_product is not None:
        lost_h_indices = []
        gained_h_indices = []

        for sub_heavy_idx, prod_heavy_idx in list(substrate_to_product.items()):
            sub_h_indices = []
            for i, (map_num, atomic_num) in enumerate(atoms_substrate):
                if atomic_num == 1 and map_num == 0 and i not in matched_sub:
                    if be_substrate[sub_heavy_idx, i] > 0:
                        sub_h_indices.append(i)

            prod_h_indices = []
            for i, (map_num, atomic_num) in enumerate(atoms_product):
                if atomic_num == 1 and map_num == 0 and i not in matched_prod:
                    if be_product[prod_heavy_idx, i] > 0:
                        prod_h_indices.append(i)

            for sub_h, prod_h in zip(sub_h_indices, prod_h_indices):
                substrate_to_product[sub_h] = prod_h
                matched_sub.add(sub_h)
                matched_prod.add(prod_h)

            if len(sub_h_indices) > len(prod_h_indices):
                lost_h_indices.extend(sub_h_indices[len(prod_h_indices) :])
            elif len(prod_h_indices) > len(sub_h_indices):
                gained_h_indices.extend(prod_h_indices[len(sub_h_indices) :])

        for sub_h, prod_h in zip(lost_h_indices, gained_h_indices):
            substrate_to_product[sub_h] = prod_h
            matched_sub.add(sub_h)
            matched_prod.add(prod_h)

        remaining_gained = [h for h in gained_h_indices if h not in matched_prod]
        if remaining_gained:
            unmapped_parent_h = []
            mapped_parent_h = []
            h2_partner = {}
            for i, (map_num, atomic_num) in enumerate(atoms_substrate):
                if atomic_num == 1 and map_num == 0 and i not in matched_sub:
                    for j in range(len(be_substrate)):
                        if be_substrate[i, j] > 0 and i != j:
                            parent_map = int(atoms_substrate[j][0])
                            if parent_map == 0:
                                unmapped_parent_h.append(i)
                                if int(atoms_substrate[j][1]) == 1:
                                    h2_partner[i] = j
                            elif j not in matched_sub:
                                mapped_parent_h.append(i)
                            break

            h2_taken = set()
            spread_unmapped = []
            deferred_unmapped = []
            for h_idx in unmapped_parent_h:
                partner = h2_partner.get(h_idx)
                if partner is not None and partner in h2_taken:
                    deferred_unmapped.append(h_idx)
                else:
                    spread_unmapped.append(h_idx)
                    if partner is not None:
                        h2_taken.add(h_idx)

            byproduct_h_indices = spread_unmapped + deferred_unmapped + mapped_parent_h

            for sub_h, prod_h in zip(byproduct_h_indices, remaining_gained):
                substrate_to_product[sub_h] = prod_h
                matched_sub.add(sub_h)
                matched_prod.add(prod_h)

    byproduct_indices = [i for i in range(len(atoms_substrate)) if i not in matched_sub]
    new_product_indices = [
        i for i in range(len(atoms_product)) if i not in matched_prod
    ]

    return substrate_to_product, byproduct_indices, new_product_indices


def _get_h_candidates_and_base_matching(
    atoms_substrate: np.ndarray,
    atoms_product: np.ndarray,
    be_substrate: np.ndarray,
    be_product: np.ndarray,
) -> Tuple[dict, set, set, List[int], List[int]]:
    """
    Run deterministic atom matching (by map number and parent atom) and return
    the ambiguous hydrogen assignment information.

    Performs the same phases as ``match_atoms_by_map`` (map-number matching,
    per-parent hydrogen matching, lost-to-gained hydrogen pairing) but stops
    before committing byproduct hydrogen sources, returning the candidate
    list instead.  Candidates are ordered as in ``match_atoms_by_map``:
    hydrogens on unmapped parents (at most one per H2 pair first, deferred
    partners next), then hydrogens on mapped-but-unmatched parents.

    Args:
        atoms_substrate: (n, 2) with [map_num, atomic_num]
        atoms_product: (m, 2) with [map_num, atomic_num]
        be_substrate: Substrate BE matrix
        be_product: Product BE matrix

    Returns:
        s2p: substrate->product mapping (before ambiguous H assignment)
        matched_sub: set of matched substrate indices
        matched_prod: set of matched product indices
        h_candidates: Substrate hydrogen indices available for byproduct-based assignment
        h_targets: Product hydrogen indices that still require assignment
    """
    sub_map_to_idx = {}
    for i, (map_num, _) in enumerate(atoms_substrate):
        if map_num > 0:
            sub_map_to_idx[map_num] = i
    prod_map_to_idx = {}
    for i, (map_num, _) in enumerate(atoms_product):
        if map_num > 0:
            prod_map_to_idx[map_num] = i

    s2p = {}
    matched_sub = set()
    matched_prod = set()
    for map_num, sub_idx in sub_map_to_idx.items():
        if map_num in prod_map_to_idx:
            prod_idx = prod_map_to_idx[map_num]
            s2p[sub_idx] = prod_idx
            matched_sub.add(sub_idx)
            matched_prod.add(prod_idx)

    lost_h_indices = []
    gained_h_indices = []
    for sub_heavy, prod_heavy in list(s2p.items()):
        sub_hs = [
            i
            for i, (m, z) in enumerate(atoms_substrate)
            if z == 1
            and m == 0
            and i not in matched_sub
            and be_substrate[sub_heavy, i] > 0
        ]
        prod_hs = [
            i
            for i, (m, z) in enumerate(atoms_product)
            if z == 1
            and m == 0
            and i not in matched_prod
            and be_product[prod_heavy, i] > 0
        ]
        for sh, ph in zip(sub_hs, prod_hs):
            s2p[sh] = ph
            matched_sub.add(sh)
            matched_prod.add(ph)
        if len(sub_hs) > len(prod_hs):
            lost_h_indices.extend(sub_hs[len(prod_hs) :])
        elif len(prod_hs) > len(sub_hs):
            gained_h_indices.extend(prod_hs[len(sub_hs) :])

    for sh, ph in zip(lost_h_indices, gained_h_indices):
        s2p[sh] = ph
        matched_sub.add(sh)
        matched_prod.add(ph)

    h_targets = [h for h in gained_h_indices if h not in matched_prod]
    h_candidates = []
    if h_targets:
        unmapped_parent_h = []
        mapped_parent_h = []
        h2_partner = {}
        for i, (map_num, atomic_num) in enumerate(atoms_substrate):
            if atomic_num == 1 and map_num == 0 and i not in matched_sub:
                for j in range(len(be_substrate)):
                    if be_substrate[i, j] > 0 and i != j:
                        parent_map = int(atoms_substrate[j][0])
                        if parent_map == 0:
                            unmapped_parent_h.append(i)
                            if int(atoms_substrate[j][1]) == 1:
                                h2_partner[i] = j
                        elif j not in matched_sub:
                            mapped_parent_h.append(i)
                        break
        h2_taken = set()
        spread_unmapped = []
        deferred_unmapped = []
        for h_idx in unmapped_parent_h:
            partner = h2_partner.get(h_idx)
            if partner is not None and partner in h2_taken:
                deferred_unmapped.append(h_idx)
            else:
                spread_unmapped.append(h_idx)
                if partner is not None:
                    h2_taken.add(h_idx)
        h_candidates = spread_unmapped + deferred_unmapped + mapped_parent_h

    return s2p, matched_sub, matched_prod, h_candidates, h_targets
