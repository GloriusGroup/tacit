"""
End-to-end reaction mechanism analysis and CLI.

``analyze_reaction`` ties the whole pipeline together: parse the reaction
SMILES, complete the atom mapping, compute and clean the R matrix, handle
radicals and intermediate bonds, search for minimum-step mechanisms, and
optionally serialize the result to JSON.

Usage:
    python mechanism.py "substrate>>product"
"""

import argparse
from itertools import combinations

import numpy as np

from atom_labels import format_change, format_step, print_step_details
from atom_matching import match_atoms_by_map
from be_components import decompose_molecules
from be_core import PT, matrices_to_smiles, smiles_to_matrices
from be_hash import is_subset
from map_assignment import assign_full_mapping, find_best_unmapped_assignment
from mech_search import (
    check_decomposable,
    find_all_minimum_mechanisms,
    find_intermediate_bonds,
    find_minimum_mechanism,
)
from mech_serialize import serialize_mechanism
from mech_steps import apply_step, extract_changes, find_active_atoms
from r_cleanup import (
    clean_aromatic_kekule_noise,
    eliminate_byproduct_lp,
    find_aromatic_rings,
    find_radicals,
    fix_radicals,
    generate_pairings,
)
from r_matrix import apply_r_matrix, find_r_matrix


def analyze_reaction(
    rxn_smiles: str,
    verbose: bool = False,
    max_mechanisms: int = 10,
    strict: bool = True,
    json_output: bool = False,
):
    """
    Analyze a reaction and find all minimum-step mechanisms.

    Pipeline:

    1. Parse the reaction SMILES into kekulized BE matrices (integer bond
       orders).
    2. Complete the atom mapping: choose unmapped heavy-atom pairings that
       minimize R matrix cost, then assign map numbers to all remaining
       unmapped non-H atoms.
    3. Compute the R matrix; in strict mode verify the product is a multiset
       subset of substrate+R.
    4. Remove Kekule noise: compute the R matrix from non-kekulized
       (aromatic) BE matrices and truncate to integers; if that conserves
       electrons it replaces the kekulized R.  Otherwise fall back to
       zeroing ring bond changes that sum to zero (``clean_aromatic_kekule_noise``).
    5. Eliminate lone pair changes on byproduct atoms by forming
       inter-byproduct bonds.
    6. If radical byproducts exist, try every pairing and keep the one
       yielding the shortest mechanism.
    7. Search for a minimum mechanism (BFS); if none is found, retry with
       one and then two intermediate bonds between active atoms (a per-atom
       balance constraint can require intermediates even when the change sum
       is divisible by 4).
    8. Enumerate all mechanisms with the minimum step count and print them.

    Args:
        rxn_smiles: Reaction SMILES "substrate>>product"
        verbose: Print detailed step information
        max_mechanisms: Maximum mechanisms to enumerate
        strict: Validate that product is a subset of substrate+R (catches bad mappings)
        json_output: Return serialized mechanism as dict

    Returns:
        Serialized mechanism dict if json_output is True and analysis
        succeeds, otherwise None.
    """
    print("=" * 70)
    print(f"Reaction: {rxn_smiles[:70]}{'...' if len(rxn_smiles) > 70 else ''}")
    print("=" * 70)

    if ">>" not in rxn_smiles:
        print("Error: Expected '>>' in reaction SMILES")
        return

    sub_smi, prod_smi = rxn_smiles.split(">>")

    try:
        be_sub, atoms_sub = smiles_to_matrices(
            sub_smi, kekulize=True, include_hydrogens=True
        )
        be_prod, atoms_prod = smiles_to_matrices(
            prod_smi, kekulize=True, include_hydrogens=True
        )
    except Exception as e:
        print(f"Error parsing SMILES: {e}")
        return

    forced_matches = find_best_unmapped_assignment(sub_smi, prod_smi, atoms_sub)

    sub_mapped, prod_mapped, atoms_sub = assign_full_mapping(
        sub_smi, prod_smi, atoms_sub, forced_heavy_matches=forced_matches
    )
    if sub_mapped != sub_smi or prod_mapped != prod_smi:
        sub_smi, prod_smi = sub_mapped, prod_mapped
        try:
            be_sub, atoms_sub = smiles_to_matrices(
                sub_smi, kekulize=True, include_hydrogens=True
            )
            be_prod, atoms_prod = smiles_to_matrices(
                prod_smi, kekulize=True, include_hydrogens=True
            )
        except Exception as e:
            print(f"Error parsing fully-mapped SMILES: {e}")
            return

    s2p, byp_idx, _ = match_atoms_by_map(atoms_sub, atoms_prod, be_sub, be_prod)
    mismatches = []
    for si, pi in s2p.items():
        sz = int(atoms_sub[si][1])
        pz = int(atoms_prod[pi][1])
        if sz != pz:
            sm = int(atoms_sub[si][0])
            se = PT.GetElementSymbol(sz)
            pe = PT.GetElementSymbol(pz)
            mismatches.append((sm, se, pe))

    if mismatches:
        print("\nWarning: Element mismatch in atom mapping (bad mapper output):")
        for m, se, pe in mismatches:
            print(f"  Atom map {m}: {se} (substrate) -> {pe} (product)")

    r_matrix, valid = find_r_matrix(be_sub, atoms_sub, be_prod, atoms_prod)

    if not valid:
        print("Error: Could not find valid R matrix")
        return

    if strict:
        be_result, atoms_result = apply_r_matrix(be_sub, atoms_sub, r_matrix)
        if not is_subset(be_prod, atoms_prod, be_result, atoms_result):
            print("Error: Product is not a subset of substrate+R (bad atom mapping?)")
            return

    try:
        be_sub_arom, _ = smiles_to_matrices(
            sub_smi, kekulize=False, include_hydrogens=True
        )
        be_prod_arom, _ = smiles_to_matrices(
            prod_smi, kekulize=False, include_hydrogens=True
        )
        r_arom, valid_arom = find_r_matrix(
            be_sub_arom, atoms_sub, be_prod_arom, atoms_prod
        )
        if valid_arom:
            r_clean = np.trunc(r_arom)
            r_clean = (r_clean + r_clean.T) / 2
            if abs(np.sum(r_clean)) < 1e-6:
                r_matrix = r_clean
    except Exception:
        pass

    sub_rings = find_aromatic_rings(sub_smi)
    if sub_rings:
        r_matrix = clean_aromatic_kekule_noise(r_matrix, atoms_sub, sub_rings)

    r_matrix = eliminate_byproduct_lp(r_matrix, atoms_sub, byproduct_idx=byp_idx)

    radicals = find_radicals(r_matrix)

    if radicals:
        if len(radicals) % 2 != 0:
            print(
                f"\nError: Odd number of radical byproducts ({len(radicals)}) - cannot pair"
            )
            return

        print(f"\nFound {len(radicals)} radical byproducts - trying pairings...")
        pairings = generate_pairings(radicals)
        print(f"  {len(pairings)} possible pairing(s)")

        best_steps = None
        best_r = None
        best_pairing = None
        best_intermediates = None

        for pairing in pairings:
            r_fixed = fix_radicals(r_matrix, pairing)
            decomposable, _ = check_decomposable(r_fixed, atoms=atoms_sub)

            if decomposable:
                steps, count = find_minimum_mechanism(be_sub, r_fixed, atoms=atoms_sub)
                if steps is not None:
                    if best_steps is None or count < len(best_steps):
                        best_steps = steps
                        best_r = r_fixed
                        best_pairing = pairing
                        best_intermediates = None
            else:
                intermediate_candidates = find_intermediate_bonds(
                    r_fixed, be_sub, atoms_sub
                )
                for intermediates in intermediate_candidates:
                    steps, count = find_minimum_mechanism(
                        be_sub, r_fixed, intermediates=intermediates, atoms=atoms_sub
                    )
                    if steps is not None:
                        if best_steps is None or count < len(best_steps):
                            best_steps = steps
                            best_r = r_fixed
                            best_pairing = pairing
                            best_intermediates = intermediates
                        break

        if best_steps is None:
            if mismatches:
                print(
                    "No valid mechanism found (likely caused by element mismatch in atom mapping)"
                )
            else:
                print("No valid mechanism found for any radical pairing!")
            return

        r_matrix = best_r
        steps = best_steps
        min_count = len(best_steps)
        used_intermediates = best_intermediates
        print(
            f"\n  Best pairing: {[(format_change((i, j, 1), atoms_sub).split()[0]) for i, j in best_pairing]}"
        )
        if best_intermediates:
            int_desc = [
                format_change((i, j, 1), atoms_sub).split()[0]
                for i, j in best_intermediates
            ]
            print(f"  With intermediate: {int_desc}")

    else:
        decomposable, msg = check_decomposable(r_matrix, atoms=atoms_sub)
        print(f"\n{msg}")

        used_intermediates = None

        if not decomposable:
            intermediate_candidates = find_intermediate_bonds(
                r_matrix, be_sub, atoms_sub
            )
            if intermediate_candidates:
                print(
                    f"\nTrying {len(intermediate_candidates)} intermediate bond candidate(s)..."
                )
                for intermediates in intermediate_candidates:
                    int_desc = [
                        format_change((i, j, 1), atoms_sub).split()[0]
                        for i, j in intermediates
                    ]
                    steps_found, count = find_minimum_mechanism(
                        be_sub, r_matrix, intermediates=intermediates, atoms=atoms_sub
                    )
                    if steps_found is not None:
                        print(f"  Found mechanism with intermediate: {int_desc}")
                        used_intermediates = intermediates
                        decomposable = True
                        break

            if not decomposable:
                print("Cannot decompose into elemental steps!")
                return

        print(f"\nSearching for minimum mechanism...")
        steps, min_count = find_minimum_mechanism(
            be_sub, r_matrix, intermediates=used_intermediates, atoms=atoms_sub
        )

        if steps is None and used_intermediates is None:
            print("No direct mechanism found — trying with intermediate bonds...")
            active = find_active_atoms(r_matrix)
            candidates = []
            for i in active:
                for j in active:
                    if i >= j:
                        continue
                    if be_sub[i, j] > 1e-6:
                        continue
                    if abs(r_matrix[i, j]) > 1e-6:
                        continue
                    candidates.append((i, j))

            print(f"  Trying {len(candidates)} single intermediate candidate(s)...")
            for bond in candidates:
                intermediates = [bond]
                steps_found, count = find_minimum_mechanism(
                    be_sub, r_matrix, intermediates=intermediates, atoms=atoms_sub
                )
                if steps_found is not None:
                    int_desc = [
                        format_change((i, j, 1), atoms_sub).split()[0]
                        for i, j in intermediates
                    ]
                    print(f"  Found mechanism with intermediate: {int_desc}")
                    used_intermediates = intermediates
                    steps = steps_found
                    min_count = count
                    break

            if steps is None:
                pair_candidates = list(combinations(candidates, 2))
                print(f"  Trying {len(pair_candidates)} pairs of 2 intermediates...")
                for bond_pair in pair_candidates:
                    intermediates = list(bond_pair)
                    steps_found, count = find_minimum_mechanism(
                        be_sub, r_matrix, intermediates=intermediates, atoms=atoms_sub
                    )
                    if steps_found is not None:
                        int_desc = [
                            format_change((i, j, 1), atoms_sub).split()[0]
                            for i, j in intermediates
                        ]
                        print(f"  Found mechanism with 2 intermediates: {int_desc}")
                        used_intermediates = intermediates
                        steps = steps_found
                        min_count = count
                        break

        if steps is None:
            print("No valid mechanism found!")
            print("(All paths lead to invalid intermediate BE matrices)")
            return

    sources, sinks = extract_changes(r_matrix, atoms=atoms_sub)
    print(f"\nR matrix changes:")
    print(f"  Sources: {', '.join(format_change(c, atoms_sub) for c in sources)}")
    print(f"  Sinks:   {', '.join(format_change(c, atoms_sub) for c in sinks)}")
    if used_intermediates:
        int_desc = [
            format_change((i, j, 1), atoms_sub).split()[0]
            for i, j in used_intermediates
        ]
        print(f"  Intermediates: {', '.join(int_desc)}")

    print(f"\nMinimum steps required: {min_count}")

    print(f"\nExample minimum mechanism:")
    be_current = be_sub.copy()

    start_mols = decompose_molecules(be_current, atoms_sub)
    start_smiles = [matrices_to_smiles(m, a) for m, a in start_mols]
    print(f"\n  Start: {'.'.join(start_smiles)}")

    for i, step in enumerate(steps):
        if verbose:
            print_step_details(step, atoms_sub, i + 1)
        else:
            print(f"\n  Step {i+1}: {format_step(step, atoms_sub)}")

        be_current = apply_step(be_current, step)

        int_mols = decompose_molecules(be_current, atoms_sub)
        int_smiles = [matrices_to_smiles(m, a) for m, a in int_mols]
        label = "Product" if i == len(steps) - 1 else f"Int {i+1}"
        print(f"  {label}:  {'.'.join(int_smiles)}")

    print(f"\nEnumerating all {min_count}-step mechanisms...")
    all_mechs = find_all_minimum_mechanisms(
        be_sub,
        r_matrix,
        min_count,
        max_mechanisms,
        atoms=atoms_sub,
        intermediates=used_intermediates,
    )

    print(f"Found {len(all_mechs)} mechanism(s)")

    if len(all_mechs) > 1:
        print(f"\nAll mechanisms:")
        for m_idx, mech in enumerate(all_mechs):
            print(f"\n  Mechanism {m_idx + 1}:")
            for i, step in enumerate(mech):
                print(f"    Step {i+1}: {format_step(step, atoms_sub)}")

    if len(all_mechs) >= max_mechanisms:
        print(f"\n  (stopped at {max_mechanisms} - there may be more)")

    if json_output:
        mapped_rxn = f"{sub_smi}>>{prod_smi}"
        result = serialize_mechanism(
            rxn_smiles,
            all_mechs,
            be_sub,
            atoms_sub,
            intermediates=used_intermediates,
            mapped_rxn=mapped_rxn,
        )
        return result

    return None


def main():
    """CLI entry point: analyze a reaction SMILES and optionally save JSON."""
    parser = argparse.ArgumentParser(
        description="Find reaction mechanisms by decomposing R matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mechanism.py "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3]"
  python mechanism.py "reaction_smiles" --verbose
  python mechanism.py "reaction_smiles" --max 100
        """,
    )
    parser.add_argument("reaction", help="Reaction SMILES (substrate>>product)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")
    parser.add_argument("--max", type=int, default=10, help="Max mechanisms to find")
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable product subset validation",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="FILE",
        help="Save mechanism to JSON file",
    )

    args = parser.parse_args()
    result = analyze_reaction(
        args.reaction,
        verbose=args.verbose,
        max_mechanisms=args.max,
        strict=args.strict,
        json_output=args.json is not None,
    )

    if args.json and result:
        import json

        with open(args.json, "w") as f:
            json.dump(result, f)
        print(f"\nMechanism saved to {args.json}")


if __name__ == "__main__":
    main()
