#!/usr/bin/env python
"""
Batch mechanism finder — process a CSV or JSONL dataset in parallel.

Streams results to JSON Lines (one JSON object per line) for constant memory
usage and automatic resume on restart.

Usage:
    python batch.py data.csv --column MAPPED_RXN --output results.jsonl [--strict] [--workers 8] [-t 30]
    python batch.py data.csv --column RXN --output results.jsonl --mixed --remove_stereo
    python batch.py data.jsonl --output results.jsonl [-t 30]
"""

import argparse
import io
import json
import os
import re
import select
import signal
import sys
import time
from contextlib import redirect_stdout
from multiprocessing import Pool, cpu_count

import pandas as pd

from mechanism import analyze_reaction
from preprocessing.reaction_cleaner import ReactionCleaner

CHUNK_SIZE = 10_000  # CSV rows per chunk
MAP_RE = re.compile(r":(\d+)")


def _remove_stereo_smiles(smi):
    """Remove stereochemistry from a dot-separated SMILES using RDKit."""
    from rdkit import Chem

    frags = smi.split(".")
    clean = []
    for frag in frags:
        mol = Chem.MolFromSmiles(frag, sanitize=False)
        if mol is None:
            clean.append(frag)
            continue
        Chem.RemoveStereochemistry(mol)
        try:
            Chem.SanitizeMol(mol)
            clean.append(Chem.MolToSmiles(mol))
        except Exception:
            clean.append(frag)
    return ".".join(clean)


def normalize_reaction(rxn_smiles, mixed=False, remove_stereo=False):
    """Preprocess a reaction SMILES string.

    Args:
        mixed: If True, handle x>y>z format by merging everything before the
               last '>' as reactants: A>B>C -> A.B>>C
        remove_stereo: If True, remove all stereochemistry via RDKit.
    """
    if mixed:
        parts = rxn_smiles.split(">")
        if len(parts) >= 2:
            products = parts[-1]
            reactants = ".".join(p for p in parts[:-1] if p)
            rxn_smiles = f"{reactants}>>{products}"

    if remove_stereo:
        sides = rxn_smiles.split(">>")
        if len(sides) == 2:
            rxn_smiles = f"{_remove_stereo_smiles(sides[0])}>>{_remove_stereo_smiles(sides[1])}"
        else:
            rxn_smiles = _remove_stereo_smiles(rxn_smiles)

    return rxn_smiles


def _parse_reactants(rxn_smiles):
    """Return RDKit reactant templates, or None if parsing fails."""
    if ">>" not in rxn_smiles:
        return None
    try:
        from rdkit.Chem import rdChemReactions

        rxn = rdChemReactions.ReactionFromSmiles(rxn_smiles)
    except Exception:
        return None
    if rxn is None:
        return None
    return list(rxn.GetReactants())


def is_single_reactant_reaction(rxn_smiles):
    """Return True when RDKit sees exactly one reactant component."""
    reactants = _parse_reactants(rxn_smiles)
    return reactants is not None and len(reactants) == 1


def build_map_to_reactant(rxn_smiles):
    """Map atom-map numbers to original reactant component indices."""
    reactants = _parse_reactants(rxn_smiles)
    if reactants is None:
        return {}

    map_to_reactant = {}
    for reactant_idx, reactant in enumerate(reactants):
        for atom in reactant.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num == 0:
                continue
            previous = map_to_reactant.get(map_num)
            if previous is None and map_num in map_to_reactant:
                continue
            if previous is not None and previous != reactant_idx:
                map_to_reactant[map_num] = None
            else:
                map_to_reactant[map_num] = reactant_idx
    return map_to_reactant


def atom_label_map(label):
    """Extract the non-zero atom-map number from a serialized atom label."""
    maps = [int(value) for value in MAP_RE.findall(label)]
    nonzero_maps = [value for value in maps if value != 0]
    if not nonzero_maps:
        return None
    # For H:0~O:3, use the parent atom map, not the zero H map.
    return nonzero_maps[-1]


def iter_bond_step_atom_labels(step):
    """Yield atom labels from bond break/form entries in one elemental step."""
    for key in ("break", "form"):
        for pair in step.get(key, []):
            if isinstance(pair, list):
                yield from (str(atom) for atom in pair)


def step_is_single_molecule_bond_swap(step, map_to_reactant):
    """Return True if all bond changes in a step involve one reactant molecule."""
    reactant_indices = set()
    saw_bond_atom = False

    for label in iter_bond_step_atom_labels(step):
        saw_bond_atom = True
        map_num = atom_label_map(label)
        if map_num is None:
            return False
        reactant_idx = map_to_reactant.get(map_num)
        if reactant_idx is None:
            return False
        reactant_indices.add(reactant_idx)

    return saw_bond_atom and len(reactant_indices) == 1


def mechanism_has_single_molecule_bond_swap(mechanism, map_to_reactant):
    """Return True if any step in a mechanism is a single-molecule bond swap."""
    for step in mechanism.get("steps", []):
        if step_is_single_molecule_bond_swap(step, map_to_reactant):
            return True
    return False


def apply_output_filters(result, drop_single_reactant=False, drop_single_molecule_steps=False):
    """Apply optional dataset filters.

    Returns (result_or_none, stats). A None result means the datapoint should not
    be written. Mechanism filtering is per-mechanism: mechanisms containing any
    single-molecule bond-swap step are removed, and the datapoint is dropped only
    if no mechanisms remain.
    """
    stats = {
        "dropped_single_reactant": 0,
        "dropped_single_molecule_step": 0,
        "filtered_mechanisms": 0,
    }

    rxn_smiles = result.get("reaction_smiles_mapped") or result.get("reaction_smiles") or ""

    if drop_single_reactant and is_single_reactant_reaction(rxn_smiles):
        stats["dropped_single_reactant"] = 1
        return None, stats

    if not drop_single_molecule_steps:
        return result, stats

    mechanisms = result.get("mechanisms", [])
    if not mechanisms:
        return result, stats

    map_to_reactant = build_map_to_reactant(rxn_smiles)
    kept = []
    for mechanism in mechanisms:
        if mechanism_has_single_molecule_bond_swap(mechanism, map_to_reactant):
            stats["filtered_mechanisms"] += 1
        else:
            kept.append(mechanism)

    if len(kept) == len(mechanisms):
        return result, stats

    if not kept:
        stats["dropped_single_molecule_step"] = 1
        return None, stats

    result["mechanisms"] = kept
    result["num_mechanisms"] = len(kept)
    result["num_steps"] = len(kept[0].get("steps", [])) if kept else 0
    result["success"] = result["num_steps"] > 0
    return result, stats


def _run_analysis(rxn_smiles, strict):
    """Run analyze_reaction in the current process, return result or None."""
    try:
        with redirect_stdout(io.StringIO()):
            return analyze_reaction(
                rxn_smiles,
                verbose=False,
                max_mechanisms=10,
                strict=strict,
                json_output=True,
            )
    except Exception:
        return None


def _run_with_hard_timeout(rxn_smiles, strict, timeout):
    """Fork a child process and SIGKILL it if it exceeds the timeout.

    Unlike signal.alarm, this kills C-level hangs (e.g. stuck RDKit calls).
    """
    r_fd, w_fd = os.pipe()
    pid = os.fork()

    if pid == 0:
        # --- Child process ---
        os.close(r_fd)
        try:
            result = _run_analysis(rxn_smiles, strict)
            if result is not None:
                os.write(w_fd, json.dumps(result).encode())
        except Exception:
            pass
        finally:
            os.close(w_fd)
            os._exit(0)

    # --- Parent process ---
    os.close(w_fd)

    # Wait for child to finish or timeout (select blocks without busy-waiting)
    ready, _, _ = select.select([r_fd], [], [], timeout)

    if not ready:
        # Timeout — hard-kill the child
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(r_fd)
        return None

    # Child finished — read result from pipe
    chunks = []
    while True:
        chunk = os.read(r_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r_fd)
    os.waitpid(pid, 0)

    if chunks:
        try:
            return json.loads(b"".join(chunks))
        except Exception:
            return None
    return None


def process_row(args):
    """Worker function for a single reaction. Runs in subprocess."""
    (
        idx,
        rxn_smiles,
        metadata,
        strict,
        timeout,
        mixed,
        remove_stereo,
        clean_coo,
        drop_single_reactant,
        drop_single_molecule_steps,
    ) = args

    metadata["rxn_smiles_original"] = rxn_smiles
    rxn_smiles = normalize_reaction(rxn_smiles, mixed=mixed, remove_stereo=remove_stereo)

    empty_filter_stats = {
        "dropped_single_reactant": 0,
        "dropped_single_molecule_step": 0,
        "filtered_mechanisms": 0,
    }

    if drop_single_reactant and is_single_reactant_reaction(rxn_smiles):
        filter_stats = empty_filter_stats.copy()
        filter_stats["dropped_single_reactant"] = 1
        return idx, None, False, filter_stats

    coo_cleaned = False
    if clean_coo:
        try:
            cleaner = ReactionCleaner()
            from rdkit.Chem import rdChemReactions
            cleaned = cleaner.clean_coo_oxygens(rxn_smiles)
            # Round-trip both through RDKit so we only detect actual atom map
            # changes, not fragment reordering or re-canonicalization.
            canonical = rdChemReactions.ReactionToSmiles(rdChemReactions.ReactionFromSmiles(rxn_smiles))
            cleaned_canonical = rdChemReactions.ReactionToSmiles(rdChemReactions.ReactionFromSmiles(cleaned))
            if cleaned_canonical != canonical:
                coo_cleaned = True
            rxn_smiles = cleaned
        except Exception:
            pass  # keep original if cleaning fails

    if timeout > 0:
        result = _run_with_hard_timeout(rxn_smiles, strict, timeout)
    else:
        result = _run_analysis(rxn_smiles, strict)

    if result is None:
        result = {
            "reaction_smiles": rxn_smiles,
            "reaction_smiles_mapped": "",
            "success": False,
            "num_steps": 0,
            "num_mechanisms": 0,
            "mechanisms": [],
        }

    result["metadata"] = metadata
    result, filter_stats = apply_output_filters(
        result,
        drop_single_reactant=drop_single_reactant,
        drop_single_molecule_steps=drop_single_molecule_steps,
    )
    return idx, result, coo_cleaned, filter_stats


def count_lines(path):
    """Count lines in a file efficiently."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def filters_enabled(args):
    return args.drop_single_reactant or args.drop_single_molecule_steps


def ensure_resume_supported(args):
    if filters_enabled(args) and os.path.exists(args.output) and count_lines(args.output) > 0:
        print(
            "Error: resume is not supported when drop filters are enabled, "
            "because filtered output is not one line per input row. Use a new "
            "output path or remove the existing output file."
        )
        sys.exit(1)


def row_to_work_item(
    i,
    row,
    column,
    strict,
    timeout,
    mixed,
    remove_stereo,
    clean_coo,
    drop_single_reactant,
    drop_single_molecule_steps,
):
    """Convert a DataFrame row to a work tuple without accumulating."""
    rxn = str(row[column])
    meta = {k: v for k, v in row.items() if k != column}
    meta = {k: v.item() if hasattr(v, "item") else v for k, v in meta.items()}
    return (
        i,
        rxn,
        meta,
        strict,
        timeout,
        mixed,
        remove_stereo,
        clean_coo,
        drop_single_reactant,
        drop_single_molecule_steps,
    )


def work_iter(
    input_path,
    sep,
    column,
    strict,
    timeout,
    mixed,
    remove_stereo,
    clean_coo,
    drop_single_reactant,
    drop_single_molecule_steps,
    skip_n,
    total,
):
    """Yield work items from CSV chunks, skipping already-processed rows."""
    skipped = 0
    global_idx = 0
    for chunk in pd.read_csv(input_path, sep=sep, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            if skipped < skip_n:
                skipped += 1
                global_idx += 1
                continue
            yield row_to_work_item(
                global_idx,
                row,
                column,
                strict,
                timeout,
                mixed,
                remove_stereo,
                clean_coo,
                drop_single_reactant,
                drop_single_molecule_steps,
            )
            global_idx += 1


def process_jsonl_input(args):
    """Process JSONL input — only analyze reactions without mechanisms (streaming)."""
    ensure_resume_supported(args)

    # First pass: categorize records
    # For records needing processing, keep (line_num, mapped_smiles, metadata) — these
    # have empty mechanisms so they're small. Passthrough records (with mechanisms) are
    # only tracked by line number.
    print(f"Scanning {args.input}...")
    passthrough_lines = []  # line numbers to copy as-is
    to_process = []  # (line_num, mapped_smiles, metadata)
    total = 0

    with open(args.input) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            total += 1
            mechanisms = rec.get("mechanisms", [])
            mapped_smiles = rec.get("reaction_smiles_mapped", "")

            if mechanisms:
                passthrough_lines.append(i)
            elif mapped_smiles:
                # Has mapping but no mechanisms — needs processing
                meta = {
                    k: v
                    for k, v in rec.items()
                    if k
                    not in (
                        "mechanisms",
                        "reaction_smiles_mapped",
                        "num_steps",
                        "num_mechanisms",
                        "success",
                    )
                }
                to_process.append((i, mapped_smiles, meta))
            else:
                passthrough_lines.append(i)  # no mapping, can't process

    print(
        f"Total: {total}, pass-through: {len(passthrough_lines)}, "
        f"needs processing: {len(to_process)}"
    )

    if not to_process:
        print("All records already processed or unmapped.")
        if not filters_enabled(args):
            return

    # Resume support
    skip_n = 0
    passthrough_written = False
    if os.path.exists(args.output):
        existing = count_lines(args.output)
        if existing >= total:
            print("All rows already processed.")
            return
        if existing >= len(passthrough_lines):
            passthrough_written = True
            skip_n = existing - len(passthrough_lines)
            if skip_n > 0:
                print(
                    f"Resuming: {skip_n}/{len(to_process)} to-process already done, "
                    f"{len(to_process) - skip_n} remaining"
                )
        elif existing > 0:
            print(
                f"Warning: output has {existing} lines but expected >= "
                f"{len(passthrough_lines)} pass-through. Restarting."
            )
            os.remove(args.output)

    # Write passthrough records (second pass of input file, streaming)
    n_dropped_single_reactant = 0
    n_dropped_single_molecule_step = 0
    n_filtered_mechanisms = 0

    if not passthrough_written:
        passthrough_set = set(passthrough_lines)
        with open(args.input) as fin, open(args.output, "w") as fout:
            for i, line in enumerate(fin):
                if i in passthrough_set:
                    if filters_enabled(args):
                        rec = json.loads(line)
                        rec, filter_stats = apply_output_filters(
                            rec,
                            drop_single_reactant=args.drop_single_reactant,
                            drop_single_molecule_steps=args.drop_single_molecule_steps,
                        )
                        n_dropped_single_reactant += filter_stats["dropped_single_reactant"]
                        n_dropped_single_molecule_step += filter_stats["dropped_single_molecule_step"]
                        n_filtered_mechanisms += filter_stats["filtered_mechanisms"]
                        if rec is None:
                            continue
                        fout.write(json.dumps(rec) + "\n")
                    else:
                        fout.write(line)
        print(f"Wrote pass-through records from {len(passthrough_lines)} input records.")

    remaining = len(to_process) - skip_n
    if remaining <= 0:
        print("All rows already processed.")
        if filters_enabled(args):
            print(f"Dropped single-reactant records: {n_dropped_single_reactant}")
            print(f"Dropped records after single-molecule-step filter: {n_dropped_single_molecule_step}")
            print(f"Filtered mechanisms: {n_filtered_mechanisms}")
            print(f"Output: {args.output}")
        return

    # Build work items from to_process
    def work_iter():
        for j, (line_num, mapped_smiles, meta) in enumerate(to_process):
            if j < skip_n:
                continue
            yield (
                line_num,
                mapped_smiles,
                meta,
                args.strict,
                args.timeout,
                args.mixed,
                args.remove_stereo,
                args.clean_coo,
                args.drop_single_reactant,
                args.drop_single_molecule_steps,
            )

    # Process in parallel
    timeout_str = f", {args.timeout}s timeout" if args.timeout > 0 else ""
    print(
        f"Processing {remaining} reactions with {args.workers} workers{timeout_str}..."
    )

    done = 0
    n_success = 0
    n_coo_cleaned = 0
    t_start = time.time()

    with open(args.output, "a") as out_f, Pool(
        processes=args.workers, maxtasksperchild=500
    ) as pool:
        for _idx, result, coo_cleaned, filter_stats in pool.imap_unordered(
            process_row, work_iter(), chunksize=4
        ):
            n_dropped_single_reactant += filter_stats["dropped_single_reactant"]
            n_dropped_single_molecule_step += filter_stats["dropped_single_molecule_step"]
            n_filtered_mechanisms += filter_stats["filtered_mechanisms"]
            if result is not None:
                out_f.write(json.dumps(result) + "\n")
            done += 1
            if result is not None and result["success"]:
                n_success += 1
            if coo_cleaned:
                n_coo_cleaned += 1

            if done % 100 == 0 or done == remaining:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (remaining - done) / rate if rate > 0 else 0
                eta_m, eta_s = divmod(int(eta), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                pct = 100 * (len(passthrough_lines) + skip_n + done) / total
                print(
                    f"\r  [{len(passthrough_lines) + skip_n + done}/{total}] {pct:.1f}%  "
                    f"{rate:.0f} rxn/s  "
                    f"ok:{n_success}  "
                    f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s",
                    end="",
                    flush=True,
                )

    elapsed = time.time() - t_start
    print(f"\nDone: {n_success}/{done} successful in {elapsed:.0f}s")
    if args.clean_coo:
        print(f"COO oxygen swaps fixed: {n_coo_cleaned}")
    if filters_enabled(args):
        print(f"Dropped single-reactant records: {n_dropped_single_reactant}")
        print(f"Dropped records after single-molecule-step filter: {n_dropped_single_molecule_step}")
        print(f"Filtered mechanisms: {n_filtered_mechanisms}")
    print(f"Output: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process a CSV/JSONL of reactions into mechanism JSONL",
    )
    parser.add_argument("input", help="Input CSV or JSONL file")
    parser.add_argument("-c", "--column", help="Column containing reaction SMILES (required for CSV input)")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file (one JSON object per line)")
    parser.add_argument("--strict", action="store_true", help="Validate product is subset of substrate+R")
    parser.add_argument("-w", "--workers", type=int, default=cpu_count(), help=f"Number of worker processes (default: {cpu_count()})")
    parser.add_argument("-t", "--timeout", type=int, default=0, help="Timeout per reaction in seconds (default: no limit)")
    parser.add_argument("--sep", default=",", help="CSV separator (default: comma)")
    parser.add_argument("--mixed", action="store_true", help="Handle x>y>z format: merge everything left of last > as reactants")
    parser.add_argument("--remove_stereo", action="store_true", help="Strip all stereochemistry (@, @@, /, \\) from both sides")
    parser.add_argument("--clean-coo", action="store_true", help="Fix swapped C(=O)-O oxygen atom mappings before analysis")
    parser.add_argument("--drop-single-reactant", action="store_true", help="Do not write reactions with exactly one RDKit reactant component")
    parser.add_argument("--drop-single-molecule-steps", action="store_true", help="Remove mechanisms containing a bond-swap step entirely within one reactant molecule; drop the record if no mechanisms remain")

    args = parser.parse_args()

    # JSONL input: separate code path
    input_ext = args.input.lower().rsplit(".", 1)[-1]
    if input_ext == "jsonl":
        process_jsonl_input(args)
        return

    # CSV input: validate column
    if not args.column:
        print("Error: --column is required for CSV input.")
        sys.exit(1)

    ensure_resume_supported(args)

    # Validate column exists (read just the header)
    header = pd.read_csv(args.input, sep=args.sep, nrows=0)
    if args.column not in header.columns:
        print(f"Error: column '{args.column}' not found. Available: {list(header.columns)}")
        sys.exit(1)

    # Count total rows (chunked, no full load)
    total = 0
    for chunk in pd.read_csv(args.input, sep=args.sep, chunksize=CHUNK_SIZE):
        total += len(chunk)

    # Resume support: count existing output lines
    skip_n = 0
    if os.path.exists(args.output):
        skip_n = count_lines(args.output)
        if skip_n > 0:
            print(f"Resuming: {skip_n}/{total} already done, {total - skip_n} remaining")

    remaining = total - skip_n
    if remaining <= 0:
        print("All rows already processed.")
        return

    timeout_str = f", {args.timeout}s timeout" if args.timeout > 0 else ""
    print(f"Processing {remaining} reactions ({total} total) with {args.workers} workers{timeout_str}...")

    done = 0
    n_success = 0
    n_coo_cleaned = 0
    t_start = time.time()

    n_dropped_single_reactant = 0
    n_dropped_single_molecule_step = 0
    n_filtered_mechanisms = 0

    work = work_iter(
        args.input,
        args.sep,
        args.column,
        args.strict,
        args.timeout,
        args.mixed,
        args.remove_stereo,
        args.clean_coo,
        args.drop_single_reactant,
        args.drop_single_molecule_steps,
        skip_n,
        total,
    )

    with open(args.output, "a") as out_f, Pool(processes=args.workers, maxtasksperchild=500) as pool:
        for _idx, result, coo_cleaned, filter_stats in pool.imap_unordered(process_row, work, chunksize=4):
            n_dropped_single_reactant += filter_stats["dropped_single_reactant"]
            n_dropped_single_molecule_step += filter_stats["dropped_single_molecule_step"]
            n_filtered_mechanisms += filter_stats["filtered_mechanisms"]
            if result is not None:
                out_f.write(json.dumps(result) + "\n")
            done += 1
            if result is not None and result["success"]:
                n_success += 1
            if coo_cleaned:
                n_coo_cleaned += 1

            if done % 100 == 0 or done == remaining:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (remaining - done) / rate if rate > 0 else 0
                eta_m, eta_s = divmod(int(eta), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                pct = 100 * (skip_n + done) / total
                print(
                    f"\r  [{skip_n + done}/{total}] {pct:.1f}%  "
                    f"{rate:.0f} rxn/s  "
                    f"ok:{n_success}  "
                    f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s",
                    end="",
                    flush=True,
                )

    elapsed = time.time() - t_start
    print(f"\nDone: {n_success}/{done} successful in {elapsed:.0f}s")
    if args.clean_coo:
        print(f"COO oxygen swaps fixed: {n_coo_cleaned}")
    if filters_enabled(args):
        print(f"Dropped single-reactant records: {n_dropped_single_reactant}")
        print(f"Dropped records after single-molecule-step filter: {n_dropped_single_molecule_step}")
        print(f"Filtered mechanisms: {n_filtered_mechanisms}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
