#!/usr/bin/env python
"""
Mechanism analysis API and command-line entry point.

Implemented across focused modules:

- ``map_assignment``: atom-map number assignment for partially mapped reactions
- ``atom_labels``: atom/step label formatting
- ``r_cleanup``: R matrix cleanup (Kekule noise, byproduct LP, radicals)
- ``mech_steps``: elemental step primitives (extract, validate, apply)
- ``mech_search``: BFS/DFS mechanism search and intermediates
- ``mech_serialize``: JSON serialization of mechanisms
- ``mech_analyze``: end-to-end ``analyze_reaction`` and the CLI
"""

from atom_labels import (  # noqa: F401
    atom_label,
    atom_label_with_context,
    format_change,
    format_step,
    print_step_details,
)

from map_assignment import (  # noqa: F401
    assign_full_mapping,
    find_best_unmapped_assignment,
)

from mech_steps import (  # noqa: F401
    Change,
    Step,
    apply_step,
    contribution,
    extract_changes,
    find_active_atoms,
    find_subsets_with_sum,
    generate_elemental_steps,
    is_valid_be,
    is_valid_electron_flow,
    is_zero,
    matrix_key,
    remove_step,
)

from mech_search import (  # noqa: F401
    add_intermediate_bond,
    check_decomposable,
    find_all_minimum_mechanisms,
    find_intermediate_bonds,
    find_minimum_mechanism,
)

from mech_serialize import serialize_mechanism  # noqa: F401

from r_cleanup import (  # noqa: F401
    clean_aromatic_kekule_noise,
    eliminate_byproduct_lp,
    find_aromatic_rings,
    find_radicals,
    fix_radicals,
    generate_pairings,
)

from mech_analyze import analyze_reaction, main  # noqa: F401

from be_core import PT  # noqa: F401

if __name__ == "__main__":
    main()
