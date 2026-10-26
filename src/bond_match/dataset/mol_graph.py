"""Molecule/graph conversions and graph feature extraction.

Provides conversions between RDKit molecules, NetworkX graphs, and
bond-electron (BE) matrices, plus node/edge feature extraction and
lone-pair (LP) node augmentation for the ranking model.
"""

import torch
import numpy as np
from rdkit import Chem
import networkx as nx
import torch

from bond_match.dataset.constants import (
    BOND_ORDER_TO_TYPE,
    PT,
    _ELEMENT_TO_ATOMIC_NUM,
)

def canonicalize_mol_ordering(mol, return_mapping=False):
    """Canonicalize atom ordering so equivalent SMILES produce identical representations.

    Atom-map numbers are saved and stripped before ranking (they would
    otherwise affect the canonical order), the canonical atom order is
    obtained via ``CanonicalRankAtoms(breakTies=True)``, atoms are
    renumbered (``RenumberAtoms`` expects ``new_order[new_idx] = old_idx``),
    and the atom maps are re-applied at their new positions.

    Args:
        mol: RDKit Mol object to canonicalize.
        return_mapping: If True, also return the ranks list where
            ranks[old_idx] = new_idx after renumbering.

    Returns:
        mol or (mol, ranks) if return_mapping=True.
    """
    saved_maps = {}
    for atom in mol.GetAtoms():
        m = atom.GetAtomMapNum()
        if m:
            saved_maps[atom.GetIdx()] = m
            atom.SetAtomMapNum(0)

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    inv_order = [0] * len(ranks)
    for old_idx, rank in enumerate(ranks):
        inv_order[rank] = old_idx

    mol = Chem.RenumberAtoms(mol, inv_order)

    for old_idx, map_num in saved_maps.items():
        new_idx = ranks[old_idx]
        mol.GetAtomWithIdx(new_idx).SetAtomMapNum(map_num)

    if return_mapping:
        return mol, ranks
    return mol

def connect_components_with_head(G):
    """Connect all components of a molecular graph through a virtual head node.

    A single head node (atomic number 0, symbol ``*``) is added and joined
    by zero-order edges (``bond_order=0.0``) to every heavy atom
    (atomic number > 1) of each connected component. Components without
    heavy atoms (e.g. H2) are connected through all of their atoms instead.
    Nodes with atomic number 0 in the input are discarded, so re-applying
    this function replaces any previous head node.

    Args:
        G: NetworkX graph with ``atomic_number`` node attributes.

    Returns:
        New NetworkX graph containing the real atoms plus the head node.
    """
    real_nodes = [n for n, d in G.nodes(data=True) if d.get("atomic_number", 0) > 0]

    if not real_nodes:
        return G

    sub_G = G.subgraph(real_nodes).copy()
    components = list(nx.connected_components(sub_G))

    new_G = sub_G.copy()

    head_node_idx = max(real_nodes) + 1
    new_G.add_node(
        head_node_idx,
        atomic_number=0,
        symbol="*",
        formal_charge=0,
        map_number=0,
        is_aromatic=False,
    )

    for comp in components:
        heavy_atoms = []
        for node in comp:
            if new_G.nodes[node].get("atomic_number", 1) > 1:
                heavy_atoms.append(node)

        if heavy_atoms:
            for node in heavy_atoms:
                new_G.add_edge(head_node_idx, node, bond_order=0.0)
        else:
            for node in comp:
                new_G.add_edge(head_node_idx, node, bond_order=0.0)

    return new_G

def mol_to_nx(mol):
    """Convert an RDKit Mol to a NetworkX graph with a virtual head node.

    Node attributes are ``atomic_number``, ``map_number``, ``symbol``, and
    ``formal_charge``. Edges carry ``bond_order`` (RDKit bond order as a
    double; dative bonds are encoded as 5.0). The result is connected via
    :func:`connect_components_with_head`.

    Args:
        mol: RDKit Mol object.

    Returns:
        NetworkX graph of the molecule including the head node.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_number=atom.GetAtomicNum(),
            map_number=atom.GetAtomMapNum(),
            symbol=atom.GetSymbol(),
            formal_charge=atom.GetFormalCharge(),
        )

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        if bond.GetBondType() == Chem.BondType.DATIVE:
            order = 5.0
        G.add_edge(u, v, bond_order=order)

    G = connect_components_with_head(G)

    return G

def nx_to_mol(G, remove_virtual=False):
    """Convert a NetworkX graph back to an RDKit Mol.

    Restores atom-map numbers and formal charges and maps ``bond_order``
    values to RDKit bond types (1.5 to aromatic, 5.0 to dative, unknown
    orders to single). Sanitization is attempted to fix implicit valences
    and aromaticity; if it fails, the unsanitized molecule is returned.

    Args:
        G: NetworkX graph with atom attributes and ``bond_order`` edges.
        remove_virtual: If True, skip virtual nodes (atomic number 0) and
            their incident edges.

    Returns:
        RDKit RWMol built from the graph.
    """
    mol = Chem.RWMol()
    node_to_idx = {}

    for node, data in G.nodes(data=True):
        if remove_virtual and data.get("atomic_number", 0) == 0:
            continue
        a = Chem.Atom(data["atomic_number"])
        a.SetAtomMapNum(data.get("map_number", 0))
        a.SetFormalCharge(data.get("formal_charge", 0))
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    for u, v, data in G.edges(data=True):
        if remove_virtual and (
            G.nodes[u].get("atomic_number", 0) == 0
            or G.nodes[v].get("atomic_number", 0) == 0
        ):
            continue
        if u in node_to_idx and v in node_to_idx:
            order = data.get("bond_order", 1.0)
            if order == 1.0:
                b_type = Chem.BondType.SINGLE
            elif order == 2.0:
                b_type = Chem.BondType.DOUBLE
            elif order == 3.0:
                b_type = Chem.BondType.TRIPLE
            elif order == 1.5:
                b_type = Chem.BondType.AROMATIC
            elif order == 5.0:
                b_type = Chem.BondType.DATIVE
            else:
                b_type = Chem.BondType.SINGLE

            mol.AddBond(node_to_idx[u], node_to_idx[v], b_type)

    try:
        Chem.SanitizeMol(mol)
    except:
        pass
    return mol

def smiles_match(smi1, smi2):
    """Compare two SMILES strings for graph equality.

    Stereochemistry is removed before comparison (the graph representation
    does not track it) and both molecules are canonicalized.

    Args:
        smi1: First SMILES string.
        smi2: Second SMILES string.

    Returns:
        True if both parse and their canonical stereo-free SMILES match.
    """
    if not smi1 or not smi2:
        return False

    try:
        m1 = Chem.MolFromSmiles(smi1)
        m2 = Chem.MolFromSmiles(smi2)
        if not m1 or not m2:
            return False

        Chem.RemoveStereochemistry(m1)
        Chem.RemoveStereochemistry(m2)

        return Chem.MolToSmiles(m1, canonical=True) == Chem.MolToSmiles(
            m2, canonical=True
        )
    except:
        return False

def find_atom_by_identifier(G, atom_id):
    """Resolve an atom identifier dict to a graph node index.

    Mapped atoms (``map > 0``) are matched by atom-map number and atomic
    number. Unmapped atoms are matched by their environment hash: for each
    candidate node with the right element and no atom map, a hash of the
    form ``"{symbol}_bonded_to_" + sorted("{nbr_symbol}:{nbr_map}({bond_order})")``
    over non-virtual neighbors is computed and compared with
    ``atom_id["env_hash"]``.

    Args:
        G: NetworkX molecular graph.
        atom_id: Dict with keys ``map``, ``atomic_num``, and optionally
            ``env_hash``.

    Returns:
        Matching node index, or None if no match is found.
    """
    if atom_id["map"] > 0:
        for node, data in G.nodes(data=True):
            if (
                data.get("map_number") == atom_id["map"]
                and data.get("atomic_number") == atom_id["atomic_num"]
            ):
                return node
        return None

    env_hash = atom_id.get("env_hash")
    if not env_hash:
        return None

    candidates = sorted(
        [
            node
            for node, data in G.nodes(data=True)
            if data["atomic_number"] == atom_id["atomic_num"]
            and data.get("map_number", 0) == 0
        ]
    )

    pt = Chem.GetPeriodicTable()
    for node in candidates:
        neighbors = []
        for nbr in G.neighbors(node):
            nbr_data = G.nodes[nbr]

            if nbr_data.get("atomic_number", 0) == 0:
                continue

            nbr_symbol = pt.GetElementSymbol(nbr_data["atomic_number"])
            nbr_map = nbr_data.get("map_number", 0)
            bond_order = G[node][nbr]["bond_order"]
            neighbors.append(f"{nbr_symbol}:{nbr_map}({bond_order})")

        symbol = pt.GetElementSymbol(atom_id["atomic_num"])
        computed_hash = f"{symbol}_bonded_to_" + "_".join(sorted(neighbors))

        if computed_hash == env_hash:
            return node
    return None

def parse_v3_atom_label(label_str):
    """Parse a v3 blockspace atom label string.

    Examples:
        ``"C:12"`` parses to ``{"element": "C", "map": 12,
        "is_unmapped_h": False}``.
        ``"H:0~N:15"`` parses to ``{"element": "H", "map": 0,
        "is_unmapped_h": True, "parent_element": "N", "parent_map": 15}``.

    Args:
        label_str: Atom label of the form ``"Elem:map"`` or
            ``"H:0~ParentElem:parent_map"``.

    Returns:
        Dict describing the atom and, for unmapped hydrogens, its parent.
    """
    if "~" in label_str:
        h_part, parent_part = label_str.split("~", 1)
        h_elem, h_map = h_part.split(":")
        p_elem, p_map = parent_part.split(":")
        return {
            "element": h_elem,
            "map": int(h_map),
            "is_unmapped_h": True,
            "parent_element": p_elem,
            "parent_map": int(p_map),
        }
    else:
        elem, map_num = label_str.split(":")
        return {
            "element": elem,
            "map": int(map_num),
            "is_unmapped_h": False,
        }

def _get_atomic_num(symbol):
    """Return the atomic number for an element symbol (0 if unknown), using a lazily built cache."""
    if not _ELEMENT_TO_ATOMIC_NUM:
        pt = Chem.GetPeriodicTable()
        for z in range(1, 119):
            try:
                _ELEMENT_TO_ATOMIC_NUM[pt.GetElementSymbol(z)] = z
            except:
                pass
    return _ELEMENT_TO_ATOMIC_NUM.get(symbol, 0)

def find_atom_by_v3_label(G, label_info, h_cache=None):
    """Resolve a parsed v3 atom label to a NetworkX node index.

    Mapped atoms (map > 0) are located by atom-map number and atomic
    number. Unmapped hydrogens (``H:0~Parent:N``) are located by first
    finding the parent atom by map, then selecting an unused hydrogen
    neighbor.

    Args:
        G: NetworkX molecular graph.
        label_info: Parsed label dict from :func:`parse_v3_atom_label`.
        h_cache: Optional dict mapping ``(parent_map, parent_element)`` to a
            set of already-used hydrogen node indices, ensuring that
            repeated ``H:0~X:N`` labels within one step resolve to distinct
            hydrogens.

    Returns:
        Matching node index, or None if no match is found.
    """
    if not label_info["is_unmapped_h"]:
        target_z = _get_atomic_num(label_info["element"])
        target_map = label_info["map"]
        for node, data in G.nodes(data=True):
            if (
                data.get("map_number") == target_map
                and data.get("atomic_number") == target_z
            ):
                return node
        return None
    else:
        parent_z = _get_atomic_num(label_info["parent_element"])
        parent_map = label_info["parent_map"]

        parent_node = None
        for node, data in G.nodes(data=True):
            if (
                data.get("map_number") == parent_map
                and data.get("atomic_number") == parent_z
            ):
                parent_node = node
                break

        if parent_node is None:
            return None

        cache_key = (parent_map, label_info["parent_element"])
        used = set()
        if h_cache is not None and cache_key in h_cache:
            used = h_cache[cache_key]

        for nbr in G.neighbors(parent_node):
            if G.nodes[nbr].get("atomic_number") == 1 and nbr not in used:
                if h_cache is not None:
                    h_cache.setdefault(cache_key, set()).add(nbr)
                return nbr

        return None

def _be_to_nx(be_matrix, atoms_array, catalyst_flags=None):
    """Convert a BE matrix back to a NetworkX graph for GNN feature extraction.

    Formal charges are computed vectorized from the BE matrix as
    ``charge = valence_electrons - off_diagonal_row_sum - diagonal`` (the
    diagonal holds lone-pair electron counts). Edges are taken from the
    strictly upper-triangular non-zero entries. Aromaticity is detected by
    building a temporary RDKit molecule from the BE matrix and marking
    aromatic bonds with ``bond_order=1.5``, matching the
    :func:`mol_to_nx` convention; failures during this step are ignored.
    The virtual head node is added via
    :func:`connect_components_with_head` (with ``is_catalyst=0`` when
    catalyst flags are supplied).

    Args:
        be_matrix: (n, n) BE matrix.
        atoms_array: (n, 2) array with [map_number, atomic_number] per atom.
        catalyst_flags: Optional dict {node_idx: 0/1} for catalyst annotation.

    Returns:
        NetworkX graph with head node, ready for get_graph_features_nx.
    """
    n = len(atoms_array)
    be = np.asarray(be_matrix, dtype=np.float64)
    G = nx.Graph()

    bond_sums = be.sum(axis=1) - np.diag(be)
    lps = np.diag(be)

    for i in range(n):
        map_num = int(atoms_array[i][0])
        z = int(atoms_array[i][1])
        v = PT.GetNOuterElecs(z)
        G.add_node(
            i,
            atomic_number=z,
            map_number=map_num,
            symbol=PT.GetElementSymbol(z),
            formal_charge=int(v - bond_sums[i] - lps[i]),
        )
        if catalyst_flags is not None:
            G.nodes[i]["is_catalyst"] = catalyst_flags.get(i, 0)

    rows, cols = np.where(np.triu(be, k=1) > 0)
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c), bond_order=float(be[r, c]))

    try:
        temp_mol = Chem.RWMol()
        for i in range(n):
            a = Chem.Atom(int(atoms_array[i][1]))
            a.SetFormalCharge(G.nodes[i]["formal_charge"])
            temp_mol.AddAtom(a)
        for r, c in zip(rows, cols):
            bo_val = be[r, c]
            if bo_val <= 1.25:
                bt = Chem.BondType.SINGLE
            elif bo_val <= 2.5:
                bt = Chem.BondType.DOUBLE
            else:
                bt = Chem.BondType.TRIPLE
            temp_mol.AddBond(int(r), int(c), bt)
        Chem.SanitizeMol(temp_mol)
        for bond in temp_mol.GetBonds():
            if bond.GetIsAromatic():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if G.has_edge(u, v):
                    G[u][v]["bond_order"] = 1.5
    except Exception:
        pass

    G = connect_components_with_head(G)

    if catalyst_flags is not None:
        for node in G.nodes():
            if G.nodes[node].get("atomic_number", 0) == 0:
                G.nodes[node]["is_catalyst"] = 0

    return G

def apply_elemental_step(G, edge1, edge2, swap_type=0):
    """Apply a two-bond break/form step to a molecular graph.

    Both edges have their bond order decremented by one (the edge is
    removed when the order reaches zero). New single bonds are then formed
    between the endpoints: for ``edge1 = (u, v)`` and ``edge2 = (x, y)``,
    ``swap_type=0`` forms (u, x) and (v, y), while ``swap_type=1`` forms
    (u, y) and (v, x); existing bonds have their order incremented instead.

    Args:
        G: NetworkX molecular graph (left unmodified).
        edge1: First bond (u, v) to break.
        edge2: Second bond (x, y) to break.
        swap_type: Endpoint pairing for the formed bonds (0 or 1).

    Returns:
        New NetworkX graph with the step applied.
    """
    new_G = G.copy()
    u, v = edge1
    x, y = edge2

    for e in [edge1, edge2]:
        if new_G.has_edge(e[0], e[1]):
            curr = new_G[e[0]][e[1]]["bond_order"]
            if curr > 1.0:
                new_G[e[0]][e[1]]["bond_order"] = curr - 1.0
            else:
                new_G.remove_edge(e[0], e[1])

    new_pairs = [(u, x), (v, y)] if swap_type == 0 else [(u, y), (v, x)]
    for n1, n2 in new_pairs:
        if new_G.has_edge(n1, n2):
            new_G[n1][n2]["bond_order"] += 1.0
        else:
            new_G.add_edge(n1, n2, bond_order=1.0)
    return new_G

def get_graph_features_nx(G, extended_features=False, canonical_rank=False):
    """Extract node features, edge tensors, and bond bookkeeping from a graph.

    Symmetry identifiers are computed on the subgraph of real atoms
    (excluding the virtual head node) by building a temporary RDKit
    molecule and ranking atoms with ``CanonicalRankAtoms(breakTies=False)``
    so symmetry-equivalent atoms share a rank; each atom's sym_id is its
    position within its rank group divided by the group size. With
    ``canonical_rank``, a tie-broken canonical rank normalized to [0, 1] is
    also computed.

    Node feature layout per row of ``x`` (float32):
    ``[atomic_number, formal_charge, num_explicit_hs, is_aromatic, sym_id]``,
    followed by ``canonical_rank`` if enabled, followed by the extended
    features ``[in_ring, ring_size (smallest of sizes 3-8, else 0), degree,
    hybridization, is_catalyst]`` if enabled.

    Edge types are looked up in ``BOND_ORDER_TO_TYPE`` from the edge
    ``bond_order``; each undirected edge contributes two directed entries.

    Args:
        G: NetworkX molecular graph including the virtual head node.
        extended_features: If True, append ring/degree/hybridization/catalyst
            features.
        canonical_rank: If True, append the normalized canonical rank
            feature.

    Returns:
        Tuple ``(x, edge_index, edge_attr, bonds, edge_map)`` where ``x`` is
        the (N, F) node feature tensor, ``edge_index`` is the (2, 2E)
        directed edge index, ``edge_attr`` is the (2E, 1) edge-type tensor,
        ``bonds`` is a list of (u_idx, v_idx) tuples (one per undirected
        edge, in remapped indices), and ``edge_map`` maps the sorted
        original node pair to the bond index.
    """
    node_indices = sorted(list(G.nodes()))
    node_map = {node_idx: i for i, node_idx in enumerate(node_indices)}

    real_nodes = [n for n in node_indices if G.nodes[n].get("atomic_number", 0) > 0]
    sym_ids = {n: 0.0 for n in node_indices}
    canon_ranks = {n: 0.0 for n in node_indices}
    ring_info = {n: 0 for n in node_indices}
    ring_size_info = {n: 0 for n in node_indices}
    degree_info = {n: 0 for n in node_indices}
    hybridization_info = {n: 0 for n in node_indices}

    if real_nodes:
        sub_G = G.subgraph(real_nodes)
        temp_mol = nx_to_mol(sub_G)
        if temp_mol:
            for atom in temp_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            ranks = list(Chem.CanonicalRankAtoms(temp_mol, breakTies=False))
            rank_groups = {}
            for rd_idx, r in enumerate(ranks):
                rank_groups.setdefault(r, []).append(rd_idx)

            sub_nodes_list = list(sub_G.nodes())
            for rd_idx, rank in enumerate(ranks):
                group = sorted(rank_groups[rank])
                pos = group.index(rd_idx)
                sym_ids[sub_nodes_list[rd_idx]] = pos / len(group)

            if canonical_rank:
                canon_ranks_raw = list(
                    Chem.CanonicalRankAtoms(temp_mol, breakTies=True)
                )
                n_real = len(canon_ranks_raw)
                for rd_idx, cr in enumerate(canon_ranks_raw):
                    canon_ranks[sub_nodes_list[rd_idx]] = cr / max(n_real - 1, 1)

            if extended_features:
                ri = temp_mol.GetRingInfo()
                for rd_idx_ext, atom in enumerate(temp_mol.GetAtoms()):
                    node = sub_nodes_list[rd_idx_ext]
                    ring_info[node] = 1 if atom.IsInRing() else 0
                    rs = 0
                    for size in range(3, 9):
                        if ri.IsAtomInRingOfSize(rd_idx_ext, size):
                            rs = size
                            break
                    ring_size_info[node] = rs
                    degree_info[node] = atom.GetDegree()
                    hybridization_info[node] = int(atom.GetHybridization())

    atom_features = []
    for n in node_indices:
        atom = G.nodes[n]
        feat = [
            atom.get("atomic_number", 0),
            atom.get("formal_charge", 0),
            atom.get("num_explicit_hs", 0),
            atom.get("is_aromatic", 0),
            sym_ids[n],
        ]
        if canonical_rank:
            feat.append(canon_ranks[n])
        if extended_features:
            feat.extend(
                [
                    ring_info[n],
                    ring_size_info[n],
                    degree_info[n],
                    hybridization_info[n],
                    atom.get("is_catalyst", 0),
                ]
            )
        atom_features.append(feat)

    x = torch.tensor(atom_features, dtype=torch.float32)

    bonds = []
    edge_map = {}
    src_list, dst_list, edge_attr_list = [], [], []

    edges = sorted(list(G.edges(data=True)))
    if len(edges) == 0:
        return (
            x,
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 1), dtype=torch.long),
            [],
            {},
        )

    for i, (u, v, data) in enumerate(edges):
        u_idx, v_idx = node_map[u], node_map[v]
        bonds.append((u_idx, v_idx))
        edge_map[tuple(sorted((u, v)))] = i

        order = data.get("bond_order", 1.0)
        b_type = BOND_ORDER_TO_TYPE.get(order, 1)

        src_list.extend([u_idx, v_idx])
        dst_list.extend([v_idx, u_idx])
        edge_attr_list.extend([[b_type], [b_type]])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)

    return x, edge_index, edge_attr, bonds, edge_map

def augment_with_lp_edges(
    x, edge_index, edge_attr, bonds_list, edge_map, be_matrix, node_map_inv
):
    """Add virtual LP nodes for all non-hydrogen atoms.

    For each non-H atom, adds one LP node connected by a bidirectional LP edge
    (bond_type=6, bond_order=7.0). This makes LP changes structurally identical
    to bond breaks/forms so the ranking model can predict both LP loss (atoms
    with existing LP >= 2) and LP gain (any non-H atom) uniformly. Atoms with
    no lone pairs receive a "potential LP" node so LP gain can be predicted;
    hydrogen (atomic number 1) and virtual nodes (atomic number 0) are
    skipped.

    LP node features are copied from the parent atom (notably sym_id at
    feature index 4) so LP nodes on different atoms remain distinguishable,
    with feature index 0 (atomic_number) and index 2 (H count) set to 0 to
    mark the node as virtual. Each LP bond is registered in the returned
    edge_map under the sentinel key ``tuple(sorted((atom_idx, -1 - k)))``
    for the k-th LP node, which cannot clash with real edge keys.

    Args:
        x: (N, F) node feature tensor
        edge_index: (2, E) edge index tensor
        edge_attr: (E, 1) edge attribute tensor
        bonds_list: list of (u_idx, v_idx) bond tuples (mapped indices)
        edge_map: dict mapping (node_a, node_b) -> bond index in bonds_list
        be_matrix: (n, n) BE matrix (original atom indices)
        node_map_inv: dict mapping mapped_index -> original_node_id (from get_graph_features_nx)

    Returns:
        (x, edge_index, edge_attr, bonds_list, edge_map, lp_node_map)
        lp_node_map: dict {original_atom_idx: mapped_lp_node_idx}
    """
    n_existing = x.size(0)
    lp_node_map = {}

    node_map_fwd = {v: k for k, v in node_map_inv.items()}

    new_x_rows = []
    new_src, new_dst, new_attr = [], [], []
    new_bonds = list(bonds_list)
    new_edge_map = dict(edge_map)

    lp_count = 0
    n_be = be_matrix.shape[0]
    for atom_idx in range(n_be):
        if atom_idx not in node_map_fwd:
            continue

        atom_mapped_idx_check = node_map_fwd[atom_idx]
        if float(x[atom_mapped_idx_check][0]) <= 1:
            continue

        lp_mapped_idx = n_existing + lp_count
        atom_mapped_idx = node_map_fwd[atom_idx]

        parent_feat = x[atom_mapped_idx].tolist()
        lp_feat = list(parent_feat)
        lp_feat[0] = 0.0
        lp_feat[2] = 0.0
        new_x_rows.append(lp_feat)

        lp_bond_type = BOND_ORDER_TO_TYPE[7.0]
        new_src.extend([atom_mapped_idx, lp_mapped_idx])
        new_dst.extend([lp_mapped_idx, atom_mapped_idx])
        new_attr.extend([[lp_bond_type], [lp_bond_type]])

        bond_key = tuple(
            sorted((atom_idx, -1 - lp_count))
        )
        bond_idx = len(new_bonds)
        new_bonds.append((atom_mapped_idx, lp_mapped_idx))
        new_edge_map[bond_key] = bond_idx

        lp_node_map[atom_idx] = lp_mapped_idx
        lp_count += 1

    if lp_count == 0:
        return x, edge_index, edge_attr, bonds_list, edge_map, lp_node_map

    lp_x = torch.tensor(new_x_rows, dtype=x.dtype)
    x = torch.cat([x, lp_x], dim=0)

    new_edge_src = torch.tensor(new_src, dtype=torch.long)
    new_edge_dst = torch.tensor(new_dst, dtype=torch.long)
    new_edge_idx = torch.stack([new_edge_src, new_edge_dst], dim=0)
    edge_index = torch.cat([edge_index, new_edge_idx], dim=1)

    new_edge_attr = torch.tensor(new_attr, dtype=edge_attr.dtype)
    edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)

    return x, edge_index, edge_attr, new_bonds, new_edge_map, lp_node_map
