"""Batch collation functions for reaction graph samples."""

import torch
import numpy as np
import torch


def collate_fn(
    batch,
    enable_matrix_output=False,
    enable_history=False,
):
    """Collate per-sample reaction graphs into a single batched graph.

    Node features are concatenated and edge/bond indices are shifted by a
    running node offset so that all samples share one index space. Per-sample
    scalars (``is_over_target``, ``is_negative_reaction``, ``step_index``)
    are stacked into tensors.

    When ``enable_matrix_output`` is set, bond labels are returned as a
    padded ``[batch, max_bonds, max_bonds]`` matrix: padded entries are -1,
    non-interacting valid entries are 0, and the reacting bond pair is
    written symmetrically. Three label encodings are supported per sample:
    a 1D vector aligned with the bond list (two non-zero entries mark the
    reacting pair; a single non-zero entry is written on the diagonal), an
    explicit ``[u, v, label]`` triplet, or an already dense ``N x N`` matrix.
    Otherwise the per-bond label vectors are concatenated.

    When ``enable_history`` is set, the history items of all samples are
    collated recursively into a separate batch with zeroed labels, together
    with a tensor mapping each history item to its source sample.

    Args:
        batch: List of sample dicts with keys ``x``, ``edge_index``,
            ``edge_attr``, ``bonds``, ``y``, and optional metadata keys.
        enable_matrix_output: If True, return labels as a padded pairwise
            bond matrix instead of a concatenated vector.
        enable_history: If True, collate the ``history`` entries of each
            sample into an auxiliary batch.

    Returns:
        Tuple ``(x_batch, y_batch, batch_idx, edge_index_batch,
        edge_attr_batch, bond_indices_batch, is_over_target_batch,
        is_negative_reaction_batch, history_batch, history_map_tensor,
        step_index_batch)``. ``history_batch`` and ``history_map_tensor``
        are None when no history items are present.
    """
    all_x = []
    all_edge_indices = []
    all_edge_attrs = []
    all_bond_indices = []
    all_y = []
    batch_map = []
    all_is_over_target = []
    all_is_negative_reaction = []
    all_step_index = []

    all_history_items = []
    history_sample_map = []

    if enable_history:
        for i, item in enumerate(batch):
            history = item.get("history", [])
            for hist_item in history:
                all_history_items.append(hist_item)
                history_sample_map.append(i)

    node_offset = 0

    y_matrix_batch = None
    if enable_matrix_output:
        max_bonds = 0
        for item in batch:
            nb = len(item["bonds"])
            if nb > max_bonds:
                max_bonds = nb

        y_matrix_batch = torch.full(
            (len(batch), max_bonds, max_bonds), -1, dtype=torch.long
        )

    for i, item in enumerate(batch):
        x = item["x"]
        num_nodes = x.size(0)
        all_x.append(x)

        all_is_over_target.append(
            item.get("is_over_target", item.get("output_is_virtual", 0))
        )
        all_is_negative_reaction.append(
            bool(
                item.get(
                    "is_negative_reaction",
                    item.get("type") == "negative_reaction",
                )
            )
        )
        step_idx_val = item.get("step_index", 0)
        all_step_index.append(step_idx_val)

        batch_map.append(torch.full((num_nodes,), i, dtype=torch.long))

        edge_index = item["edge_index"]
        if edge_index.numel() > 0:
            all_edge_indices.append(edge_index + node_offset)
            all_edge_attrs.append(item["edge_attr"])

        bonds = torch.as_tensor(item["bonds"], dtype=torch.long)
        num_b = bonds.size(0)

        if num_b > 0:
            all_bond_indices.append(bonds + node_offset)

            y_data = torch.as_tensor(item["y"], dtype=torch.long)

            if enable_matrix_output:
                y_matrix_batch[i, :num_b, :num_b] = 0

                if y_data.dim() == 1 and y_data.size(0) == num_b:
                    nonzero_indices = torch.nonzero(y_data).squeeze(1)

                    if nonzero_indices.numel() == 2:
                        u, v = nonzero_indices[0], nonzero_indices[1]
                        label = y_data[u]

                        y_matrix_batch[i, u, v] = label
                        y_matrix_batch[i, v, u] = label

                    elif nonzero_indices.numel() == 1:
                        u = nonzero_indices[0]
                        y_matrix_batch[i, u, u] = y_data[u]

                elif y_data.numel() == 3 and y_data.dim() == 1:
                    u, v, l = y_data[0], y_data[1], y_data[2]
                    if u < num_b and v < num_b:
                        y_matrix_batch[i, u, v] = l
                        y_matrix_batch[i, v, u] = l

                elif y_data.numel() == num_b * num_b:
                    y_matrix_batch[i, :num_b, :num_b] = y_data.view(num_b, num_b)

            else:
                all_y.append(y_data)

        node_offset += num_nodes

    x_batch = torch.cat(all_x, dim=0)
    batch_idx = torch.cat(batch_map, dim=0)

    is_over_target_batch = torch.tensor(all_is_over_target, dtype=torch.long)
    is_negative_reaction_batch = torch.tensor(
        all_is_negative_reaction, dtype=torch.bool
    )
    step_index_batch = torch.tensor(all_step_index, dtype=torch.long)

    if len(all_edge_indices) > 0:
        edge_index_batch = torch.cat(all_edge_indices, dim=1)
        edge_attr_batch = torch.cat(all_edge_attrs, dim=0)
    else:
        edge_index_batch = torch.empty((2, 0), dtype=torch.long)
        edge_attr_batch = torch.empty((0, 1), dtype=torch.long)

    if len(all_bond_indices) > 0:
        bond_indices_batch = torch.cat(all_bond_indices, dim=0)
        if enable_matrix_output:
            y_batch = y_matrix_batch
        else:
            y_batch = torch.cat(all_y, dim=0)
    else:
        bond_indices_batch = torch.empty((0, 2), dtype=torch.long)
        if enable_matrix_output:
            y_batch = torch.empty((len(batch), 0, 0), dtype=torch.long)
        else:
            y_batch = torch.empty((0,), dtype=torch.long)

    history_batch = None
    history_map_tensor = None
    if all_history_items:
        hist_samples = []
        for h in all_history_items:
            hs = {
                "x": h["x"],
                "edge_index": h["edge_index"],
                "edge_attr": h["edge_attr"],
                "bonds": h["bonds"],
                "y": np.zeros(len(h["bonds"])),
                "is_over_target": 0,
                "step_index": h.get("step_index", 0),
                "reactant_smi": h.get("reactant_smi", ""),
            }
            hist_samples.append(hs)
        history_batch = collate_fn(
            hist_samples,
            enable_matrix_output=False,
            enable_history=False,
        )
        history_map_tensor = torch.tensor(history_sample_map, dtype=torch.long)

    return (
        x_batch,
        y_batch,
        batch_idx,
        edge_index_batch,
        edge_attr_batch,
        bond_indices_batch,
        is_over_target_batch,
        is_negative_reaction_batch,
        history_batch,
        history_map_tensor,
        step_index_batch,
    )

def trajectory_collate_fn(
    batch,
    enable_matrix_output=False,
):
    """Collate trajectory samples by flattening their states into one batch.

    Each batch item is either a trajectory dict with a ``states`` list or a
    single state, which is treated as a one-state trajectory. All states are
    collated with :func:`collate_fn` and a trajectory-id tensor mapping each
    state to its source trajectory is appended.

    Args:
        batch: List of trajectory dicts or single state dicts.
        enable_matrix_output: Forwarded to :func:`collate_fn`.

    Returns:
        The :func:`collate_fn` output tuple with an additional trailing
        tensor of per-state trajectory indices.
    """
    flat_states = []
    trajectory_ids = []

    for traj_idx, item in enumerate(batch):
        states = item.get("states") if isinstance(item, dict) else None
        if states is None:
            states = [item]
        for state in states:
            flat_states.append(state)
            trajectory_ids.append(traj_idx)

    collated = collate_fn(
        flat_states,
        enable_matrix_output=enable_matrix_output,
        enable_history=False,
    )
    return collated + (torch.tensor(trajectory_ids, dtype=torch.long),)
