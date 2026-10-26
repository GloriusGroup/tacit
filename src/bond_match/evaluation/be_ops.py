"""Bond-electron (BE) matrix operations and action scoring for evaluation.

Provides bond-swap application to BE matrices (with and without lone-pair
virtual nodes), BE-matrix-to-SMILES conversion, action ranking over bond
pairs plus the stop action, and swap-type decoding from the pair-level
bond type classifier.
"""

import numpy as np
import torch
from bond_match.blockspace import (
    decompose_molecules,
    matrices_to_smiles,
)
from bond_match.config import PairSelectionMode

from bond_match.evaluation.results import (
    PairResult,
)

def apply_predicted_step_be(be, bond1, bond2, swap_type=0):
    """Apply a predicted bond swap step to a BE matrix.

    Breaks both input bonds, forms the two new bonds selected by
    ``swap_type``, and compensates each atom's net bond-order change on the
    diagonal (lone-pair electrons).

    Args:
        be: Square BE matrix.
        bond1: Atom index pair (u, v) of the first bond.
        bond2: Atom index pair (x, y) of the second bond.
        swap_type: 0 forms (u, x) and (v, y); any other value forms
            (u, y) and (v, x).

    Returns:
        A new BE matrix with the swap applied; the input is not modified.
    """
    result = be.copy()
    n = len(be)
    u, v = bond1
    x, y = bond2

    atom_bond_change = np.zeros(n, dtype=np.float32)

    for a, b in [(u, v), (x, y)]:
        result[a, b] -= 1.0
        result[b, a] -= 1.0
        atom_bond_change[a] -= 1.0
        atom_bond_change[b] -= 1.0

    new_pairs = [(u, x), (v, y)] if swap_type == 0 else [(u, y), (v, x)]
    for a, b in new_pairs:
        result[a, b] += 1.0
        result[b, a] += 1.0
        atom_bond_change[a] += 1.0
        atom_bond_change[b] += 1.0

    for k in range(n):
        if abs(atom_bond_change[k]) > 1e-6:
            result[k, k] -= atom_bond_change[k]

    return result

def apply_step_with_lp(be, bond1, bond2, swap_type, reverse_lp_map):
    """Apply a predicted bond swap step to a BE matrix with lone-pair virtual nodes.

    Bond breaks or formations that involve a lone-pair virtual node are
    applied as a change of 2 on the real atom's diagonal entry instead of an
    off-diagonal bond change.

    Args:
        be: Square BE matrix over real atoms.
        bond1: Node index pair (u, v) of the first bond.
        bond2: Node index pair (x, y) of the second bond.
        swap_type: 0 forms (u, x) and (v, y); any other value forms
            (u, y) and (v, x).
        reverse_lp_map: Mapping from lone-pair virtual node index to the
            owning real atom index.

    Returns:
        A new BE matrix with the swap applied; the input is not modified.
    """
    result = be.copy()
    n = len(be)
    u, v = bond1
    x, y = bond2

    def is_lp(node_idx):
        """Return True if the node index is a lone-pair virtual node."""
        return node_idx in reverse_lp_map

    def do_break(a, b):
        """Remove one bond order between a and b, or two diagonal electrons if a lone pair is involved."""
        if is_lp(a) or is_lp(b):
            real_atom = b if is_lp(a) else a
            if real_atom < n:
                result[real_atom, real_atom] -= 2
        else:
            if a < n and b < n:
                result[a, b] -= 1.0
                result[b, a] -= 1.0

    def do_form(a, b):
        """Add one bond order between a and b, or two diagonal electrons if a lone pair is involved."""
        if is_lp(a) or is_lp(b):
            real_atom = b if is_lp(a) else a
            if real_atom < n:
                result[real_atom, real_atom] += 2
        else:
            if a < n and b < n:
                result[a, b] += 1.0
                result[b, a] += 1.0

    do_break(u, v)
    do_break(x, y)

    new_pairs = [(u, x), (v, y)] if swap_type == 0 else [(u, y), (v, x)]
    for a, b in new_pairs:
        do_form(a, b)

    return result

def be_result_to_smiles(be, atoms):
    """Convert a BE matrix and atom identities to a dot-separated SMILES string.

    Args:
        be: Square BE matrix.
        atoms: Atom identity array aligned with the BE matrix.

    Returns:
        SMILES of all connected components joined by ``.``.
    """
    mols = decompose_molecules(be, atoms)
    return ".".join(matrices_to_smiles(m, a) for m, a in mols)

def score_actions(
    model,
    embeddings,
    binary_logits,
    stop_logit,
    device,
    top_k=1,
    selection_mode=PairSelectionMode.ARGMAX,
    temperature=1.0,
    softmax_topk=0,
    pair_scoring_mode="pair_only",
):
    """Rank actions for the current state and select the top ``top_k``.

    The action set is all unordered bond pairs (upper triangle of the
    pairwise score matrix) plus a single ``is_over`` stop action. Selection
    is either deterministic (argmax with stable tie-breaking at a tolerance
    of 1e-5 on the scores) or sampled from a temperature-scaled softmax,
    optionally restricted to the top ``softmax_topk`` actions.

    Args:
        model: BondGAT model providing ``compute_pair_scores_upper_triangle``.
        embeddings: Bond embeddings for the molecule.
        binary_logits: Per-bond logits passed to the pair scorer.
        stop_logit: Logit of the stop action.
        device: Torch device (unused in the computation itself).
        top_k: Number of actions to return.
        selection_mode: ``PairSelectionMode.ARGMAX`` for deterministic
            selection, otherwise multinomial sampling.
        temperature: Softmax temperature; values <= 0 force argmax selection.
        softmax_topk: If > 0, restrict the softmax to this many top actions.
        pair_scoring_mode: Mode forwarded to the pair scorer.

    Returns:
        List of ``PairResult`` objects, one per selected action, each carrying
        its score, action probability and the probability of the stop action;
        stop actions have ``is_over=True``.
    """
    pair_scores, triu_idx = model.compute_pair_scores_upper_triangle(
        embeddings,
        binary_logits=binary_logits,
        mode=pair_scoring_mode,
    )

    stop_score = stop_logit.reshape(1)
    action_scores = torch.cat([pair_scores, stop_score], dim=0)
    use_argmax = selection_mode == PairSelectionMode.ARGMAX or temperature <= 0
    score_tie_tol = 1e-5

    if softmax_topk > 0 and action_scores.size(0) > softmax_topk:
        topk_scores, topk_idx = torch.topk(action_scores, softmax_topk)
        topk_probs = torch.softmax(topk_scores / max(temperature, 1e-8), dim=0)
        action_probs = torch.zeros_like(action_scores)
        action_probs[topk_idx] = topk_probs
    else:
        action_probs = torch.softmax(action_scores / max(temperature, 1e-8), dim=0)

    def _select_indices(k: int) -> torch.Tensor:
        """Select k action indices by sampling or by quantized stable argsort."""
        if not use_argmax:
            return torch.multinomial(action_probs, num_samples=k, replacement=False)
        quantized_scores = torch.round(action_scores / score_tie_tol)
        indices = torch.sort(quantized_scores, descending=True, stable=True).indices
        return indices[:k]

    k = min(top_k, action_scores.size(0))
    selected = _select_indices(k)
    stop_idx = pair_scores.size(0)
    stop_prob = action_probs[stop_idx].item()
    results = []
    for idx in selected.tolist():
        if idx == stop_idx:
            results.append(
                PairResult(
                    is_over=True,
                    score=action_scores[idx].item(),
                    action_prob=action_probs[idx].item(),
                    stop_prob=stop_prob,
                )
            )
            continue
        results.append(
            PairResult(
                is_over=False,
                local_b1=triu_idx[0][idx].item(),
                local_b2=triu_idx[1][idx].item(),
                score=action_scores[idx].item(),
                action_prob=action_probs[idx].item(),
                stop_prob=stop_prob,
            )
        )
    return results

def decode_swap_type(
    model,
    local_b1,
    local_b2,
    node_embeddings=None,
    bond_atom_indices=None,
    bond_embeddings=None,
):
    """Decode the swap type from the model's pair-level bond type classifier.

    Supports MLPSwapTypeHead (bond embeddings only), AtomSwapTypeHead and
    NodeAttentionSwapHead (node embeddings of the four bond endpoints), and
    BondSplitSwapHead (bond embeddings plus the four endpoint node
    embeddings). NodeAttentionSwapHead emits two classes (swap type 0,
    swap type 1) used directly; all other heads emit three classes
    (stable, swap type 0, swap type 1) and the stable class is excluded
    before the softmax.

    Args:
        model: BondGAT model with a ``pair_bond_type_classifier``.
        local_b1: Local bond index of the first bond.
        local_b2: Local bond index of the second bond.
        node_embeddings: (num_nodes, d_model) node embeddings from the GAT.
        bond_atom_indices: (num_bonds, 2) atom index pairs for each bond.
        bond_embeddings: (num_bonds, d_model) post-transformer bond embeddings.

    Returns:
        Tuple ``(pred_class, probs_str)`` where ``pred_class`` is the argmax
        class index over the retained classes and ``probs_str`` is a
        formatted probability string such as ``'(0: 20%, 1: 80%)'``.
    """
    from bond_match.model.heads import (
        AtomSwapTypeHead,
        BondSplitSwapHead,
        MLPSwapTypeHead,
        NodeAttentionSwapHead,
    )

    classifier = model.pair_bond_type_classifier
    with torch.no_grad():
        if isinstance(classifier, MLPSwapTypeHead):
            assert bond_embeddings is not None, (
                "MLPSwapTypeHead requires bond_embeddings."
            )
            logits = classifier(bond_embeddings[local_b1], bond_embeddings[local_b2])
        else:
            assert node_embeddings is not None, (
                "decode_swap_type: node_embeddings is None."
            )
            assert bond_atom_indices is not None, (
                "decode_swap_type: bond_atom_indices is None."
            )
            h_u1 = node_embeddings[bond_atom_indices[local_b1, 0]]
            h_u2 = node_embeddings[bond_atom_indices[local_b1, 1]]
            h_v1 = node_embeddings[bond_atom_indices[local_b2, 0]]
            h_v2 = node_embeddings[bond_atom_indices[local_b2, 1]]

            if isinstance(classifier, BondSplitSwapHead):
                assert bond_embeddings is not None, (
                    "BondSplitSwapHead requires bond_embeddings."
                )
                logits = classifier(
                    bond_embeddings[local_b1],
                    bond_embeddings[local_b2],
                    h_u1,
                    h_u2,
                    h_v1,
                    h_v2,
                )
            elif isinstance(classifier, (AtomSwapTypeHead, NodeAttentionSwapHead)):
                logits = classifier(h_u1, h_u2, h_v1, h_v2)
            else:
                raise TypeError(f"Unknown classifier: {type(classifier)}")

        if logits.dim() == 2:
            logits = logits.squeeze(0)

        if isinstance(classifier, NodeAttentionSwapHead):
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits[1:], dim=-1)
        pred_class = torch.argmax(probs).item()
        probs_str = ", ".join(
            f"{i}: {probs[i].item() * 100:.0f}%" for i in range(probs.shape[0])
        )
    return pred_class, f"({probs_str})"

