"""Parity test: EBM loss fast (vectorized) path == loop reference."""

import torch
import pytest

from bond_match.losses.ranking_loss import EBMPairRankingLoss
from bond_match.model import BondGAT


def _build_model(num_bond_types: int = 7, with_swap: bool = True):
    return BondGAT(
        d_model=24,
        heads=4,
        num_layers=1,
        transformer_layers=1,
        atom_max_num=10,
        num_bond_types=num_bond_types,
        dropout=0.0,
        use_pair_bond_type=with_swap,
        use_attention_pair_scorer=True,
        use_symmetric_pair_scorer=True,
        pair_score_backend="exact_attention_chunked",
        pair_score_chunk_size=512,
        return_type_prediction=with_swap,
    )


def _make_outputs(model, n_bonds_per_mol, n_atoms_per_mol, n_react_per_mol, seed=0):
    """Construct a synthetic out_dict matching what BondGAT.forward would emit
    for the loss, with controlled reactive bond layouts."""
    g = torch.Generator().manual_seed(seed)
    n_mols = len(n_bonds_per_mol)
    total_bonds = sum(n_bonds_per_mol)
    total_atoms = sum(n_atoms_per_mol)
    D = model.d_model

    z = torch.randn(total_bonds, D, generator=g) * 0.5
    binary_logits = torch.randn(total_bonds, 1, generator=g) * 0.3
    stop_logits = torch.randn(n_mols, 1, generator=g) * 0.3
    h_nodes = torch.randn(total_atoms, D, generator=g) * 0.2

    bond_batch = []
    targets = torch.zeros(total_bonds, dtype=torch.long)
    bond_indices_out = torch.zeros(total_bonds, 2, dtype=torch.long)

    bond_offset = 0
    atom_offset = 0
    for m, (nb, na, nr) in enumerate(
        zip(n_bonds_per_mol, n_atoms_per_mol, n_react_per_mol)
    ):
        bond_batch.extend([m] * nb)
        # Pick atom indices per bond uniformly.
        for b in range(nb):
            a, c = torch.randint(0, na, (2,), generator=g).tolist()
            if a == c:
                c = (c + 1) % na
            bond_indices_out[bond_offset + b, 0] = atom_offset + a
            bond_indices_out[bond_offset + b, 1] = atom_offset + c
        # Mark ``nr`` bonds as reactive using nonzero class identifiers.
        if nr > 0 and nb >= nr:
            picks = torch.randperm(nb, generator=g)[:nr]
            for p in picks.tolist():
                targets[bond_offset + p] = int(
                    torch.randint(1, 3, (1,), generator=g).item()
                )
        bond_offset += nb
        atom_offset += na

    bond_batch_vec = torch.tensor(bond_batch, dtype=torch.long)

    out_dict = {
        "embeddings": z,
        "logits_binary": binary_logits,
        "logits_stop": stop_logits,
        "node_embeddings": h_nodes,
        "bond_indices_out": bond_indices_out,
        "model_ref": model,
    }
    is_over_targets = torch.zeros(n_mols, dtype=torch.bool)
    # Mark mols with no reactive bonds as stop targets.
    for m, nr in enumerate(n_react_per_mol):
        if nr == 0:
            is_over_targets[m] = True
    return out_dict, targets, bond_batch_vec, is_over_targets


@pytest.mark.parametrize(
    "n_bonds_per_mol,n_react_per_mol",
    [
        ([5, 4, 6], [2, 0, 2]),
        ([3, 3, 3, 3], [2, 2, 0, 2]),
        ([2, 8, 5, 4, 6], [0, 2, 2, 0, 2]),
        ([7], [2]),
        ([2, 2], [0, 0]),  # all stop targets
    ],
)
@pytest.mark.parametrize("pair_scoring_mode", ["pair_only", "fused"])
@pytest.mark.parametrize("stop_rank_aux_weight", [0.0, 0.1])
@pytest.mark.parametrize("with_swap", [True, False])
def test_fast_path_matches_loop(
    n_bonds_per_mol,
    n_react_per_mol,
    pair_scoring_mode,
    stop_rank_aux_weight,
    with_swap,
):
    torch.manual_seed(7)
    model = _build_model(with_swap=with_swap)
    n_atoms_per_mol = [max(b + 2, 4) for b in n_bonds_per_mol]
    out_dict, targets, bond_batch_vec, is_over_targets = _make_outputs(
        model, n_bonds_per_mol, n_atoms_per_mol, n_react_per_mol
    )

    loss = EBMPairRankingLoss(
        bond_type_weight=0.5 if with_swap else 0.0,
        binary_weight=1.0,
        lambda_anchor=0.05,
        hard_negative_k=0,
        virtual_pos_weight=1.0,
        negative_reaction_weight=0.0,
        pair_scoring_mode=pair_scoring_mode,
        stop_rank_aux_weight=stop_rank_aux_weight,
    )

    fast_val = loss._forward_fast(out_dict, targets, bond_batch_vec, is_over_targets)
    loop_val = loss._forward_loop(
        out_dict, targets, bond_batch_vec, is_over_targets, None
    )
    assert torch.isfinite(fast_val) and torch.isfinite(loop_val)
    assert torch.allclose(fast_val, loop_val, atol=1e-5, rtol=1e-4), (
        f"fast={fast_val.item()} loop={loop_val.item()} diff={(fast_val-loop_val).item()}"
    )


def test_fast_path_dispatched_by_forward():
    torch.manual_seed(0)
    model = _build_model(with_swap=False)
    out_dict, targets, bond_batch_vec, is_over_targets = _make_outputs(
        model, [4, 5, 6], [6, 7, 8], [2, 0, 2]
    )
    loss = EBMPairRankingLoss(
        bond_type_weight=0.0,
        lambda_anchor=0.05,
        hard_negative_k=0,
        negative_reaction_weight=0.0,
        pair_scoring_mode="fused",
        stop_rank_aux_weight=0.1,
    )
    assert loss._can_use_fast_path(out_dict, None)
    val = loss(out_dict, targets, bond_batch_vec, is_over_targets, None)
    loop_val = loss._forward_loop(
        out_dict, targets, bond_batch_vec, is_over_targets, None
    )
    assert torch.allclose(val, loop_val, atol=1e-5, rtol=1e-4)


def test_fast_path_grad_matches_loop():
    """Ensure backward through fast path produces the same gradients."""
    torch.manual_seed(3)
    model = _build_model(with_swap=False)
    out_dict, targets, bond_batch_vec, is_over_targets = _make_outputs(
        model, [4, 6, 5], [5, 7, 6], [2, 0, 2]
    )
    out_dict["embeddings"] = out_dict["embeddings"].requires_grad_(True)

    loss = EBMPairRankingLoss(
        bond_type_weight=0.0,
        lambda_anchor=0.05,
        hard_negative_k=0,
        negative_reaction_weight=0.0,
        pair_scoring_mode="fused",
        stop_rank_aux_weight=0.1,
    )

    z_loop = out_dict["embeddings"].detach().clone().requires_grad_(True)
    out_loop = dict(out_dict, embeddings=z_loop)
    l_loop = loss._forward_loop(
        out_loop, targets, bond_batch_vec, is_over_targets, None
    )
    l_loop.backward()
    g_loop = z_loop.grad.detach().clone()

    z_fast = out_dict["embeddings"].detach().clone().requires_grad_(True)
    out_fast = dict(out_dict, embeddings=z_fast)
    l_fast = loss._forward_fast(
        out_fast, targets, bond_batch_vec, is_over_targets
    )
    l_fast.backward()
    g_fast = z_fast.grad.detach().clone()

    assert torch.allclose(l_fast, l_loop, atol=1e-5, rtol=1e-4)
    assert torch.allclose(g_fast, g_loop, atol=1e-4, rtol=1e-3), (
        f"max grad diff = {(g_fast - g_loop).abs().max().item()}"
    )
