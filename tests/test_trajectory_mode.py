import torch

from bond_match.dataset import trajectory_collate_fn
from bond_match.model import BondGAT
from bond_match.config import RankingConfig
from bond_match.train_ranking import BatchInputs, build_loss


def _pack_trajectory_tokens_reference(
    bond_tokens: torch.Tensor,
    bond_batch_vec: torch.Tensor,
    trajectory_ids: torch.Tensor,
    step_index: torch.Tensor,
):
    """Loop-based reference for _pack_trajectory_tokens."""
    traj_values = torch.unique(trajectory_ids, sorted=True)
    packed_orders = []
    packed_steps = []
    max_len = 0
    for traj_id in traj_values:
        state_ids = torch.where(trajectory_ids == traj_id)[0]
        state_ids = state_ids[torch.argsort(step_index[state_ids], stable=True)]
        order_parts = []
        step_parts = []
        for state_id in state_ids:
            token_idx = torch.where(bond_batch_vec == state_id)[0]
            if token_idx.numel() == 0:
                continue
            order_parts.append(token_idx)
            step_parts.append(step_index[state_id].expand(token_idx.numel()))
        if order_parts:
            order = torch.cat(order_parts, dim=0)
            steps = torch.cat(step_parts, dim=0)
        else:
            order = torch.empty(0, dtype=torch.long)
            steps = torch.empty(0, dtype=torch.long)
        packed_orders.append(order)
        packed_steps.append(steps)
        max_len = max(max_len, int(order.numel()))

    B = len(packed_orders)
    D = bond_tokens.size(-1)
    if max_len == 0:
        return (
            bond_tokens.new_zeros((B, 0, D)),
            torch.zeros((B, 0), dtype=torch.bool),
            torch.zeros((B, 0), dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.zeros((B, 0), dtype=torch.long),
        )

    dense = bond_tokens.new_zeros((B, max_len, D))
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    token_steps = torch.zeros((B, max_len), dtype=torch.long)
    group_ids = torch.zeros((B, max_len), dtype=torch.long)
    flat_order = []
    for row, (order, steps) in enumerate(zip(packed_orders, packed_steps)):
        n = order.numel()
        if n == 0:
            continue
        dense[row, :n] = bond_tokens[order]
        mask[row, :n] = True
        token_steps[row, :n] = steps
        group_ids[row, :n] = bond_batch_vec[order]
        flat_order.append(order)
    flat_order_t = (
        torch.cat(flat_order, dim=0) if flat_order else torch.empty(0, dtype=torch.long)
    )
    return dense, mask, token_steps, flat_order_t, group_ids


def _state(step_index: int, atomic_num: float = 6.0, is_over: bool = False):
    return {
        "x": torch.tensor(
            [
                [atomic_num, 0.0, 0.0, 0.0, 0.0],
                [atomic_num, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_attr": torch.tensor([[1], [1]], dtype=torch.long),
        "bonds": [(0, 1)],
        "y": torch.zeros(1, dtype=torch.long),
        "type": "product" if is_over else f"mech0_step_{step_index}",
        "is_over_target": is_over,
        "is_negative_reaction": False,
        "has_catalyst": False,
        "step_index": step_index,
    }


def test_trajectory_collate_flattens_states_with_ids():
    batch = trajectory_collate_fn(
        [
            {"states": [_state(0), _state(1, is_over=True)]},
            {"states": [_state(0, atomic_num=8.0)]},
        ]
    )

    trajectory_ids = batch[-1]
    step_index = batch[10]

    assert trajectory_ids.tolist() == [0, 0, 1]
    assert step_index.tolist() == [0, 1, 0]




def test_trajectory_forward_loss_backward_smoke():
    torch.manual_seed(1)
    model = BondGAT(
        d_model=32,
        heads=4,
        num_layers=1,
        transformer_layers=1,
        atom_max_num=100,
        num_bond_types=7,
        dropout=0.0,
        return_type_prediction=True,
        enable_history_context=False,
        use_pair_bond_type=True,
        use_attention_pair_scorer=True,
        use_symmetric_pair_scorer=True,
        pair_score_backend="exact_attention_chunked",
        trajectory_attention=True,
    )
    cfg = RankingConfig(
        model={"use_pair_bond_type": True},
        loss={"ranking_loss_type": "ebm", "pair_scoring_mode": "fused"},
    )
    loss_fn = build_loss(cfg)

    s0 = _state(0)
    s0["bonds"] = [(0, 1), (0, 2)]
    s0["x"] = torch.tensor(
        [
            [6.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    s0["edge_index"] = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    s0["edge_attr"] = torch.tensor([[1], [1], [1], [1]], dtype=torch.long)
    s0["y"] = torch.tensor([1, 1], dtype=torch.long)

    batch = trajectory_collate_fn([{"states": [s0, _state(1, is_over=True)]}])
    inputs = BatchInputs.from_batch(batch, torch.device("cpu"))
    out = model(
        inputs.x,
        inputs.batch_idx,
        inputs.edge_index,
        inputs.edge_attr,
        inputs.bond_indices,
        step_index=inputs.step_index,
        trajectory_ids=inputs.trajectory_ids,
    )
    out["model_ref"] = model
    loss = loss_fn(
        out,
        inputs.y,
        inputs.bond_batch_vec,
        inputs.is_over_target,
        inputs.is_negative_reaction,
    )
    assert torch.isfinite(loss)
    loss.backward()


def _make_pack_inputs(
    n_trajs: int,
    states_per_traj: list[int],
    bonds_per_state: list[int],
    d_model: int = 6,
    seed: int = 0,
    interleave: bool = False,
    extra_traj_ids: list[int] | None = None,
):
    """Build (bond_tokens, bond_batch_vec, trajectory_ids, step_index)."""
    g = torch.Generator().manual_seed(seed)
    trajectory_ids = []
    step_index = []
    for traj in range(n_trajs):
        n_states = states_per_traj[traj]
        for s in range(n_states):
            trajectory_ids.append(traj)
            step_index.append(s)
    # Optionally append extra trajectory_ids that appear in states but
    # contribute no bonds (exercises the empty-row path).
    if extra_traj_ids:
        for tid in extra_traj_ids:
            trajectory_ids.append(tid)
            step_index.append(0)

    trajectory_ids_t = torch.tensor(trajectory_ids, dtype=torch.long)
    step_index_t = torch.tensor(step_index, dtype=torch.long)

    bond_batch = []
    for state_id, n_bonds in enumerate(bonds_per_state):
        bond_batch.extend([state_id] * n_bonds)
    bond_batch_t = torch.tensor(bond_batch, dtype=torch.long)
    if interleave and bond_batch_t.numel() > 1:
        perm = torch.randperm(bond_batch_t.numel(), generator=g)
        bond_batch_t = bond_batch_t[perm]

    n_bonds_total = bond_batch_t.numel()
    bond_tokens = torch.randn(n_bonds_total, d_model, generator=g)
    return bond_tokens, bond_batch_t, trajectory_ids_t, step_index_t


def _assert_pack_equivalence(*args):
    ref = _pack_trajectory_tokens_reference(*args)
    out = BondGAT._pack_trajectory_tokens(None, *args)

    # dense/mask/token_steps/group_ids must match. flat_order may differ in
    # tie order but must reconstruct the same dense after scatter-back.
    assert torch.equal(ref[1], out[1])
    assert torch.equal(ref[2], out[2])
    assert torch.allclose(ref[0], out[0])
    assert torch.equal(ref[4], out[4])

    # Round-trip: scatter packed_context back via flat_order and re-pack.
    bond_tokens = args[0]
    bond_batch_vec = args[1]
    if bond_tokens.numel() == 0:
        return
    packed_context = ref[0][ref[1]]
    bond_context_ref = torch.empty_like(bond_tokens)
    bond_context_ref[ref[3]] = packed_context

    packed_context_out = out[0][out[1]]
    bond_context_out = torch.empty_like(bond_tokens)
    bond_context_out[out[3]] = packed_context_out
    assert torch.allclose(bond_context_ref, bond_context_out)


def test_pack_trajectory_tokens_matches_reference():
    cases = [
        # Simple: 2 trajs, varying states and bond counts.
        dict(n_trajs=2, states_per_traj=[2, 1], bonds_per_state=[3, 2, 4]),
        dict(n_trajs=1, states_per_traj=[3], bonds_per_state=[1, 5, 2]),
        # State with zero bonds.
        dict(n_trajs=2, states_per_traj=[2, 2], bonds_per_state=[3, 0, 2, 1]),
        dict(n_trajs=2, states_per_traj=[1, 1], bonds_per_state=[0, 0]),
        # Trajectory with no bonds at all (extra_traj_ids).
        dict(
            n_trajs=2,
            states_per_traj=[1, 1],
            bonds_per_state=[2, 3],
            extra_traj_ids=[5],
        ),
    ]
    for case in cases:
        for seed in range(3):
            args = _make_pack_inputs(**case, seed=seed)
            _assert_pack_equivalence(*args)


def test_pack_trajectory_tokens_handles_interleaved_batch_vec():
    args = _make_pack_inputs(
        n_trajs=2,
        states_per_traj=[2, 2],
        bonds_per_state=[3, 2, 4, 1],
        d_model=4,
        seed=7,
        interleave=True,
    )
    _assert_pack_equivalence(*args)


def test_pack_trajectory_tokens_random_stress():
    rng = torch.Generator().manual_seed(123)
    for trial in range(20):
        n_trajs = int(torch.randint(1, 5, (1,), generator=rng).item())
        states_per_traj = torch.randint(1, 5, (n_trajs,), generator=rng).tolist()
        n_states_total = sum(states_per_traj)
        bonds_per_state = torch.randint(0, 6, (n_states_total,), generator=rng).tolist()
        args = _make_pack_inputs(
            n_trajs=n_trajs,
            states_per_traj=states_per_traj,
            bonds_per_state=bonds_per_state,
            d_model=8,
            seed=trial,
        )
        _assert_pack_equivalence(*args)
