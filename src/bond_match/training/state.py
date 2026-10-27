"""Training state containers: precision modes, evaluation metrics, batch inputs."""

from dataclasses import dataclass
from enum import Enum
import torch


class Precision(str, Enum):
    """Numeric precision modes for training."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"

@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for one evaluation pass."""

    mrr: float = 0.0
    top1: float = 0.0
    top3: float = 0.0
    top5: float = 0.0
    top10: float = 0.0
    bond_type_acc: float = 0.0
    sim_separation: float = 0.0
    stop_acc: float = 0.0
    val_loss: float = 0.0

    @property
    def virtual_acc(self) -> float:
        """Alias for ``stop_acc``."""
        return self.stop_acc

@dataclass
class BatchInputs:
    """Unpacked and device-transferred batch tensors."""

    x: torch.Tensor
    y: torch.Tensor
    batch_idx: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    bond_indices: torch.Tensor
    is_over_target: torch.Tensor
    is_negative_reaction: torch.Tensor
    history_batch: object
    history_map: object
    step_index: torch.Tensor
    bond_batch_vec: torch.Tensor
    trajectory_ids: torch.Tensor | None = None

    @property
    def output_virtual(self) -> torch.Tensor:
        """Alias for ``is_over_target``."""
        return self.is_over_target

    @classmethod
    def from_batch(cls, batch: tuple, device: torch.device) -> "BatchInputs":
        """Unpack a collated batch tuple and move its tensors to a device.

        A 12-element batch carries trailing ``trajectory_ids``; the
        11-element form does not. ``bond_batch_vec`` is derived as the
        molecule index of each bond's first atom. Transfers use
        ``non_blocking=True`` on CUDA devices to allow asynchronous copies
        from pinned memory.

        Args:
            batch: Collated tuple of 11 or 12 elements.
            device: Target device for all tensors.

        Returns:
            BatchInputs: Populated container with device-resident tensors.
        """
        trajectory_ids = None
        if len(batch) == 12:
            *batch, trajectory_ids = batch
        (
            x,
            y,
            batch_idx,
            edge_index,
            edge_attr,
            bond_indices,
            is_over_target,
            is_negative_reaction,
            history_batch,
            history_map,
            step_index,
        ) = batch

        nb = device.type == "cuda"
        x = x.to(device, non_blocking=nb)
        y = y.to(device, non_blocking=nb)
        batch_idx = batch_idx.to(device, non_blocking=nb)
        edge_index = edge_index.to(device, non_blocking=nb)
        edge_attr = edge_attr.to(device, non_blocking=nb)
        bond_indices = bond_indices.to(device, non_blocking=nb)
        is_over_target = is_over_target.to(device, non_blocking=nb)
        is_negative_reaction = is_negative_reaction.to(device, non_blocking=nb)
        step_index = step_index.to(device, non_blocking=nb)
        bond_batch_vec = batch_idx[bond_indices[:, 0]]

        if trajectory_ids is not None:
            trajectory_ids = trajectory_ids.to(device, non_blocking=nb)

        return cls(
            x=x,
            y=y,
            batch_idx=batch_idx,
            edge_index=edge_index,
            edge_attr=edge_attr,
            bond_indices=bond_indices,
            is_over_target=is_over_target,
            is_negative_reaction=is_negative_reaction,
            history_batch=history_batch,
            history_map=history_map,
            step_index=step_index,
            bond_batch_vec=bond_batch_vec,
            trajectory_ids=trajectory_ids,
        )
