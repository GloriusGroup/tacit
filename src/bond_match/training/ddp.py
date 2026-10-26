"""Helpers for PyTorch DistributedDataParallel (DDP) process management."""

import os
import torch
import torch.distributed as dist


def setup_ddp():
    """Initialize the NCCL process group and bind this process to its GPU.

    The collective timeout defaults to 12 hours and can be overridden via the
    ``BOND_MATCH_DDP_TIMEOUT_HOURS`` environment variable.

    Returns:
        int: The local rank read from the ``LOCAL_RANK`` environment variable.
    """
    from datetime import timedelta

    timeout_hours = float(os.environ.get("BOND_MATCH_DDP_TIMEOUT_HOURS", "12"))
    dist.init_process_group(
        backend="nccl", timeout=timedelta(hours=timeout_hours)
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Destroy the process group if distributed training is initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    """Return True if this process is rank 0 or DDP is not initialized."""
    return not dist.is_initialized() or dist.get_rank() == 0
