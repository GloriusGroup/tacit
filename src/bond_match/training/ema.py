"""Exponential moving average (EMA) of model parameters for evaluation."""

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    Shadow values are updated after each optimizer step via
    ``shadow = decay * shadow + (1 - decay) * param``. EMA weights are
    swapped in for evaluation through the ``apply`` context manager.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Clone the model state dict as the initial shadow parameters.

        Args:
            model: Model whose parameters are tracked.
            decay: Multiplicative decay applied to the shadow on each update.
        """
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Blend current model parameters into the shadow with the EMA decay."""
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v, alpha=1 - d)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        """Return a context manager that swaps in EMA weights and restores the originals on exit."""
        return _EMAContext(model, self.shadow)

    def state_dict(self) -> dict:
        """Return a cloned copy of the shadow parameters for checkpointing."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict):
        """Copy checkpointed shadow values into matching shadow entries."""
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

class _EMAContext:
    """Context manager that temporarily loads EMA shadow weights into a model."""

    def __init__(self, model: nn.Module, shadow: dict):
        """Store the model and shadow weights to swap on entry."""
        self.model = model
        self.shadow = shadow
        self.backup = {}

    def __enter__(self):
        """Back up current weights and load the shadow weights (non-strict)."""
        self.backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow, strict=False)
        return self.model

    def __exit__(self, *args):
        """Restore the backed-up weights (non-strict)."""
        self.model.load_state_dict(self.backup, strict=False)
