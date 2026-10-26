"""Optimizer wrapper that combines two optimizers behind one interface."""

import torch.optim as optim


class _DualOptimizer(optim.Optimizer):
    """Wraps two optimizers (e.g. MuonClip + AdamW) into a single interface.

    Inherits from ``optim.Optimizer`` so learning-rate schedulers accept it.
    ``Optimizer.__init__`` is intentionally not called; ``param_groups`` is the
    concatenation of both sub-optimizers' groups so schedulers can read and
    write per-group learning rates. ``state`` is an empty dict because each
    sub-optimizer keeps its own state.
    """

    def __init__(self, opt1, opt2):
        """Store both sub-optimizers and expose their combined param groups.

        Args:
            opt1: First optimizer; its ``defaults`` are exposed as this
                wrapper's ``defaults``.
            opt2: Second optimizer.
        """
        self.opt1 = opt1
        self.opt2 = opt2
        self.defaults = opt1.defaults
        self.param_groups = opt1.param_groups + opt2.param_groups
        self.state = {}

    def step(self, closure=None):
        """Step both sub-optimizers."""
        self.opt1.step(closure)
        self.opt2.step(closure)

    def zero_grad(self, set_to_none=True):
        """Zero gradients on both sub-optimizers."""
        self.opt1.zero_grad(set_to_none=set_to_none)
        self.opt2.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return both sub-optimizer state dicts under ``opt1``/``opt2`` keys."""
        return {"opt1": self.opt1.state_dict(), "opt2": self.opt2.state_dict()}

    def load_state_dict(self, state_dict):
        """Restore both sub-optimizer states from a ``state_dict()`` payload."""
        self.opt1.load_state_dict(state_dict["opt1"])
        self.opt2.load_state_dict(state_dict["opt2"])
