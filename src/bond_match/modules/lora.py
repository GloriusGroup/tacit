"""Low-rank adaptation (LoRA) for Linear layers.

Provides a LoRALinear layer with a frozen base weight and a trainable low-rank
residual, plus utilities to inject adapters into a model, freeze non-adapter
parameters, collect parameter statistics, and detect adapter weights in a
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """Linear layer with a trainable low-rank residual adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """Initialize a Linear layer with a frozen base weight and LoRA adapter.

        The base weight and bias are frozen; only lora_A and lora_B are
        trainable. lora_B is zero-initialized so the adapter is initially a
        no-op. The adapter output is scaled by alpha / rank.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            bias: Whether the base layer has a bias.
            rank: Adapter rank; must be positive.
            alpha: Adapter scaling numerator.
            dropout: Dropout applied to the adapter input.
        """
        super().__init__(in_features, out_features, bias=bias)
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_lora_parameters()
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_lora_parameters(self) -> None:
        """Reinitialize adapter parameters (lora_A Kaiming-uniform, lora_B zero)."""
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        """Build a LoRALinear that copies an existing Linear layer's weights.

        Args:
            linear: Source Linear layer whose weight/bias and device/dtype are
                copied into the new adapter layer.
            rank: Adapter rank.
            alpha: Adapter scaling numerator.
            dropout: Dropout applied to the adapter input.

        Returns:
            A LoRALinear with the frozen base weights copied from linear.
        """
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None and module.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base layer plus the scaled low-rank adapter.

        Args:
            input: Input tensor, shape (..., in_features).

        Returns:
            Output tensor, shape (..., out_features).
        """
        output = F.linear(input, self.weight, self.bias)
        lora_hidden = F.linear(self.lora_dropout(input), self.lora_A)
        lora_update = F.linear(lora_hidden, self.lora_B)
        return output + self.scaling * lora_update


@dataclass(frozen=True)
class LoRAStats:
    """Summary of LoRA application: adapter count and parameter totals."""

    replaced_modules: int
    trainable_params: int
    total_params: int


def _matches_target(module_name: str, targets: list[str]) -> bool:
    """Return True if any target substring occurs in module_name."""
    return any(target in module_name for target in targets)


def apply_lora_to_model(
    module: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Replace matching Linear layers in a model with LoRALinear adapters.

    Recurses through the module tree and swaps each nn.Linear whose qualified
    name matches a target substring (and that is not already a LoRALinear).

    Args:
        module: Root module to modify in place.
        target_modules: Substrings matched against qualified submodule names.
        rank: Adapter rank.
        alpha: Adapter scaling numerator.
        dropout: Dropout applied to the adapter input.

    Returns:
        Qualified names of the replaced modules.
    """
    replaced: list[str] = []

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        """Recursively replace matching Linear children under parent."""
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                if _matches_target(full_name, target_modules):
                    setattr(
                        parent,
                        child_name,
                        LoRALinear.from_linear(
                            child,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                        ),
                    )
                    replaced.append(full_name)
                    continue
            _replace(child, full_name)

    _replace(module)
    return replaced


def freeze_non_lora_parameters(module: nn.Module) -> None:
    """Set requires_grad to True only for lora_A and lora_B parameters."""
    for name, param in module.named_parameters():
        param.requires_grad = name.endswith("lora_A") or name.endswith("lora_B")


def collect_lora_stats(module: nn.Module) -> LoRAStats:
    """Count LoRALinear modules and trainable / total parameters.

    Args:
        module: Module to inspect.

    Returns:
        LoRAStats with adapter count and parameter totals.
    """
    replaced_modules = sum(1 for m in module.modules() if isinstance(m, LoRALinear))
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return LoRAStats(
        replaced_modules=replaced_modules,
        trainable_params=trainable_params,
        total_params=total_params,
    )


def checkpoint_has_lora_weights(state_dict: dict[str, torch.Tensor]) -> bool:
    """Return True if any state-dict key is a LoRA adapter parameter."""
    return any(key.endswith("lora_A") or key.endswith("lora_B") for key in state_dict)
