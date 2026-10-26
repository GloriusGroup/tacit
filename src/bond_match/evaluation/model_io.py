"""Model and configuration loading for evaluation."""

import torch
from bond_match.config import load_config
from bond_match.modules.lora import checkpoint_has_lora_weights
from bond_match.model import BondGAT


def load_eval_model(config_path, checkpoint_path, device):
    """Load a BondGAT model and its configuration for evaluation.

    Builds the model from the configuration with dropout disabled, loads the
    checkpoint weights, and puts the model in eval mode. Weights are loaded
    strictly, so the configuration must match the checkpoint. Raises
    ``RuntimeError`` if the checkpoint contains LoRA weights but LoRA is
    disabled in the configuration.

    Args:
        config_path: Path to the YAML configuration file.
        checkpoint_path: Path to the checkpoint file.
        device: Torch device to place the model on.

    Returns:
        Tuple ``(model, config)``.
    """
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    model = BondGAT.from_config(config, dropout=0.0).to(device)

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if checkpoint_has_lora_weights(state_dict) and not config.lora_enabled:
        raise RuntimeError(
            "Checkpoint contains LoRA weights, but config has lora.enabled=false. "
            "Use the matching LoRA config to evaluate this checkpoint."
        )
    model.load_state_dict(state_dict)
    model.eval()
    return model, config
