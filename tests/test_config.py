"""Unit tests for bond_match.config: schema, flat/nested restructuring, IO."""
import pytest

from bond_match.config import (
    RankingConfig,
    PairSelectionMode,
    ResumeSchedulerMode,
    DataVersion,
    load_config,
    save_config,
)


def test_enums_values():
    """Configuration enums expose the expected string values."""
    assert PairSelectionMode.ARGMAX.value == "argmax"
    assert PairSelectionMode.SOFTMAX.value == "softmax"
    assert {m.value for m in ResumeSchedulerMode} == {"checkpoint", "remaining"}
    assert DataVersion.V3_BLOCKSPACE.value == "v3_blockspace"


def test_flat_config_is_restructured_into_nested():
    """A flat config without a top-level 'model' key is restructured into nested sub-configs."""
    cfg = RankingConfig.model_validate(
        {"embed_dim": 123, "learning_rate": 0.0033, "batch_size": 7}
    )
    assert cfg.model.embed_dim == 123
    assert cfg.train.learning_rate == 0.0033
    assert cfg.train.batch_size == 7


def test_backward_compat_property_proxies_flat_access():
    """Top-level attribute access proxies the corresponding nested config value."""
    cfg = RankingConfig.model_validate({"embed_dim": 123})
    assert cfg.embed_dim == cfg.model.embed_dim == 123


def test_nested_config_preserved():
    """A config with a top-level 'model' key is validated as-is, preserving nested sections."""
    cfg = RankingConfig.model_validate(
        {"model": {"embed_dim": 256}, "train": {"batch_size": 9}}
    )
    assert cfg.model.embed_dim == 256
    assert cfg.train.batch_size == 9


def test_save_config_writes_file(tmp_path):
    """save_config writes a non-empty YAML file."""
    cfg = RankingConfig.model_validate({"embed_dim": 64, "batch_size": 5})
    path = tmp_path / "cfg.yaml"
    save_config(cfg, str(path))
    assert path.exists() and path.stat().st_size > 0


def test_load_config_from_yaml(tmp_path):
    """load_config parses a nested YAML file into a RankingConfig."""
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"model": {"embed_dim": 64}, "train": {"batch_size": 5}})
    )
    cfg = load_config(str(path))
    assert cfg.model.embed_dim == 64
    assert cfg.train.batch_size == 5
