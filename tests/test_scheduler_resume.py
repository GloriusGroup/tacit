"""Tests for learning-rate scheduler restoration when resuming training
under the available ResumeSchedulerMode settings."""
import torch

from bond_match.config import RankingConfig, ResumeSchedulerMode
from bond_match.train_ranking import Trainer


def _nested_config(train: dict) -> dict:
    """Wrap a train section into a complete nested RankingConfig dict."""
    return {
        "detection_type": "ranking",
        "model": {},
        "loss": {},
        "train": train,
        "eval": {},
        "data": {},
    }


def _optimizer_and_scheduler(epochs: int = 8):
    """Create an AdamW optimizer and cosine-annealing scheduler over the
    given epoch horizon."""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=5e-5,
    )
    return optimizer, scheduler


def test_resume_scheduler_mode_parses_from_train_config():
    """resume_scheduler_mode in the train section parses to the
    ResumeSchedulerMode enum."""
    cfg = RankingConfig.model_validate(
        _nested_config({"resume_scheduler_mode": "checkpoint"})
    )

    assert cfg.resume_scheduler_mode == ResumeSchedulerMode.CHECKPOINT


def test_remaining_resume_scheduler_does_not_jump_after_completed_cosine():
    """In 'remaining' mode, resuming after a completed 8-epoch cosine
    schedule with a 12-epoch target rebuilds the scheduler over the 4
    remaining epochs: training restarts at epoch 8 with the learning rate at
    eta_min and decays monotonically afterwards."""
    old_optimizer, old_scheduler = _optimizer_and_scheduler(epochs=8)
    for _ in range(8):
        old_optimizer.step()
        old_scheduler.step()

    checkpoint = {
        "optimizer_state_dict": old_optimizer.state_dict(),
        "scheduler_state_dict": old_scheduler.state_dict(),
        "epoch": 7,
        "batch_idx": -1,
    }

    cfg = RankingConfig.model_validate(
        _nested_config(
            {
                "epochs": 12,
                "learning_rate": 2e-4,
                "eta_min": 5e-5,
                "resume_scheduler_mode": "remaining",
            }
        )
    )
    new_optimizer, new_scheduler = _optimizer_and_scheduler(epochs=12)
    trainer = Trainer.__new__(Trainer)
    trainer.config = cfg
    trainer.optimizer = new_optimizer
    trainer.scheduler = new_scheduler
    trainer.scaler = None
    trainer._deferred_checkpoint = checkpoint
    trainer.start_epoch = 0
    trainer._resume_batch_idx = -1

    trainer._restore_training_state()

    resumed_lr = trainer.optimizer.param_groups[0]["lr"]
    assert trainer.start_epoch == 8
    assert trainer.scheduler.T_max == 4
    assert resumed_lr == cfg.eta_min

    trainer.optimizer.step()
    trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] <= resumed_lr


def test_checkpoint_resume_scheduler_preserves_checkpoint_t_max():
    """In 'checkpoint' mode, the restored scheduler keeps the checkpoint's
    T_max of 8 rather than adopting the new 12-epoch target."""
    old_optimizer, old_scheduler = _optimizer_and_scheduler(epochs=8)
    for _ in range(8):
        old_optimizer.step()
        old_scheduler.step()

    checkpoint = {
        "optimizer_state_dict": old_optimizer.state_dict(),
        "scheduler_state_dict": old_scheduler.state_dict(),
        "epoch": 7,
        "batch_idx": -1,
    }

    cfg = RankingConfig.model_validate(
        _nested_config(
            {
                "epochs": 12,
                "learning_rate": 2e-4,
                "eta_min": 5e-5,
                "resume_scheduler_mode": "checkpoint",
            }
        )
    )
    new_optimizer, new_scheduler = _optimizer_and_scheduler(epochs=12)
    trainer = Trainer.__new__(Trainer)
    trainer.config = cfg
    trainer.optimizer = new_optimizer
    trainer.scheduler = new_scheduler
    trainer.scaler = None
    trainer._deferred_checkpoint = checkpoint
    trainer.start_epoch = 0
    trainer._resume_batch_idx = -1

    trainer._restore_training_state()

    assert trainer.start_epoch == 8
    assert trainer.scheduler.T_max == 8
