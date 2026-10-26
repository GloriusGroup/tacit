"""Unit tests for training-package components: state, EMA, loss builder,
dual optimizer, and the eval collector."""
import torch
import torch.nn as nn

from bond_match.config import RankingConfig
from bond_match.training.state import Precision, EvalMetrics
from bond_match.training.ema import ModelEMA
from bond_match.training.loss import build_loss
from bond_match.training.optimizer import _DualOptimizer
from bond_match.training.eval_collector import EvalCollector
from bond_match.training.constants import TOP_K_VALUES
from bond_match.losses.ranking_loss import PairRankingLoss, EBMPairRankingLoss


def test_precision_enum():
    """Precision enum values map to their dtype names."""
    assert Precision.FLOAT32.value == "float32"
    assert Precision.BFLOAT16.value == "bfloat16"


def test_eval_metrics_defaults_and_virtual_alias():
    """EvalMetrics defaults to zero and virtual_acc aliases stop_acc."""
    em = EvalMetrics()
    assert em.mrr == 0.0 and em.top1 == 0.0
    assert em.virtual_acc == em.stop_acc


def test_model_ema_update_formula():
    """ModelEMA.update follows shadow = decay*old + (1-decay)*new; with
    decay 0.5, old weights 0, and new weights 1 the shadow becomes 0.5."""
    lin = nn.Linear(4, 3)
    with torch.no_grad():
        lin.weight.fill_(0.0)
    ema = ModelEMA(lin, decay=0.5)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema.update(lin)
    assert torch.allclose(ema.shadow["weight"], torch.full_like(lin.weight, 0.5))


def test_model_ema_apply_swaps_and_restores():
    """ema.apply swaps in the shadow weights (initialized to the model values
    at construction, 2.0) inside the context and restores the diverged live
    weights (5.0) afterwards."""
    lin = nn.Linear(4, 3)
    with torch.no_grad():
        lin.weight.fill_(2.0)
    ema = ModelEMA(lin, decay=0.9)
    with torch.no_grad():
        lin.weight.fill_(5.0)
    with ema.apply(lin):
        assert torch.allclose(lin.weight, torch.full_like(lin.weight, 2.0))
    assert torch.allclose(lin.weight, torch.full_like(lin.weight, 5.0))


def test_build_loss_selects_implementation():
    """build_loss returns PairRankingLoss with the configured temperature for
    'softmax' and EBMPairRankingLoss for 'ebm'."""
    soft = build_loss(
        RankingConfig.model_validate({"ranking_loss_type": "softmax", "temperature": 0.07})
    )
    assert isinstance(soft, PairRankingLoss)
    assert soft.temperature == 0.07
    ebm = build_loss(RankingConfig.model_validate({"ranking_loss_type": "ebm"}))
    assert isinstance(ebm, EBMPairRankingLoss)


def test_dual_optimizer_steps_both():
    """_DualOptimizer.step applies both inner optimizers to the shared
    parameter (two lr=1 SGD steps on a unit gradient give -2) and state_dict
    returns a dict."""
    p = nn.Parameter(torch.zeros(3))
    o1 = torch.optim.SGD([p], lr=1.0)
    o2 = torch.optim.SGD([p], lr=1.0)
    dual = _DualOptimizer(o1, o2)
    p.grad = torch.ones(3)
    dual.step()
    assert torch.allclose(p.detach(), torch.full((3,), -2.0))
    sd = dual.state_dict()
    assert isinstance(sd, dict)


def test_top_k_values_constant():
    """TOP_K_VALUES is (1, 3, 5, 10)."""
    assert TOP_K_VALUES == (1, 3, 5, 10)


def test_eval_collector_ranking_accumulation():
    """A rank-0 hit counts toward every top-k bucket; a rank-3 hit counts
    only toward top-5 and top-10. Reciprocal ranks are recorded in order."""
    col = EvalCollector()
    col.add_ranking_hit(0, 1.0)
    col.add_ranking_hit(3, 0.25)
    assert col.top_hits == {1: 1, 3: 1, 5: 2, 10: 2}
    assert col.reciprocal_ranks == [1.0, 0.25]


def test_eval_collector_loss_filters_nonfinite():
    """Non-finite losses are excluded from the loss sum and count and tallied
    separately."""
    col = EvalCollector()
    col.add_loss(1.0)
    col.add_loss(float("nan"))
    col.add_loss(3.0)
    assert col.loss_count == 2
    assert col.loss_sum == 4.0
    assert col.nonfinite_loss_count == 1


def test_eval_collector_compute_metrics():
    """compute_metrics aggregates collected hits into EvalMetrics: top1 is
    0.5 (1 of 2 samples ranked first), MRR is (1.0 + 0.25)/2 = 0.625, and
    stop accuracy is 0.5."""
    col = EvalCollector()
    col.total_reactive = 2
    col.add_ranking_hit(0, 1.0)
    col.add_ranking_hit(3, 0.25)
    col.add_preds([1, 0], [1, 1])
    col.add_bond_type(0, 0)
    col.add_stop(1, 2)
    metrics = col.compute_metrics()
    assert isinstance(metrics, EvalMetrics)
    assert metrics.top1 == 0.5
    assert abs(metrics.mrr - 0.625) < 1e-6
    assert metrics.stop_acc == 0.5
