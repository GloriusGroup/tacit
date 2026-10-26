"""Training package for ranking-based bond-pair prediction.

Re-exports the public training API: configuration constants, state
containers, DDP helpers, MLflow utilities, evaluation collection, the
optimizer wrapper, loss construction, EMA support, the Trainer, and the
command-line entry point.
"""

from bond_match.training.constants import (
    _mlflow_executor,
    TOP_K_VALUES,
)
from bond_match.training.state import (
    Precision,
    EvalMetrics,
    BatchInputs,
)
from bond_match.training.ddp import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
)
from bond_match.training.mlflow_utils import (
    _safe_mlflow_log,
)
from bond_match.training.eval_collector import (
    EvalCollector,
)
from bond_match.training.optimizer import (
    _DualOptimizer,
)
from bond_match.training.loss import (
    build_loss,
)
from bond_match.training.ema import (
    ModelEMA,
    _EMAContext,
)
from bond_match.training.trainer import (
    Trainer,
)
from bond_match.training.cli import (
    main,
)

__all__ = [
    '_mlflow_executor',
    'TOP_K_VALUES',
    'Precision',
    'EvalMetrics',
    'BatchInputs',
    'setup_ddp',
    'cleanup_ddp',
    'is_main_process',
    '_safe_mlflow_log',
    'EvalCollector',
    '_DualOptimizer',
    'build_loss',
    'ModelEMA',
    '_EMAContext',
    'Trainer',
    'main',
]
