"""Non-blocking, retry-capable MLflow logging utilities."""

import time
import mlflow

from bond_match.training.constants import (
    _mlflow_executor,
)

def _safe_mlflow_log(fn, *args, retries=3, backoff=1.0, **kwargs):
    """Submit an MLflow logging call to a background thread with retries.

    MLflow's ``active_run()`` is thread-local, so the background thread cannot
    see the run started on the main thread. The active run id is captured here
    and the call is wrapped in ``mlflow.start_run(run_id=...)`` so metrics and
    artifacts land in the correct run instead of auto-creating a new one.

    Args:
        fn: MLflow logging function to invoke (e.g. ``mlflow.log_metrics``).
        *args: Positional arguments forwarded to ``fn``.
        retries: Number of attempts before the call is skipped with a warning.
        backoff: Base sleep in seconds between attempts, scaled linearly.
        **kwargs: Keyword arguments forwarded to ``fn``.
    """
    active = mlflow.active_run()
    run_id = active.info.run_id if active else None

    def _do():
        """Execute the logging call with retries inside the captured run.

        ``mlflow.end_run(status="RUNNING")`` in the finally block clears the
        thread-local active-run binding without ending the run itself, leaving
        the run open in the backend.
        """
        if run_id:
            mlflow.start_run(run_id=run_id, nested=False)
        try:
            for attempt in range(retries):
                try:
                    return fn(*args, **kwargs)
                except mlflow.exceptions.MlflowException:
                    if attempt < retries - 1:
                        time.sleep(backoff * (attempt + 1))
                    else:
                        print(
                            f"  [WARNING] mlflow logging failed after {retries} retries, skipping"
                        )
        finally:
            if run_id:
                mlflow.end_run(status="RUNNING")

    _mlflow_executor.submit(_do)
