from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_run_dir(output_dir: str, experiment_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / experiment_name / run_id
    for sub in ["checkpoints", "metrics", "reports", "figures", "predictions"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir
