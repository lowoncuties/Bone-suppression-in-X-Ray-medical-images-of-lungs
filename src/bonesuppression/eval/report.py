from __future__ import annotations

from pathlib import Path
from typing import Dict

from bonesuppression.config import AppConfig, config_to_dict
from bonesuppression.utils.logging import write_json, write_markdown


def write_evaluation_report(config: AppConfig, metrics: Dict[str, float], run_dir: Path) -> None:
    report = {
        "experiment": config.experiment.experiment_name,
        "model": config.model.name,
        "loss": config.model.loss,
        "data": {
            "source_dir": config.data.source_dir,
            "target_dir": config.data.target_dir,
            "image_size": config.data.image_size,
        },
        "training": {
            "epochs": config.train.epochs,
            "batch_size": config.train.batch_size,
            "learning_rate": config.train.learning_rate,
            "seed": config.train.seed,
        },
        "metrics": metrics,
    }
    write_json(run_dir / "reports" / "evaluation_report.json", report)


def write_methods_summary(config: AppConfig, metrics: Dict[str, float], run_dir: Path) -> None:
    lines = [
        "# Reproducibility / Experimental Setup",
        "",
        f"- Dataset source directory: `{config.data.source_dir}`",
        f"- Dataset target directory: `{config.data.target_dir}`",
        f"- Input resolution: {config.data.image_size}×{config.data.image_size}, grayscale",
        f"- Model: {config.model.name}",
        f"- Loss: {config.model.loss}",
        f"- Optimizer: Adam (lr={config.train.learning_rate})",
        f"- Batch size: {config.train.batch_size}",
        f"- Epochs: {config.train.epochs}",
        f"- Random seed: {config.train.seed}",
        "",
        "## Final test metrics",
    ]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v:.6f}")

    lines += [
        "",
        "## Mapping to paper sections",
        "- **Data**: directories and preprocessing listed above correspond to Methods/Data section.",
        "- **Training setup**: optimizer, learning rate, epochs, and batch size correspond to Methods/Training section.",
        "- **Metrics**: loss/PSNR/MS-SSIM outputs correspond to Results tables and quantitative comparison.",
    ]
    write_markdown(run_dir / "reports" / "reproducibility_methods_summary.md", lines)
    write_json(run_dir / "reports" / "resolved_config.json", config_to_dict(config))
