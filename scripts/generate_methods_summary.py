#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Generate a Methods-section reproducibility summary.")
    p.add_argument("--resolved-config", required=True, help="Path to resolved_config.json")
    p.add_argument("--evaluation-report", required=True, help="Path to evaluation_report.json")
    p.add_argument("--output", required=True, help="Output markdown path")
    args = p.parse_args()

    cfg = json.loads(Path(args.resolved_config).read_text())
    rep = json.loads(Path(args.evaluation_report).read_text())

    lines = [
        "# Reproducibility / Experimental Setup",
        "",
        f"- Dataset source directory: `{cfg['data']['source_dir']}`",
        f"- Dataset target directory: `{cfg['data']['target_dir']}`",
        f"- Input size: {cfg['data']['image_size']}x{cfg['data']['image_size']} ({cfg['data']['channels']} channel)",
        f"- Model: {cfg['model']['name']}",
        f"- Loss: {cfg['model']['loss']}",
        f"- Learning rate: {cfg['train']['learning_rate']}",
        f"- Batch size: {cfg['train']['batch_size']}",
        f"- Epochs: {cfg['train']['epochs']}",
        f"- Seed: {cfg['train']['seed']}",
        "",
        "## Final metrics",
    ]
    for k, v in rep.get("metrics", {}).items():
        lines.append(f"- {k}: {v}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
