#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser(description="Train bone suppression model.")
    p.add_argument("--config", type=str, default="configs/base.json")
    p.add_argument("--source-dir", type=str)
    p.add_argument("--target-dir", type=str)
    p.add_argument("--image-size", type=int)
    p.add_argument("--model", type=str, choices=["autoencoder", "unet"])
    p.add_argument("--loss", type=str, choices=["mse", "ms_ssim", "mixed_l2", "mixed_l1"])
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir", type=str)
    p.add_argument("--experiment-name", type=str)
    p.add_argument("--resume", type=str)
    return p.parse_args()


def build_overrides(args) -> dict:
    ov = {"data": {}, "model": {}, "train": {}, "experiment": {}}
    mapping = [
        ("source_dir", ("data", "source_dir")), ("target_dir", ("data", "target_dir")),
        ("image_size", ("data", "image_size")), ("model", ("model", "name")),
        ("loss", ("model", "loss")), ("lr", ("train", "learning_rate")),
        ("batch_size", ("train", "batch_size")), ("epochs", ("train", "epochs")),
        ("seed", ("train", "seed")), ("output_dir", ("experiment", "output_dir")),
        ("experiment_name", ("experiment", "experiment_name")), ("resume", ("experiment", "resume_checkpoint")),
    ]
    for src, (s1, s2) in mapping:
        val = getattr(args, src)
        if val is not None:
            ov[s1][s2] = val
    return {k: v for k, v in ov.items() if v}


def run(args):
    from bonesuppression.config import load_config, save_config, save_config_json
    from bonesuppression.data.dataset import load_paired_data, make_tf_dataset
    from bonesuppression.eval.report import write_evaluation_report, write_methods_summary
    from bonesuppression.models.factory import build_model
    from bonesuppression.training.trainer import evaluate_model, predict_samples, train_supervised
    from bonesuppression.utils.io import build_run_dir
    from bonesuppression.utils.repro import set_global_seed

    config = load_config(args.config, build_overrides(args))
    set_global_seed(config.train.seed)
    run_dir = build_run_dir(config.experiment.output_dir, config.experiment.experiment_name)
    save_config(config, run_dir / "reports" / "resolved_config.yaml")
    save_config_json(config, run_dir / "reports" / "resolved_config.json")
    x_train, y_train, x_val, y_val, x_test, y_test = load_paired_data(config.data, config.train.seed)
    train_ds = make_tf_dataset(x_train, y_train, config.train, training=True)
    val_ds = make_tf_dataset(x_val, y_val, config.train, training=False)
    test_ds = make_tf_dataset(x_test, y_test, config.train, training=False)
    model = build_model(config.model, input_shape=(config.data.image_size, config.data.image_size, config.data.channels))
    train_supervised(model, train_ds, val_ds, config, run_dir)
    metrics = evaluate_model(model, test_ds)
    predict_samples(model, x_test, run_dir / "predictions")
    write_evaluation_report(config, metrics, run_dir)
    write_methods_summary(config, metrics, run_dir)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    run(parse_args())
