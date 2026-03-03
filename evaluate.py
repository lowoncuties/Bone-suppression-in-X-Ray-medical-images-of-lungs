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
    p = argparse.ArgumentParser(description="Evaluate saved model checkpoint.")
    p.add_argument("--config", type=str, default="configs/base.json")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def run(args):
    import tensorflow as tf

    from bonesuppression.config import load_config
    from bonesuppression.data.dataset import load_paired_data, make_tf_dataset
    from bonesuppression.eval.report import write_evaluation_report, write_methods_summary
    from bonesuppression.models.losses import mixed_loss_l1, mixed_loss_l2, ms_ssim_loss, ms_ssim_metric, psnr
    from bonesuppression.training.trainer import evaluate_model, predict_samples
    from bonesuppression.utils.repro import set_global_seed

    config = load_config(args.config)
    set_global_seed(config.train.seed)
    _, _, _, _, x_test, y_test = load_paired_data(config.data, config.train.seed)
    test_ds = make_tf_dataset(x_test, y_test, config.train, training=False)
    custom = {"ms_ssim_loss": ms_ssim_loss, "mixed_loss_l2": mixed_loss_l2, "mixed_loss_l1": mixed_loss_l1, "ms_ssim_metric": ms_ssim_metric, "psnr": psnr}
    model = tf.keras.models.load_model(args.checkpoint, custom_objects=custom)
    metrics = evaluate_model(model, test_ds)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    predict_samples(model, x_test, run_dir / "predictions")
    write_evaluation_report(config, metrics, run_dir)
    write_methods_summary(config, metrics, run_dir)
    print(metrics)


if __name__ == "__main__":
    run(parse_args())
