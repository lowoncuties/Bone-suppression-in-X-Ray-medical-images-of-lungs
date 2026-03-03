from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from bonesuppression.config import AppConfig
from bonesuppression.models.losses import ms_ssim_metric, psnr, resolve_loss


def train_supervised(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    config: AppConfig,
    run_dir: Path,
) -> tf.keras.callbacks.History:
    loss = resolve_loss(config.model.loss)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.train.learning_rate),
        loss=loss,
        metrics=[psnr, ms_ssim_metric],
    )

    ckpt_path = run_dir / "checkpoints" / "best.keras"
    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
        CSVLogger(str(run_dir / "metrics" / "keras_history.csv")),
    ]

    if config.experiment.resume_checkpoint:
        model.load_weights(config.experiment.resume_checkpoint)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.train.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(run_dir / "checkpoints" / "last.keras")
    _save_curves(history.history, run_dir / "figures" / "training_curves.png")
    return history


def evaluate_model(model: tf.keras.Model, test_ds) -> Dict[str, float]:
    vals = model.evaluate(test_ds, return_dict=True, verbose=0)
    return {k: float(v) for k, v in vals.items()}


def predict_samples(model: tf.keras.Model, x: np.ndarray, out_dir: Path, max_items: int = 8) -> None:
    import cv2

    preds = model.predict(x[:max_items], verbose=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(preds):
        img = np.clip(arr.squeeze() * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"pred_{i:03d}.png"), img)


def _save_curves(history: Dict[str, list], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    for key in ["loss", "val_loss", "psnr", "val_psnr", "ms_ssim_metric", "val_ms_ssim_metric"]:
        if key in history:
            plt.plot(history[key], label=key)
    plt.legend()
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
