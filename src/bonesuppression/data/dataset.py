from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from bonesuppression.config import DataConfig, TrainConfig


def _load_grayscale_sorted(folder: str, image_size: int) -> Tuple[np.ndarray, List[str]]:
    fpath = Path(folder)
    files = sorted([p for p in fpath.iterdir() if p.is_file()])
    images = []
    names = []
    for file in files:
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        images.append(img.astype(np.float32) / 255.0)
        names.append(file.name)
    arr = np.array(images, dtype=np.float32)[..., np.newaxis]
    return arr, names


def load_paired_data(data_cfg: DataConfig, seed: int):
    src, src_names = _load_grayscale_sorted(data_cfg.source_dir, data_cfg.image_size)
    tgt, tgt_names = _load_grayscale_sorted(data_cfg.target_dir, data_cfg.image_size)
    common = sorted(set(src_names).intersection(tgt_names))
    if not common:
        raise ValueError("No paired files found between source and target directories.")

    src_map = {n: i for i, n in enumerate(src_names)}
    tgt_map = {n: i for i, n in enumerate(tgt_names)}
    x = np.array([src[src_map[n]] for n in common], dtype=np.float32)
    y = np.array([tgt[tgt_map[n]] for n in common], dtype=np.float32)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=data_cfg.val_fraction + data_cfg.test_fraction, random_state=seed, shuffle=True
    )
    rel_test = data_cfg.test_fraction / (data_cfg.val_fraction + data_cfg.test_fraction)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=rel_test, random_state=seed, shuffle=True
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def make_tf_dataset(x: np.ndarray, y: np.ndarray, train_cfg: TrainConfig, training: bool):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(train_cfg.shuffle_buffer, seed=train_cfg.seed, reshuffle_each_iteration=True)
    return ds.batch(train_cfg.batch_size).prefetch(tf.data.AUTOTUNE)
