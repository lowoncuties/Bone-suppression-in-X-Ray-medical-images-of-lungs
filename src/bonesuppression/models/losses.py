from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K


def ms_ssim_loss(y_true, y_pred, max_val: float = 1.0):
    return 1.0 - tf.image.ssim_multiscale(y_true, y_pred, max_val=max_val)


def mixed_loss_l2(y_true, y_pred, alpha: float = 0.84, max_val: float = 1.0):
    msssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=max_val)
    l2 = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    return alpha * (1.0 - msssim) + (1.0 - alpha) * l2


def mixed_loss_l1(y_true, y_pred, alpha: float = 0.84, max_val: float = 1.0):
    msssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=max_val)
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])
    return alpha * (1.0 - msssim) + (1.0 - alpha) * l1


def ms_ssim_metric(y_true, y_pred):
    return tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)


def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    return 10.0 * (K.log(1.0 / mse) / K.log(10.0))


def resolve_loss(name: str):
    name = name.lower()
    if name == "mse":
        return "mse"
    if name == "ms_ssim":
        return ms_ssim_loss
    if name == "mixed_l2":
        return mixed_loss_l2
    if name == "mixed_l1":
        return mixed_loss_l1
    raise ValueError(f"Unsupported loss: {name}")
