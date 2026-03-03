from __future__ import annotations

from bonesuppression.config import ModelConfig
from bonesuppression.models.autoencoder import build_autoencoder
from bonesuppression.models.unet import build_unet


def build_model(model_cfg: ModelConfig, input_shape):
    name = model_cfg.name.lower()
    if name == "autoencoder":
        return build_autoencoder(input_shape=input_shape, base_filters=model_cfg.base_filters)
    if name == "unet":
        return build_unet(input_shape=input_shape, base_filters=model_cfg.base_filters)
    raise ValueError(f"Unsupported model: {model_cfg.name}")
