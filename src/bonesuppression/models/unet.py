from __future__ import annotations

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, UpSampling2D


def _block(x, f: int):
    x = Conv2D(f, 3, activation="relu", padding="same")(x)
    x = Conv2D(f, 3, activation="relu", padding="same")(x)
    return x


def build_unet(input_shape=(512, 512, 1), base_filters: int = 32) -> Model:
    inp = Input(shape=input_shape)
    c1 = _block(inp, base_filters)
    p1 = MaxPooling2D()(c1)

    c2 = _block(p1, base_filters * 2)
    p2 = MaxPooling2D()(c2)

    b = _block(p2, base_filters * 4)

    u2 = UpSampling2D()(b)
    u2 = Concatenate()([u2, c2])
    c3 = _block(u2, base_filters * 2)

    u1 = UpSampling2D()(c3)
    u1 = Concatenate()([u1, c1])
    c4 = _block(u1, base_filters)

    out = Conv2D(1, 1, activation="sigmoid")(c4)
    return Model(inp, out, name="unet")
