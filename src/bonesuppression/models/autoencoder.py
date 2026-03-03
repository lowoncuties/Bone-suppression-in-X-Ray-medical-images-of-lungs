from __future__ import annotations

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def build_autoencoder(input_shape=(512, 512, 1), base_filters: int = 32) -> Model:
    inp = Input(shape=input_shape)
    x = Conv2D(base_filters, 3, activation="relu", padding="same")(inp)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(base_filters * 2, 3, activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(base_filters * 2, 3, activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(base_filters, 3, activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    out = Conv2D(1, 1, activation="sigmoid", padding="same")(x)
    return Model(inp, out, name="autoencoder")
