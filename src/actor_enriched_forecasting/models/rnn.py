from __future__ import annotations

import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..features import FeatureConfig, actual_future_target, feature_engineering, reconstruct_prediction


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_attention_rnn(
    input_shape,
    rnn_type: str = "gru",
    hidden_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    filters: int = 64,
    kernel_size: int = 3,
    pool_size: int = 2,
    attention_heads: int = 4,
    attention_key_dim: int = 32,
):
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        Conv1D, Dense, Dropout, GRU, Input, LSTM, LayerNormalization,
        MaxPooling1D, MultiHeadAttention, GlobalAveragePooling1D
    )

    inputs = Input(shape=input_shape)
    x = Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=pool_size)(x)
    recurrent = GRU if rnn_type.lower() == "gru" else LSTM
    x = recurrent(hidden_units, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    attention = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_key_dim)(x, x)
    x = LayerNormalization()(x + attention)
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model


def make_sequences(frame: pd.DataFrame, features: list[str], target_column: str = "target_scaled", steps: int = 15):
    X, y, indices = [], [], []
    for i in range(steps, len(frame)):
        X.append(frame[features].iloc[i - steps:i].to_numpy(dtype=float))
        y.append(float(frame[target_column].iloc[i]))
        indices.append(frame.index[i])
    return np.asarray(X), np.asarray(y), indices


# Training orchestration is kept deliberately small here: final tuned architectures
# should be called from scripts/04_run_rnn_models.py after final parameter selection.
