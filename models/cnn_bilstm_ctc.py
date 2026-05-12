"""
models/cnn_bilstm_ctc.py
CNN-BiLSTM-CTC model for Odia line-level OCR.

Architecture:
  Input (H=64, W=variable, C=1)
  → 4 CNN blocks  [Conv2D → BN → ReLU → MaxPool]
       block 0: 64 filters,  pool (2,2) → H/2,  W/2
       block 1: 128 filters, pool (2,2) → H/4,  W/4
       block 2: 256 filters, pool (2,1) → H/8,  W/4   (no further W halving)
       block 3: 256 filters, pool (2,1) → H/16, W/4
  → Reshape: collapse H dim into channels  → (T=W/4, features=H/16*256)
  → 3 BiLSTM layers (256 units each direction → 512 per layer)
  → Dropout between layers
  → Dense → num_classes+1 (incl. CTC blank)
  → CTC loss computed in custom train_step
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── CTC Loss ─────────────────────────────────────────────────────────────────

def ctc_loss_fn(labels, logits, input_lengths, label_lengths):
    """Wrapper around tf.nn.ctc_loss (uses log-softmax internally)."""
    logit_length = tf.cast(input_lengths, tf.int32)
    label_length = tf.cast(label_lengths, tf.int32)
    # tf.nn.ctc_loss expects time-major logits: (T, B, C)
    logits_t = tf.transpose(logits, [1, 0, 2])
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits_t,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=True,
        blank_index=0,
    )
    return tf.reduce_mean(loss)


# ── Model Definition ──────────────────────────────────────────────────────────

def build_model(num_classes: int, cfg: dict) -> keras.Model:
    """
    Returns a Keras Model with a custom train_step that computes CTC loss.

    Inputs (dict):
        image        : (B, H, W, 1)  float32
        label        : (B, max_L)    int32   (padded with 0)
        input_length : (B,)          int32   (actual T after CNN)
        label_length : (B,)          int32   (actual label length)

    Outputs: logits (B, T, num_classes+1)
    """
    mcfg = cfg["model"]
    filters    = mcfg["cnn_filters"]       # [64, 128, 256, 256]
    kernels    = mcfg["cnn_kernels"]       # [3, 3, 3, 3]
    strides    = mcfg["pool_strides"]      # [[2,2],[2,2],[2,1],[2,1]]
    lstm_units = mcfg["lstm_units"]        # [256, 256, 256]
    lstm_drop  = mcfg["lstm_dropout"]
    inter_drop = mcfg["inter_layer_dropout"]

    # ── Inputs ──────────────────────────────────────────────
    image        = keras.Input(shape=(None, None, 1), name="image")
    label        = keras.Input(shape=(None,),         name="label",        dtype=tf.int32)
    input_length = keras.Input(shape=(),              name="input_length",  dtype=tf.int32)
    label_length = keras.Input(shape=(),              name="label_length",  dtype=tf.int32)

    # ── CNN blocks ───────────────────────────────────────────
    x = image
    for i, (f, k, s) in enumerate(zip(filters, kernels, strides)):
        x = layers.Conv2D(f, (k, k), padding="same", use_bias=False,
                          kernel_initializer="glorot_uniform",
                          name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.MaxPool2D(pool_size=s, strides=s, name=f"pool_{i}")(x)

    # ── Reshape: (B, H', W', C) → (B, W', H'*C) ─────────────
    # After 4 blocks with H-halving: H_out = 64//(2^4) = 4
    # After 2 W-halving blocks:      W_out = W//4
    shape = tf.shape(x)
    x = layers.Reshape(target_shape=(-1, x.shape[2] * x.shape[3]), name="reshape")(
        tf.reshape(x, [shape[0], shape[1], shape[2], x.shape[3]])
    )
    # Use a Lambda to do the dynamic reshape safely
    x = layers.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1],
                                 t.shape[2] * t.shape[3]]),
        name="reshape_lambda"
    )(x if False else _cnn_output_block(image, filters, kernels, strides))

    # ── BiLSTM layers ────────────────────────────────────────
    for i, units in enumerate(lstm_units):
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True,
                        dropout=lstm_drop,
                        recurrent_dropout=0.0,   # keep recurrent fast
                        kernel_initializer="glorot_uniform"),
            merge_mode="concat",
            name=f"bilstm_{i}",
        )(x)
        if i < len(lstm_units) - 1:
            x = layers.Dropout(inter_drop, name=f"drop_{i}")(x)

    # ── Output projection ─────────────────────────────────────
    # num_classes + 1 for the CTC blank (index 0)
    logits = layers.Dense(num_classes + 1, activation=None,
                          kernel_initializer="glorot_uniform",
                          name="logits")(x)

    # ── Assemble model ────────────────────────────────────────
    model = CTCModel(
        inputs=[image, label, input_length, label_length],
        outputs=logits,
        name="OdiaOCR_CNN_BiLSTM_CTC",
    )
    return model


def _cnn_output_block(image, filters, kernels, strides):
    """Standalone CNN sub-graph reused in build_model."""
    x = image
    for i, (f, k, s) in enumerate(zip(filters, kernels, strides)):
        x = layers.Conv2D(f, (k, k), padding="same", use_bias=False,
                          kernel_initializer="glorot_uniform",
                          name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.MaxPool2D(pool_size=s, strides=s, name=f"pool_{i}")(x)
    return x


# ── Cleaner functional build ──────────────────────────────────────────────────

def build_ocr_model(num_classes: int, cfg: dict) -> "CTCModel":
    """Clean, correct functional build."""
    mcfg       = cfg["model"]
    filters    = mcfg["cnn_filters"]
    kernels    = mcfg["cnn_kernels"]
    strides    = mcfg["pool_strides"]
    lstm_units = mcfg["lstm_units"]
    lstm_drop  = mcfg["lstm_dropout"]
    inter_drop = mcfg["inter_layer_dropout"]

    # Inputs
    image        = keras.Input(shape=(None, None, 1), name="image")
    label        = keras.Input(shape=(None,),         name="label",        dtype=tf.int32)
    input_length = keras.Input(shape=(),              name="input_length",  dtype=tf.int32)
    label_length = keras.Input(shape=(),              name="label_length",  dtype=tf.int32)

    # CNN
    x = image
    for i, (f, k, s) in enumerate(zip(filters, kernels, strides)):
        x = layers.Conv2D(f, (k, k), padding="same", use_bias=False,
                          kernel_initializer="glorot_uniform", name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.MaxPool2D(pool_size=s, strides=s, name=f"pool_{i}")(x)

    # Reshape (B, H', W', C) → (B, W', H'*C)
    # H_out = 64 / (2^4) = 4;  feature_dim = 4 * last_filter
    feature_dim = (cfg["image"]["target_height"] // (2 ** len(filters))) * filters[-1]
    x = layers.Permute((2, 1, 3), name="permute")(x)          # (B, W', H', C)
    x = layers.Reshape((-1, feature_dim), name="reshape")(x)  # (B, W', H'*C)

    # BiLSTM
    for i, units in enumerate(lstm_units):
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True,
                        dropout=lstm_drop,
                        kernel_initializer="glorot_uniform"),
            merge_mode="concat",
            name=f"bilstm_{i}",
        )(x)
        if i < len(lstm_units) - 1:
            x = layers.Dropout(inter_drop, name=f"drop_{i}")(x)

    # Dense output
    logits = layers.Dense(num_classes + 1, activation=None,
                          kernel_initializer="glorot_uniform",
                          name="logits")(x)

    model = CTCModel(
        inputs={"image": image, "label": label,
                "input_length": input_length, "label_length": label_length},
        outputs=logits,
        name="OdiaOCR_CNN_BiLSTM_CTC",
    )
    return model


# ── Custom Model with CTC train_step ─────────────────────────────────────────

class CTCModel(keras.Model):
    """Keras Model subclass that embeds CTC loss into train_step/test_step."""

    def train_step(self, data):
        inputs, _ = data
        image        = inputs["image"]
        label        = inputs["label"]
        input_length = inputs["input_length"]
        label_length = inputs["label_length"]

        with tf.GradientTape() as tape:
            logits = self(inputs, training=True)
            loss = ctc_loss_fn(label, logits, input_length, label_length)

        grads = tape.gradient(loss, self.trainable_variables)
        # Gradient clipping
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        inputs, _ = data
        label        = inputs["label"]
        input_length = inputs["input_length"]
        label_length = inputs["label_length"]
        logits = self(inputs, training=False)
        loss = ctc_loss_fn(label, logits, input_length, label_length)
        return {"loss": loss}

    def predict_step(self, data):
        """Returns greedy-decoded token sequences."""
        inputs, _ = data
        logits = self(inputs, training=False)
        # Greedy decode via tf.nn.ctc_greedy_decoder
        input_length = inputs["input_length"]
        logits_t = tf.transpose(logits, [1, 0, 2])
        decoded, _ = tf.nn.ctc_greedy_decoder(logits_t, tf.cast(input_length, tf.int32))
        return decoded[0]
