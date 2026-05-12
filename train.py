"""
train.py
Main training script for Odia Line-Level OCR.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/best_model.keras
"""

import argparse
import os
import random
import math

import numpy as np
import tensorflow as tf
import yaml

from data.dataset import load_charset, read_mapping, make_tf_dataset
from models.cnn_bilstm_ctc import build_ocr_model
from utils.lr_schedule import WarmupCosineDecay
from utils.metrics import batch_cer


# ── Reproducibility ───────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main(args):
    # ── Load config ──────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── GPU setup ────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    else:
        print("[WARNING] No GPU found. Training on CPU will be very slow.")

    # Mixed precision (fp16 on GPU)
    if cfg["training"].get("mixed_precision", False) and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled (float16)")

    # ── Build charset ────────────────────────────────────────
    charset_path = cfg["dataset"]["charset_file"]
    if not os.path.exists(charset_path):
        raise FileNotFoundError(
            f"Charset file not found: {charset_path}\n"
            "Run:  python utils/build_charset.py --config config.yaml"
        )
    chars, char2idx, idx2char = load_charset(charset_path)
    num_classes = len(chars)
    print(f"[INFO] Charset size: {num_classes} characters")

    # ── Load all samples ─────────────────────────────────────
    samples = read_mapping(
        mapping_file=cfg["dataset"]["mapping_file"],
        image_root=cfg["dataset"]["image_root"],
        char2idx=char2idx,
        max_label_len=cfg["dataset"]["max_label_len"],
    )
    random.shuffle(samples)

    n = len(samples)
    val_n  = int(n * cfg["dataset"]["val_split"])
    test_n = int(n * cfg["dataset"]["test_split"])
    train_n = n - val_n - test_n

    train_samples = samples[:train_n]
    val_samples   = samples[train_n:train_n + val_n]
    test_samples  = samples[train_n + val_n:]
    print(f"[INFO] Split — train: {train_n}, val: {val_n}, test: {test_n}")

    # ── Build datasets ───────────────────────────────────────
    train_ds = make_tf_dataset(train_samples, cfg, augment=True,  shuffle=True)
    val_ds   = make_tf_dataset(val_samples,   cfg, augment=False, shuffle=False)

    # ── Build model ──────────────────────────────────────────
    model = build_ocr_model(num_classes=num_classes, cfg=cfg)
    model.summary(line_length=90)

    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        model.load_weights(args.resume)

    # ── LR schedule & optimizer ──────────────────────────────
    batch_size  = cfg["training"]["batch_size"]
    epochs      = cfg["training"]["epochs"]
    steps_epoch = math.ceil(train_n / batch_size)
    total_steps = steps_epoch * epochs
    warmup_steps = steps_epoch * cfg["training"]["warmup_epochs"]
    initial_lr   = cfg["training"]["initial_lr"]

    sched_name = cfg["training"].get("lr_schedule", "cosine")
    if sched_name == "cosine":
        lr = WarmupCosineDecay(initial_lr, total_steps, warmup_steps)
    else:
        lr = initial_lr  # ReduceLROnPlateau handles it via callback

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=cfg["training"]["gradient_clip"])
    model.compile(optimizer=optimizer)

    # ── Callbacks ────────────────────────────────────────────
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg["training"]["checkpoint_dir"], "best_model.keras"),
            monitor="val_loss",
            save_best_only=cfg["training"]["save_best_only"],
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=cfg["training"]["log_dir"],
            update_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        CERCallback(val_samples[:500], idx2char, cfg),  # quick CER on 500 val samples each epoch
    ]

    if sched_name == "reduce_on_plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
            )
        )

    # ── Train ─────────────────────────────────────────────────
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ── Save final model ─────────────────────────────────────
    final_path = os.path.join(cfg["training"]["checkpoint_dir"], "final_model.keras")
    model.save(final_path)
    print(f"[INFO] Final model saved: {final_path}")
    print("[INFO] Training complete.")


# ── CER Callback ──────────────────────────────────────────────────────────────

class CERCallback(tf.keras.callbacks.Callback):
    """Computes CER on a subset of validation samples at end of each epoch."""

    def __init__(self, val_samples, idx2char, cfg):
        super().__init__()
        self.val_samples = val_samples
        self.idx2char    = idx2char
        self.cfg         = cfg

    def on_epoch_end(self, epoch, logs=None):
        from data.dataset import make_tf_dataset, decode_ctc_greedy
        ds = make_tf_dataset(self.val_samples, self.cfg, augment=False, shuffle=False)

        preds_all = []
        gts_all   = []

        for inputs, _ in ds:
            logits = self.model(inputs, training=False)
            input_length = inputs["input_length"]
            label         = inputs["label"]
            label_length  = inputs["label_length"]

            # Greedy decode
            logits_t = tf.transpose(logits, [1, 0, 2])
            logits_t = tf.cast(logits_t, tf.float32)

            decoded, _ = tf.nn.ctc_greedy_decoder(logits_t,tf.cast(input_length, tf.int32))
            decoded_sparse = tf.sparse.to_dense(decoded[0], default_value=0).numpy()

            for i in range(decoded_sparse.shape[0]):
                pred_str = decode_ctc_greedy(decoded_sparse[i], self.idx2char)
                gt_indices = label[i].numpy()[:label_length[i].numpy()]
                gt_str = "".join(self.idx2char.get(int(idx), "?") for idx in gt_indices)
                preds_all.append(pred_str)
                gts_all.append(gt_str)

        cer_val = batch_cer(preds_all, gts_all)
        print(f"\n  [CER] epoch {epoch+1}: {cer_val:.4f}  ({len(preds_all)} samples)")

        # Print 3 examples
        for i in range(min(3, len(preds_all))):
            print(f"    GT  : {gts_all[i]}")
            print(f"    PRED: {preds_all[i]}")
            print()

        if logs is not None:
            logs["val_cer"] = cer_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Odia OCR CNN-BiLSTM-CTC")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    main(parser.parse_args())
