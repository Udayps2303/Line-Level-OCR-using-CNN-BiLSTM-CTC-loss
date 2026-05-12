"""
data/dataset.py
TensorFlow dataset pipeline for Odia line-level OCR.
"""

import os
import random
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


# ── Charset helpers ──────────────────────────────────────────

def load_charset(path: str):
    with open(path, encoding="utf-8") as f:
        chars = [line.rstrip("\n") for line in f if line.rstrip("\n")]
    char2idx = {ch: i + 1 for i, ch in enumerate(chars)}  # 0 = CTC blank
    idx2char = {i + 1: ch for i, ch in enumerate(chars)}
    idx2char[0] = ""
    return chars, char2idx, idx2char


def encode_label(text: str, char2idx: dict, max_len: int) -> Optional[np.ndarray]:
    indices = [char2idx[ch] for ch in text if ch in char2idx]
    if len(indices) == 0 or len(indices) > max_len:
        return None
    return np.array(indices, dtype=np.int32)


def decode_ctc_greedy(pred: np.ndarray, idx2char: dict) -> str:
    prev = -1
    out = []
    for idx in pred:
        if idx != prev:
            if idx != 0:
                out.append(idx2char.get(int(idx), "?"))
            prev = idx
    return "".join(out)


# ── Image preprocessing ──────────────────────────────────────

def preprocess_image(img_path, target_height, max_width, augment, aug_cfg):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    new_w = min(int(w * target_height / h), max_width)
    img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    if augment and aug_cfg.get("enabled", False):
        img = _augment(img, aug_cfg)
    img = img.astype(np.float32) / 255.0
    return img[:, :, np.newaxis]


def _augment(img, cfg):
    prob = cfg.get("prob", 0.5)
    if random.random() < prob and cfg.get("random_brightness"):
        delta = random.uniform(-cfg["random_brightness"], cfg["random_brightness"])
        img = np.clip(img.astype(np.float32) / 255.0 + delta, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    if random.random() < prob and cfg.get("random_contrast"):
        factor = 1.0 + random.uniform(-cfg["random_contrast"], cfg["random_contrast"])
        mean = img.mean()
        img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    if random.random() < prob and cfg.get("random_noise"):
        noise = np.random.normal(0, 7, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < prob and cfg.get("dilation_erosion"):
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)
    return img


# ── Mapping file reader ──────────────────────────────────────

def read_mapping(mapping_file, image_root, char2idx, max_label_len):
    samples = []
    skipped = 0
    with open(mapping_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            rel_path, label = line.split("\t", 1)
            img_path = os.path.join(image_root, rel_path)
            encoded = encode_label(label, char2idx, max_label_len)
            if encoded is None:
                skipped += 1
                continue
            samples.append((img_path, encoded))
    print(f"Loaded {len(samples)} samples ({skipped} skipped).")
    return samples


# ── tf.data pipeline ─────────────────────────────────────────

def make_tf_dataset(samples, cfg, augment, shuffle):
    img_cfg = cfg["image"]
    aug_cfg = cfg.get("augmentation", {})
    batch_size = cfg["training"]["batch_size"]
    target_h = img_cfg["target_height"]
    max_w = img_cfg["max_width"]
    max_label_len = cfg["dataset"]["max_label_len"]

    paths = [s[0] for s in samples]
    labels = [s[1].tolist() for s in samples]

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.ragged.constant(labels, dtype=tf.int32)
    )
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(50000, len(samples)), reshuffle_each_iteration=True)

    def _load_sample(img_path, label):
        img, label_out, input_len, label_len = tf.py_function(
            func=lambda p, l: _py_load(
                p.numpy().decode(), l.numpy(), target_h, max_w, augment, aug_cfg
            ),
            inp=[img_path, label],
            Tout=[tf.float32, tf.int32, tf.int32, tf.int32],
        )
        img.set_shape([target_h, None, 1])
        label_out.set_shape([None])
        input_len.set_shape([])
        label_len.set_shape([])
        return img, label_out, input_len, label_len

    ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda img, lbl, il, ll: tf.math.greater(il, ll))  # CTC constraint: T > L

    ds = ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [target_h, max_w, 1],
            [max_label_len],
            [],
            [],
        ),
        padding_values=(0.0, tf.constant(0, dtype=tf.int32), 0, 0),
        drop_remainder=False,
    )

    def _pack(img, label, input_len, label_len):
        inputs = {
            "image": img,
            "label": label,
            "input_length": input_len,
            "label_length": label_len,
        }
        dummy = tf.zeros([tf.shape(img)[0]], dtype=tf.float32)
        return inputs, dummy

    ds = ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _py_load(img_path, label, target_h, max_w, augment, aug_cfg):
    img = preprocess_image(img_path, target_h, max_w, augment, aug_cfg)
    if img is None:
        return (
            np.zeros((target_h, 1, 1), dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.int32(0),
            np.int32(0),
        )
    # W is halved by 2 maxpool layers with stride 2 along W axis
    input_len = img.shape[1] // 4
    return (
        img.astype(np.float32),
        label.astype(np.int32),
        np.int32(input_len),
        np.int32(len(label)),
    )
