import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
import yaml

from data.dataset import load_charset, preprocess_image, decode_ctc_greedy
from pyctcdecode import build_ctcdecoder


# -------------------------------
# 🔹 Edit Distance
# -------------------------------
def edit_distance(ref, hyp):
    dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)

    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[len(ref)][len(hyp)]


def compute_cer(gt, pred):
    if len(gt) == 0:
        return 0.0
    return edit_distance(gt, pred) / len(gt)


def compute_wer(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()

    if len(gt_words) == 0:
        return 0.0

    return edit_distance(gt_words, pred_words) / len(gt_words)


# -------------------------------
# 🔹 Load Ground Truth Mapping
# -------------------------------
def load_gt_map(path):
    gt_map = {}
    if path is None:
        return gt_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            img, text = line.strip().split("\t", 1)
            gt_map[os.path.basename(img)] = text

    return gt_map


# -------------------------------
# 🔹 Prediction
# -------------------------------
def predict_image(model, img_path, cfg, idx2char, decoder=None):
    img_cfg = cfg["image"]
    img = preprocess_image(
        img_path,
        target_height=img_cfg["target_height"],
        max_width=img_cfg["max_width"],
        augment=False,
        aug_cfg={},
    )
    if img is None:
        return "[LOAD ERROR]"

    img_batch = img[np.newaxis, ...]
    input_len = np.array([img.shape[1] // 4], dtype=np.int32)

    dummy_label = np.zeros((1, 1), dtype=np.int32)
    dummy_llen = np.ones((1,), dtype=np.int32)

    inputs = {
        "image": tf.constant(img_batch),
        "label": tf.constant(dummy_label),
        "input_length": tf.constant(input_len),
        "label_length": tf.constant(dummy_llen),
    }

    logits = model(inputs, training=False)
    logits_t = tf.transpose(logits, [1, 0, 2])
    logits_t = tf.cast(logits_t, tf.float32)

# for greedy decoder - the 6 line below
    decoded, _ = tf.nn.ctc_greedy_decoder(
        logits_t, tf.cast(input_len, tf.int32)
    )
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=0).numpy()

    return decode_ctc_greedy(decoded_dense[0], idx2char)

    
# # for bean decoder - 14 lines below
    # decode_cfg = cfg.get("decode", {})
    # strategy = decode_cfg.get("strategy", "greedy")

    # if strategy == "beam":
    #     return decoder.decode(
    #         logits_t[:, 0, :].numpy(),   # shape (T, vocab)
    #         beam_width=decode_cfg.get("beam_width", 10)
    #     )
    # else:
    #     decoded, _ = tf.nn.ctc_greedy_decoder(
    #         logits_t, tf.cast(input_len, tf.int32)
    #     )
    #     decoded_dense = tf.sparse.to_dense(decoded[0], default_value=0).numpy()
    #     return decode_ctc_greedy(decoded_dense[0], idx2char)


# -------------------------------
# 🔹 Main
# -------------------------------
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

# # for greedy decoder - the below 2 lines
    _, _, idx2char = load_charset(cfg["dataset"]["charset_file"])
    model = tf.keras.models.load_model(args.checkpoint, compile=False)
    
# # for beam decoder - the below 6 lines
    # chars, _, idx2char = load_charset(cfg["dataset"]["charset_file"])
    # model = tf.keras.models.load_model(args.checkpoint, compile=False)

    # # Build beam decoder (used only when strategy = "beam")
    # vocab = [""] + chars   # index 0 = CTC blank
    # decoder = build_ctcdecoder(labels=vocab)

    gt_map = load_gt_map(args.gt)

    output_file = open(args.output, "w", encoding="utf-8")

    if args.image:
# for greedy decoder - the below 1 lines
        pred = predict_image(model, args.image, cfg, idx2char)
# # for beam decoder - the below 1 lines
        # pred = predict_image(model, args.image, cfg, idx2char, decoder)
        fname = os.path.basename(args.image)

        gt = gt_map.get(fname, "")
        cer = compute_cer(gt, pred) if gt else -1
        wer = compute_wer(gt, pred) if gt else -1

        line = f"{fname}\t{pred}\tCER:{cer:.4f}\tWER:{wer:.4f}"
        print(line)
        output_file.write(line + "\n")

    elif args.dir:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        files = sorted(
            f for f in os.listdir(args.dir)
            if os.path.splitext(f)[1].lower() in exts
        )

        for fname in files:
            path = os.path.join(args.dir, fname)
# for greedy decoder - the below 1 lines
            pred = predict_image(model, path, cfg, idx2char)
# # for beam decoder - the below 1 lines
            # pred = predict_image(model, path, cfg, idx2char, decoder)

            gt = gt_map.get(fname, "")
            cer = compute_cer(gt, pred) if gt else -1
            wer = compute_wer(gt, pred) if gt else -1

            line = f"{fname}\t{pred}\tCER:{cer:.4f}\tWER:{wer:.4f}"
            print(line)
            output_file.write(line + "\n")

    output_file.close()
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.keras")
    parser.add_argument("--image", default=None)
    parser.add_argument("--dir", default=None)

    # 🔥 NEW ARGUMENTS
    parser.add_argument("--gt", default="output/map_sorted.txt", help="Ground truth mapping file")
    parser.add_argument("--output", default="predictions.txt")

    main(parser.parse_args())