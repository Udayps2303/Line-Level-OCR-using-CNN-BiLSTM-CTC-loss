"""
evaluate.py
Evaluate a trained model on the test set and print CER/WER.

Usage:
    python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.keras
"""

import argparse
import yaml

import tensorflow as tf

from data.dataset import load_charset, read_mapping, make_tf_dataset, decode_ctc_greedy
from utils.metrics import batch_cer, batch_wer


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    chars, char2idx, idx2char = load_charset(cfg["dataset"]["charset_file"])

    samples = read_mapping(
        mapping_file=cfg["dataset"]["mapping_file"],
        image_root=cfg["dataset"]["image_root"],
        char2idx=char2idx,
        max_label_len=cfg["dataset"]["max_label_len"],
    )

    import random
    random.seed(42)
    random.shuffle(samples)
    n      = len(samples)
    val_n  = int(n * cfg["dataset"]["val_split"])
    test_n = int(n * cfg["dataset"]["test_split"])
    test_samples = samples[n - test_n:]
    print(f"Evaluating on {len(test_samples)} test samples")

    ds = make_tf_dataset(test_samples, cfg, augment=False, shuffle=False)

    model = tf.keras.models.load_model(args.checkpoint, compile=False)

    preds_all, gts_all = [], []

    for inputs, _ in ds:
        logits       = model(inputs, training=False)
        input_length = inputs["input_length"]
        label        = inputs["label"]
        label_length = inputs["label_length"]

        logits_t = tf.transpose(logits, [1, 0, 2])
        logits_t = tf.cast(logits_t, tf.float32)   # 🔥 FIX
        decoded, _ = tf.nn.ctc_greedy_decoder(logits_t, tf.cast(input_length, tf.int32))
        decoded_dense = tf.sparse.to_dense(decoded[0], default_value=0).numpy()

        for i in range(decoded_dense.shape[0]):
            pred_str = decode_ctc_greedy(decoded_dense[i], idx2char)
            gt_idx   = label[i].numpy()[:label_length[i].numpy()]
            gt_str   = "".join(idx2char.get(int(j), "?") for j in gt_idx)
            preds_all.append(pred_str)
            gts_all.append(gt_str)

    print(f"\nCER : {batch_cer(preds_all, gts_all):.4f}")
    print(f"WER : {batch_wer(preds_all, gts_all):.4f}")

    print("\n── Sample predictions ───────────────────────────────────────")
    for i in range(min(10, len(preds_all))):
        print(f"GT  : {gts_all[i]}")
        print(f"PRED: {preds_all[i]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.keras")
    main(parser.parse_args())
