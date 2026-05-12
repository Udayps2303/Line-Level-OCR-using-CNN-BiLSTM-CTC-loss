"""
utils/metrics.py
CER (Character Error Rate) and WER (Word Error Rate) using edit distance.
"""
import editdistance
import numpy as np


def cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / len(gt)


def wer(pred: str, gt: str) -> float:
    p_words = pred.split()
    g_words = gt.split()
    if len(g_words) == 0:
        return 0.0 if len(p_words) == 0 else 1.0
    return editdistance.eval(p_words, g_words) / len(g_words)


def batch_cer(preds: list[str], gts: list[str]) -> float:
    total_dist = sum(editdistance.eval(p, g) for p, g in zip(preds, gts))
    total_len  = sum(len(g) for g in gts)
    return total_dist / max(total_len, 1)


def batch_wer(preds: list[str], gts: list[str]) -> float:
    total_dist = sum(editdistance.eval(p.split(), g.split()) for p, g in zip(preds, gts))
    total_len  = sum(len(g.split()) for g in gts)
    return total_dist / max(total_len, 1)
