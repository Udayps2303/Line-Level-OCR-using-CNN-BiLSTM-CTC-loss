"""
utils/lr_schedule.py
Learning rate schedules.
"""
import math
import tensorflow as tf


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then cosine annealing."""

    def __init__(self, initial_lr: float, total_steps: int, warmup_steps: int):
        super().__init__()
        self.initial_lr   = initial_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.warmup_steps
        total  = self.total_steps
        lr0    = self.initial_lr

        # Linear warmup
        warmup_lr = lr0 * step / tf.cast(warmup, tf.float32)

        # Cosine decay after warmup
        progress   = (step - warmup) / tf.cast(total - warmup, tf.float32)
        cosine_lr  = 0.5 * lr0 * (1.0 + tf.cos(math.pi * progress))

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr":   self.initial_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
        }
