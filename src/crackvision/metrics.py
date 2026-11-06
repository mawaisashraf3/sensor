"""Pixel micro confusion matrix; mIoU averages background and crack IoU."""

import numpy as np


class Confusion:
    def __init__(self):
        self.counts = np.zeros(4, dtype=np.int64)  # TP FP FN TN

    def update(self, pred, target):
        p = np.asarray(pred, dtype=bool)
        t = np.asarray(target, dtype=bool)
        if p.shape != t.shape:
            raise ValueError("Prediction and label shapes differ")
        self.counts += np.array(
            [(p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()]
        )

    def compute(self):
        tp, fp, fn, tn = map(int, self.counts)

        def div(a, b):
            return a / b if b else 0.0

        ci = div(tp, tp + fp + fn)
        bi = div(tn, tn + fp + fn)
        return dict(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=div(tp, tp + fp),
            recall=div(tp, tp + fn),
            f1=div(2 * tp, 2 * tp + fp + fn),
            accuracy=div(tp + tn, tp + tn + fp + fn),
            crack_iou=ci,
            background_iou=bi,
            miou=(ci + bi) / 2,
        )
