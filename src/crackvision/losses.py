import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, name="hybrid", smooth=1.0, focal_gamma=2.0):
        super().__init__()
        self.name, self.smooth, self.gamma = name, smooth, focal_gamma
        if name not in ("hybrid", "bce", "dice", "focal"):
            raise ValueError(name)

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = logits.sigmoid().flatten(1)
        t = target.flatten(1)
        dice = (
            1 - (2 * (p * t).sum(1) + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        ).mean()
        if self.name == "bce":
            return bce
        if self.name == "dice":
            return dice
        if self.name == "focal":
            pt = torch.where(target.bool(), logits.sigmoid(), 1 - logits.sigmoid())
            return (
                (1 - pt) ** self.gamma
                * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            ).mean()
        return 0.5 * bce + 0.5 * dice
