"""Loss functions. Unchanged from upstream apart from dropping the optional
LovaszSoftmax dependency, which the paper's configuration never uses."""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['BCEDiceLoss', 'FocalTverskyLoss', 'BCEFocalTverskyLoss']


class BCEDiceLoss(nn.Module):
    """0.5 * BCE + Dice, as used in the UNeXt paper.

    BCE gives stable per-pixel gradients early in training; Dice directly rewards
    region overlap and is insensitive to the heavy background dominance typical of
    medical lesion masks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (Abraham & Khan, ISBI 2019).

    Dice weights false positives and false negatives equally. The Tversky index makes that
    trade-off tunable:

        TI = TP / (TP + alpha*FP + beta*FN)

    with alpha + beta = 1. Setting beta > alpha penalises *missed foreground* more than
    spurious foreground, which is what small-lesion segmentation needs: on BUSI the mean
    lesion covers only 9.4% of the image (range 0.3%-56%), so a model can score well while
    systematically under-segmenting small targets. Our worst validation case fails exactly
    that way -- it responds to large hypoechoic regions and misses the small true lesion.

    The focal exponent gamma then reshapes the gradient: (1 - TI)^(1/gamma) with gamma > 1
    flattens the loss for already-easy images and concentrates learning on hard ones, the
    same motivation as focal loss for detection.

    alpha=0.3 / beta=0.7 / gamma=4/3 are the values from the original paper.
    Zero parameters, negligible compute -- it is a loss swap, not an architecture change.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=4.0 / 3.0, smooth=1e-5):
        super().__init__()
        assert abs(alpha + beta - 1.0) < 1e-6, 'alpha + beta should sum to 1'
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, input, target):
        num = target.size(0)
        p = torch.sigmoid(input).view(num, -1)
        t = target.view(num, -1)

        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)

        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        # per-image focal term, then average -- keeps small-lesion images influential
        return ((1 - ti) ** (1 / self.gamma)).mean()


class BCEFocalTverskyLoss(nn.Module):
    """0.5*BCE + FocalTversky -- mirrors the baseline's BCE+Dice structure.

    BCE supplies dense, stable per-pixel gradients early in training (region losses are
    poorly conditioned when predictions are near-empty); Focal Tversky supplies the
    imbalance-aware region term. Keeping the same 0.5 BCE weight as `BCEDiceLoss` means the
    only variable changed versus the baseline is Dice -> Focal Tversky.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=4.0 / 3.0, bce_weight=0.5):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha, beta, gamma)
        self.bce_weight = bce_weight

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        return self.bce_weight * bce + self.ft(input, target)
