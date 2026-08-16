"""Segmentation metrics.

NOTE on a discrepancy with upstream. Upstream `iou_score` derives Dice from IoU
algebraically as `2*IoU/(IoU+1)`. That identity only holds for a single set, so
applied to a whole batch it is *not* the mean per-image Dice that the segmentation
literature reports -- it is a monotone transform of the batch-aggregate IoU.

`iou_score` below reproduces upstream exactly, so baseline numbers stay comparable
to the paper's reported table. `iou_dice_per_image` computes the per-image mean of
both metrics, which is the more standard reading. Both are logged; report which one
you use in the write-up.
"""

import numpy as np
import torch

SMOOTH = 1e-5


def iou_score(output, target):
    """Upstream metric: batch-aggregate IoU, with Dice derived as 2*IoU/(IoU+1)."""
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice = (2 * iou) / (iou + 1)
    return iou, dice


def iou_dice_per_image(output, target):
    """Mean per-image IoU and true mean per-image Dice (2|A∩B| / (|A|+|B|))."""
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    n = output_.shape[0]
    o = output_.reshape(n, -1)
    t = target_.reshape(n, -1)

    intersection = (o & t).sum(axis=1)
    union = (o | t).sum(axis=1)
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice = (2.0 * intersection + SMOOTH) / (o.sum(axis=1) + t.sum(axis=1) + SMOOTH)
    return float(np.mean(iou)), float(np.mean(dice))


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
