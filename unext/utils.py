"""Small helpers. Adds seeding and parameter/FLOP counting used for the report tables."""

import argparse
import os
import random

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['true', '1', 'yes']:
        return True
    elif v.lower() in ['false', '0', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed):
    """Seed python/numpy/torch so baseline vs. modified runs are comparable.

    Note this does not force fully deterministic cuDNN kernels -- doing so would
    slow training noticeably. Run-to-run variation of a few tenths of a point in
    Dice is still expected; that is why the paper averages over 3 splits.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
