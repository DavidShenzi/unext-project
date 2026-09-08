"""Threshold sweep: is the wavelet model better behaved, or just less confident?

The false-positive advantage in normals.csv is measured at a fixed 0.5 threshold, which
leaves an obvious objection unanswered. A model whose sigmoid outputs simply sit lower
would produce fewer positive pixels on *everything* -- fewer false alarms on normal scans,
but also worse lesion coverage. That is a calibration difference, not an architectural
one, and it would be undone by anyone who moved the threshold.

The test is to sweep the threshold and evaluate both sides at once:

  * false-positive burden on the 133 held-out normal scans (no lesion, empty mask)
  * lesion IoU on the run's own validation split, at the same threshold

Plotted against each other this is an operating curve. If the wavelet model merely
trades sensitivity for specificity, its curve lies on top of the baseline's -- the same
tradeoff reached by a different default. If it dominates -- less hallucination at
matched lesion IoU -- the advantage is real and survives any threshold choice.

Usage:
    python threshold_sweep.py                      # the four wave/aug pairs
    python threshold_sweep.py --runs a b --csv out.csv
"""

import argparse
import csv
import os

import numpy as np
import torch

import boundary_eval as B
import normal_eval as N

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# The pairs the finding rests on: each modification run against the baseline trained on
# the same split with the same seed.
PAIRS = [('busi_split41_wave', 'busi_split41_aug'),
         ('busi_split42_wave', 'busi_split42_aug'),
         ('busi_split43_wave', 'busi_split43_aug'),
         ('busi_split41_wave_s101', 'busi_split41_aug_s101')]


def sweep(name, normals, device):
    """Per-threshold false-positive burden (normals) and lesion IoU (own val split)."""
    model, cfg = N.load_model(name)
    if model is None:
        return None
    model = model.to(device).eval()

    # --- probabilities on the 133 normal scans; every positive pixel is a false alarm ---
    normal_probs = []
    with torch.no_grad():
        for _, arr in normals:
            x = torch.from_numpy(arr).unsqueeze(0).to(device)
            normal_probs.append(torch.sigmoid(model(x)).cpu().numpy()[0, 0])

    # --- probabilities and ground truth on this run's own validation split ---
    val_probs, val_gts = [], []
    with torch.no_grad():
        for x, y, _ in B.val_loader(cfg):
            val_probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy()[0, 0])
            val_gts.append(y.numpy()[0, 0] > 0.5)

    rows = []
    for t in THRESHOLDS:
        fp = np.array([(p > t).sum() for p in normal_probs])
        # Aggregate IoU, matching the convention train.py selects checkpoints on.
        inter = union = 0
        for p, g in zip(val_probs, val_gts):
            pr = p > t
            inter += np.logical_and(pr, g).sum()
            union += np.logical_or(pr, g).sum()
        rows.append({
            'run': name, 'arch': cfg['arch'], 'thresh': t,
            'clean_rate': float((fp == 0).mean()),
            'mean_fp': float(fp.mean()),
            'val_iou': float(inter / union) if union else 0.0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--csv', default='threshold_sweep.csv')
    a = ap.parse_args()

    normals = N.load_normals()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runs = a.runs or sorted({r for pair in PAIRS for r in pair})

    allrows = []
    for r in runs:
        rows = sweep(r, normals, device)
        if rows is None:
            print(f'{r}: no checkpoint')
            continue
        allrows += rows
        print(f'{r}:')
        print(f'    {"thr":>5} {"cleanRate":>10} {"meanFP":>9} {"valIoU":>8}')
        for x in rows:
            print(f'    {x["thresh"]:>5.1f} {x["clean_rate"]:>10.3f} '
                  f'{x["mean_fp"]:>9.0f} {x["val_iou"]:>8.4f}')

    if allrows:
        with open(os.path.join(HERE, a.csv), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0]))
            w.writeheader()
            w.writerows(allrows)
        print(f'\nwrote {a.csv}')


if __name__ == '__main__':
    main()
