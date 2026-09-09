"""Pixel precision and recall per run, on the run's own validation split.

The appendix slide on Focal Tversky quoted recall 0.7882 -> 0.8536 and precision
0.7928 -> 0.7510. Those four numbers appear in no log file, no eval.yml, and nowhere in
EXPERIMENTS.md -- they were hardcoded into make_figures.py from a measurement whose
output no longer exists, and they describe a single split rather than the six paired
runs we now have.

This recomputes them from the checkpoints, so the figure can be built from evidence.

Both conventions are reported, because they answer different questions and can disagree:
  aggregate  -- pool every pixel in the split, then take the ratio. Large lesions
                dominate. This matches the IoU convention train.py selects on.
  per-image  -- compute per scan, then average. Every scan counts equally.

Usage:
    python pr_eval.py                     # the six aug/ftl pairs
    python pr_eval.py --runs a b --csv pr.csv
"""

import argparse
import csv
import os

import numpy as np
import torch

import boundary_eval as B
import normal_eval as N

HERE = os.path.dirname(os.path.abspath(__file__))

PAIRS = [(f'busi_split{s}_ftl{sfx}', f'busi_split{s}_aug{sfx}')
         for sfx in ('', '_s101') for s in (41, 42, 43)]


def evaluate(name, thresh=0.5):
    model, cfg = N.load_model(name)
    if model is None:
        return None
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev).eval()

    tp = fp = fn = 0
    precs, recs = [], []
    with torch.no_grad():
        for x, y, _ in B.val_loader(cfg):
            p = (torch.sigmoid(model(x.to(dev))) > thresh).cpu().numpy()[0, 0]
            g = y.numpy()[0, 0] > 0.5
            a = int(np.logical_and(p, g).sum())
            b = int(np.logical_and(p, ~g).sum())
            c = int(np.logical_and(~p, g).sum())
            tp += a
            fp += b
            fn += c
            # per-image: a scan with no predicted positives has undefined precision;
            # skip it there rather than scoring it 0, which would understate precision.
            if a + b:
                precs.append(a / (a + b))
            if a + c:
                recs.append(a / (a + c))
    return {
        'run': name,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
        'precision_per_image': float(np.mean(precs)) if precs else 0.0,
        'recall_per_image': float(np.mean(recs)) if recs else 0.0,
        'n': len(recs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--csv', default='precision_recall.csv')
    a = ap.parse_args()

    runs = a.runs or sorted({r for pair in PAIRS for r in pair})
    rows = []
    print(f'{"run":26s} {"prec":>7} {"recall":>7} {"prec/img":>9} {"rec/img":>8}')
    for r in runs:
        res = evaluate(r)
        if res is None:
            print(f'{r}: no checkpoint')
            continue
        rows.append(res)
        print(f'{res["run"]:26s} {res["precision"]:>7.4f} {res["recall"]:>7.4f} '
              f'{res["precision_per_image"]:>9.4f} {res["recall_per_image"]:>8.4f}')

    if not a.runs and rows:
        look = {r['run']: r for r in rows}
        print(f'\n{"pair":26s} {"d recall":>9} {"d precision":>12}')
        dr, dp = [], []
        for ftl, aug in PAIRS:
            if ftl in look and aug in look:
                x = look[ftl]['recall'] - look[aug]['recall']
                y = look[ftl]['precision'] - look[aug]['precision']
                dr.append(x)
                dp.append(y)
                print(f'{ftl.replace("busi_", ""):26s} {x:>+9.4f} {y:>+12.4f}')
        if dr:
            print(f'\nmean over {len(dr)} pairs:  recall {np.mean(dr):+.4f}, '
                  f'precision {np.mean(dp):+.4f}')

    if rows:
        with open(os.path.join(HERE, a.csv), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {a.csv}')


if __name__ == '__main__':
    main()
