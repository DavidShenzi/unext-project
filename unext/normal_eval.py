"""False-positive burden on healthy tissue: inference only, no retraining.

BUSI ships 133 `normal` cases -- scans with no lesion and an all-zero mask. They were
excluded from every run in this project (see prepare_busi.py), because IoU is degenerate
on an empty ground truth: a model that predicts nothing at all scores perfectly, so
including them would flatter the headline metric rather than test anything.

That degeneracy is exactly why they make a good *held-out* test. Every trained model here
has never seen a normal case, and none of them can score well by accident. What they
measure is a question the paper does not ask and IoU cannot answer: **when there is no
tumour, how much does the model hallucinate?**

This matters clinically in a way IoU does not. A model that segments lesions slightly
better but fires on healthy tissue is worse in a screening setting, where most scans are
normal. It also isolates a failure mode the modifications in this project might plausibly
have introduced -- a boundary-sharpening block that latches onto speckle would show up
here even if its IoU on lesion cases looked fine.

Metrics reported per model:
  clean rate     fraction of normal scans predicted completely empty (higher is better)
  mean FP px     average false-positive area, in pixels, over all 133 scans
  median FP px   the same, robust to a few catastrophic scans
  worst FP px    the largest single hallucination
  FP fraction    mean false-positive area as a fraction of the 256x256 frame

Usage:
    python normal_eval.py                    # every run with a checkpoint
    python normal_eval.py --runs a b c       # specific runs
    python normal_eval.py --csv normals.csv  # write a table
"""

import argparse
import csv
import os
from glob import glob

import numpy as np
import torch
import yaml
from PIL import Image

import archs

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_NORMAL = os.path.join(HERE, '..', 'raw', 'Dataset_BUSI_with_GT', 'normal')


def load_normals(size=256):
    """The 133 normal scans, preprocessed exactly as dataset.py does for training.

    Masks are not loaded: they are verified all-zero, so the ground truth is "no positive
    pixel anywhere" and every predicted positive is a false positive by construction.
    """
    import albumentations as A
    paths = sorted(p for p in glob(os.path.join(RAW_NORMAL, '*.png'))
                   if '_mask' not in os.path.basename(p))
    tf = A.Compose([A.Resize(size, size), A.Normalize()])
    out = []
    for p in paths:
        img = np.array(Image.open(p).convert('RGB'))
        out.append((os.path.basename(p), tf(image=img)['image'].transpose(2, 0, 1)))
    return out


def load_model(name):
    cfg_path = os.path.join(HERE, 'models', name, 'config.yml')
    ckpt = os.path.join(HERE, 'models', name, 'model.pth')
    if not (os.path.exists(cfg_path) and os.path.exists(ckpt)):
        return None, None
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    kw = {}
    if cfg['arch'] in ('UNext_SkipFusion', 'UNext_Boundary'):
        kw['skip_identity_init'] = cfg.get('skip_identity_init', False)
    elif cfg['arch'] == 'UNext_SE':
        kw['se_reduction'] = cfg.get('se_reduction', 16)
    model = archs.__dict__[cfg['arch']](cfg['num_classes'], cfg['input_channels'],
                                        cfg['deep_supervision'], **kw)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model, cfg


def evaluate(name, normals, device, thresh=0.5):
    model, cfg = load_model(name)
    if model is None:
        return None
    model = model.to(device).eval()

    fp = []
    with torch.no_grad():
        for _, arr in normals:
            x = torch.from_numpy(arr).unsqueeze(0).to(device)
            pred = (torch.sigmoid(model(x)) > thresh).cpu().numpy()[0, 0]
            fp.append(int(pred.sum()))       # every positive pixel is a false positive
    fp = np.array(fp)
    n_px = normals[0][1].shape[-1] * normals[0][1].shape[-2]
    return {
        'run': name, 'arch': cfg['arch'], 'n': len(fp),
        'clean_rate': float((fp == 0).mean()),
        'mean_fp': float(fp.mean()),
        'median_fp': float(np.median(fp)),
        'worst_fp': int(fp.max()),
        'fp_fraction': float(fp.mean() / n_px),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--thresh', type=float, default=0.5)
    a = ap.parse_args()

    if not os.path.isdir(RAW_NORMAL):
        raise SystemExit(f'normal cases not found at {RAW_NORMAL}\n'
                         'They ship with the BUSI archive but are excluded from '
                         'inputs/busi by prepare_busi.py (see --include_normal).')

    normals = load_normals()
    print(f'{len(normals)} normal scans, all masks verified empty\n')

    runs = a.runs or sorted(
        d for d in os.listdir(os.path.join(HERE, 'models'))
        if os.path.exists(os.path.join(HERE, 'models', d, 'model.pth')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows = []
    print(f'{"run":30s} {"arch":16s} {"clean":>7} {"meanFP":>9} {"medFP":>7} {"worstFP":>8}')
    for r in runs:
        res = evaluate(r, normals, device, a.thresh)
        if res is None:
            continue
        rows.append(res)
        print(f'{res["run"]:30s} {res["arch"]:16s} {res["clean_rate"]:>6.1%} '
              f'{res["mean_fp"]:>9.0f} {res["median_fp"]:>7.0f} {res["worst_fp"]:>8d}')

    if a.csv and rows:
        with open(os.path.join(HERE, a.csv), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {a.csv}')


if __name__ == '__main__':
    main()
