"""Boundary-localised evaluation: does a modification actually help at the margin?

Both new modifications (the wavelet token mixer and the boundary-gated skip) are
motivated by the same claim -- that UNeXt handles blurred, infiltrative lesion margins
poorly. Whole-mask IoU cannot test that claim. A lesion is mostly interior, so a model
can improve its margins substantially and barely move IoU, or improve IoU purely by
filling interior and never touch the margin at all.

This script measures two things whole-mask IoU hides:

  boundary F1   overlap restricted to a band of `tol` pixels around the true contour,
                which is where the interesting disagreement lives (the standard
                "trimap"/BF-score protocol from segmentation benchmarks)

  Hausdorff-95  the 95th-percentile symmetric surface distance, i.e. how far the worst
                (non-outlier) part of the predicted contour strays from the truth --
                the metric clinicians care about, since a single large excursion matters
                more than diffuse small error

Both are computed per image and averaged, so one large lesion cannot dominate.

Usage:
    python boundary_eval.py --runs busi_split41_wave busi_split41_aug
    python boundary_eval.py --compare wave      # all splits, modification vs baseline
"""

import argparse
import os

import numpy as np
import torch
import yaml
from scipy import ndimage

import archs
from dataset import Dataset
from utils import str2bool  # noqa: F401  (kept for CLI parity with the other scripts)

HERE = os.path.dirname(os.path.abspath(__file__))


def contour(mask):
    """The 1-pixel inner boundary of a binary mask."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    return m & ~ndimage.binary_erosion(m, border_value=1)


def boundary_f1(gt, pr, tol=2):
    """F1 of predicted vs true contour, allowing `tol` pixels of slack.

    A prediction's contour point counts as matched if it lies within `tol` of the true
    contour, and vice versa. tol=2 at 256x256 is roughly 1% of the image width -- tight
    enough to be about the margin, loose enough not to punish sub-pixel disagreement.
    """
    gc, pc = contour(gt), contour(pr)
    if not gc.any() and not pc.any():
        return 1.0                      # both empty: trivially perfect
    if not gc.any() or not pc.any():
        return 0.0
    # Distance to the nearest contour pixel of the other mask.
    d_gt = ndimage.distance_transform_edt(~gc)
    d_pr = ndimage.distance_transform_edt(~pc)
    precision = (d_gt[pc] <= tol).mean()      # predicted contour close to truth
    recall = (d_pr[gc] <= tol).mean()         # true contour found by prediction
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hausdorff95(gt, pr):
    """95th-percentile symmetric surface distance, in pixels. NaN if either is empty."""
    gc, pc = contour(gt), contour(pr)
    if not gc.any() or not pc.any():
        return float('nan')
    d_gt = ndimage.distance_transform_edt(~gc)
    d_pr = ndimage.distance_transform_edt(~pc)
    return float(max(np.percentile(d_gt[pc], 95), np.percentile(d_pr[gc], 95)))


def load_run(name):
    with open(os.path.join(HERE, 'models', name, 'config.yml')) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    ckpt = os.path.join(HERE, 'models', name, 'model.pth')
    if not os.path.exists(ckpt):
        return None, None
    kw = {}
    if cfg['arch'] in ('UNext_SkipFusion', 'UNext_Boundary'):
        kw['skip_identity_init'] = cfg.get('skip_identity_init', False)
    elif cfg['arch'] == 'UNext_SE':
        kw['se_reduction'] = cfg.get('se_reduction', 16)
    model = archs.__dict__[cfg['arch']](cfg['num_classes'], cfg['input_channels'],
                                        cfg['deep_supervision'], **kw)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model, cfg


def val_loader(cfg):
    """The run's own validation split -- read from its config so the split matches."""
    import albumentations as A
    from glob import glob
    from sklearn.model_selection import train_test_split

    ids = [os.path.splitext(os.path.basename(p))[0]
           for p in glob(os.path.join(HERE, 'inputs', cfg['dataset'], 'images', '*'
                                      + cfg['img_ext']))]
    _, val_ids = train_test_split(ids, test_size=0.2,
                                  random_state=cfg.get('split_seed', cfg['seed']))
    tf = A.Compose([A.Resize(cfg['input_h'], cfg['input_w']), A.Normalize()])
    ds = Dataset(val_ids, os.path.join(HERE, 'inputs', cfg['dataset'], 'images'),
                 os.path.join(HERE, 'inputs', cfg['dataset'], 'masks'),
                 cfg['img_ext'], cfg['mask_ext'], cfg['num_classes'], tf)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


def evaluate(name, tol=2):
    model, cfg = load_run(name)
    if model is None:
        return None
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev).eval()

    bf, hd, ious = [], [], []
    with torch.no_grad():
        for x, y, _ in val_loader(cfg):
            p = (torch.sigmoid(model(x.to(dev))) > 0.5).cpu().numpy()[0, 0]
            g = y.numpy()[0, 0] > 0.5
            bf.append(boundary_f1(g, p, tol))
            hd.append(hausdorff95(g, p))
            inter = (g & p).sum()
            union = (g | p).sum()
            ious.append(inter / union if union else 1.0)
    hd = [h for h in hd if not np.isnan(h)]
    return {'run': name, 'n': len(bf), 'boundary_f1': float(np.mean(bf)),
            'hd95': float(np.mean(hd)) if hd else float('nan'),
            'iou_per_image': float(np.mean(ious))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--compare', default=None,
                    help='suffix of a modification, e.g. "wave" or "bnd"')
    ap.add_argument('--tol', type=int, default=2)
    a = ap.parse_args()

    if a.compare:
        pairs = [(f'busi_split{s}_{a.compare}', f'busi_split{s}_aug') for s in (41, 42, 43)]
        d_bf, d_hd = [], []
        print(f'{"split":>6}  {"boundary F1":>22}  {"HD95 (px)":>20}')
        for mod, base in pairs:
            rm, rb = evaluate(mod, a.tol), evaluate(base, a.tol)
            if not rm or not rb:
                print(f'{mod:>6}  incomplete')
                continue
            dbf = rm['boundary_f1'] - rb['boundary_f1']
            dhd = rm['hd95'] - rb['hd95']
            d_bf.append(dbf)
            d_hd.append(dhd)
            print(f'{mod.split("split")[1][:2]:>6}  '
                  f'{rb["boundary_f1"]:.4f} -> {rm["boundary_f1"]:.4f} ({dbf:+.4f})  '
                  f'{rb["hd95"]:6.2f} -> {rm["hd95"]:6.2f} ({dhd:+.2f})')
        if d_bf:
            print(f'\nmean boundary F1 delta {np.mean(d_bf):+.4f}  '
                  f'(higher is better, n={len(d_bf)})')
            print(f'mean HD95 delta        {np.mean(d_hd):+.2f} px  '
                  f'(LOWER is better)')
        return

    for r in (a.runs or []):
        res = evaluate(r, a.tol)
        if res is None:
            print(f'{r}: no checkpoint')
            continue
        print(f'{res["run"]:26s} n={res["n"]:3d}  boundary_F1={res["boundary_f1"]:.4f}  '
              f'HD95={res["hd95"]:6.2f}px  IoU/img={res["iou_per_image"]:.4f}')


if __name__ == '__main__':
    main()
