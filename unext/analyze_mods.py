"""Paired comparison of a modification against its baseline, across splits.

The headline numbers in this project are differences of ~0.005-0.013 IoU measured on a
130-image validation set, which is small enough that "B scored higher than A" is not by
itself evidence of anything. This script does the comparison the honest way: pair each
modified run with the baseline run on the *same data split*, and test the differences.

Pairing matters. The split-to-split spread (~0.02 IoU) is several times larger than the
effects being measured, so an unpaired comparison is dominated by which images happened
to land in validation. Pairing cancels that: each difference is measured within one split.

Usage:
    python analyze_mods.py                          # all known comparisons
    python analyze_mods.py --mod wave               # just one
    python analyze_mods.py --csv results_mods.csv   # also write a table
"""

import argparse
import csv
import os
import statistics
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = [41, 42, 43]

# Each modification, and the run it should be compared against on the same split.
# The baseline choice is not cosmetic -- it defines what the number means:
#   wave -> *_aug      : both trained from scratch with strong augmentation, so the
#                        difference is attributable to the token mixer.
#   bnd  -> *_aug      : bnd is grafted FROM *_aug, so *_aug is literally its starting
#                        point and the difference is what fine-tuning the gates bought.
COMPARISONS = {
    'wave': ('busi_split{s}_wave', 'busi_split{s}_aug',
             'wavelet token mixer vs strong-aug baseline'),
    'bnd': ('busi_split{s}_bnd', 'busi_split{s}_aug',
            'boundary-gated skip vs its own graft parent'),
}


# Epochs a run must have logged before its number means anything. A job still training
# would otherwise be compared against a completed baseline and read as a catastrophic
# regression -- a wavelet run at epoch 20 of 400 scored 0.2703 against a baseline's
# 0.6464, which is not a result, it is an unfinished run.
MIN_EPOCHS = {'wave': 400, 'bnd': 100, 'aug': 1, 'ftl': 1}


def _min_epochs_for(run):
    for suffix, n in MIN_EPOCHS.items():
        if run.endswith(suffix) or f'_{suffix}_s' in run:
            return n
    return 1


def best_val_iou(run, require_complete=True):
    """Best validation IoU from a run's own log.

    Returns None when the run is absent, unreadable, or has not yet reached its full
    epoch schedule -- so a partially-trained job is excluded rather than reported.
    """
    p = os.path.join(HERE, 'models', run, 'log.csv')
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            v = [float(r['val_iou']) for r in csv.DictReader(f) if r.get('val_iou')]
        if not v:
            return None
        if require_complete and len(v) < _min_epochs_for(run):
            return None
        return max(v)
    except Exception:
        return None


def seed_variants(pattern, s):
    """All runs for one split: the base run plus any _s<seed> repeats."""
    base = pattern.format(s=s)
    out = []
    if best_val_iou(base) is not None:
        out.append(base)
    for p in sorted(glob(os.path.join(HERE, 'models', base + '_s*'))):
        name = os.path.basename(p)
        if best_val_iou(name) is not None:
            out.append(name)
    return out


def ttest_rel(diffs):
    """Two-sided paired t-test on the differences. Returns (t, p, df) or None.

    Implemented here rather than pulled from scipy so the project keeps its current
    dependency set; the p-value comes from the exact Student-t CDF via a regularised
    incomplete beta, so it agrees with scipy.stats.ttest_rel to numerical precision.
    """
    n = len(diffs)
    if n < 2:
        return None
    m = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return (float('inf') if m else 0.0), (0.0 if m else 1.0), n - 1
    t = m / (sd / n ** 0.5)
    df = n - 1
    p = _t_sf(abs(t), df) * 2
    return t, p, df


def _t_sf(t, df):
    """P(T > t) for Student-t with df degrees of freedom."""
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b), via the standard continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    import math
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz's algorithm
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    r = front * (f - 1.0)
    return r if a < (a + b) * x else 1.0 - _betainc(b, a, 1 - x)


def analyze(key, rows):
    mod_pat, base_pat, label = COMPARISONS[key]
    print(f'\n=== {key}: {label} ===')

    diffs, per_split = [], []
    for s in SPLITS:
        base = best_val_iou(base_pat.format(s=s))
        mods = seed_variants(mod_pat, s)
        if base is None or not mods:
            print(f'  split {s}: incomplete '
                  f'(baseline {"ok" if base else "MISSING"}, {len(mods)} mod run(s))')
            continue
        # Average the seed repeats for this split: they estimate the same quantity, so
        # averaging first keeps the t-test's pairing (one difference per split) intact
        # rather than inflating n with correlated observations.
        vals = [best_val_iou(m) for m in mods]
        mv = statistics.fmean(vals)
        d = mv - base
        diffs.append(d)
        per_split.append((s, base, mv, d, len(vals)))
        seeds = f' (mean of {len(vals)} seeds)' if len(vals) > 1 else ''
        print(f'  split {s}: {base:.4f} -> {mv:.4f}   {d:+.4f}{seeds}')
        rows.append({'comparison': key, 'split': s, 'baseline': f'{base:.4f}',
                     'modified': f'{mv:.4f}', 'delta': f'{d:+.4f}', 'n_seeds': len(vals)})

    if len(diffs) < 2:
        print('  not enough completed splits for a test yet')
        return

    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    res = ttest_rel(diffs)
    print(f'  ---')
    print(f'  mean delta {mean:+.4f}  (sd {sd:.4f}, n={len(diffs)} splits)')
    if res:
        t, p, df = res
        verdict = 'SIGNIFICANT at 0.05' if p < 0.05 else 'not significant at 0.05'
        print(f'  paired t-test: t={t:.3f}, df={df}, p={p:.4f}  -> {verdict}')
        if p >= 0.05 and mean > 0:
            print('  (positive but unresolved -- consistent with a real small effect '
                  'that n=3 cannot separate from noise)')
        rows.append({'comparison': key, 'split': 'ALL', 'baseline': '',
                     'modified': f'mean {mean:+.4f}', 'delta': f'p={p:.4f}',
                     'n_seeds': len(diffs)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mod', default=None, choices=sorted(COMPARISONS))
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()

    rows = []
    for key in ([a.mod] if a.mod else sorted(COMPARISONS)):
        analyze(key, rows)

    if a.csv and rows:
        with open(os.path.join(HERE, a.csv), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {a.csv}')


if __name__ == '__main__':
    main()
