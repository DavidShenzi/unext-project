"""Decide whether the boundary-gate question is settled, and re-plan the queue if so.

Run unattended between jobs. The rule it applies is the one agreed with the user: once
the boundary-gate comparison has reached a confident answer, stop spending GPU time on
further variants of it and put the remainder into more seeds of the established
conditions (paper baseline, strong augmentation, Focal Tversky), where extra runs
tighten numbers that actually appear in the report.

"Confident" here does not mean "significant". With three splits a paired t-test cannot
go below p ~ 0.1 however clean the data, so demanding significance would never fire. What
can be established is a bound: if the controls show the gate's effect is small relative
to the init-only noise floor, the question is answered -- the answer is "no effect worth
reporting" -- and more runs of the same kind will not change it.

Decision inputs:
  * the _cont control  (same continuation, no gate)  -> isolates the gate
  * the _bndf runs     (frozen backbone, gates only) -> isolates it a second way
  * the seed repeats   -> the noise floor the effect must clear

Usage:
    python decide_next.py            # report the decision, change nothing
    python decide_next.py --apply    # also rewrite queue_plan.txt for the runner
"""

import argparse
import csv
import os
import statistics

import analyze_mods as A

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = [41, 42, 43]


def best(run, need):
    p = os.path.join(HERE, 'models', run, 'log.csv')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        v = [float(r['val_iou']) for r in csv.DictReader(f) if r.get('val_iou')]
    return max(v) if len(v) >= need else None


def noise_floor():
    """Spread between seeds of the same configuration -- what an effect must beat.

    Uses every pair of runs that differ only in `seed`, across whatever conditions have
    repeats available.
    """
    diffs = []
    for s in SPLITS:
        for stem, need in (('wave', 400), ('bnd', 100), ('bndf', 100)):
            base = best(f'busi_split{s}_{stem}', need)
            if base is None:
                continue
            for seed in (101, 202):
                rep = best(f'busi_split{s}_{stem}_s{seed}', need)
                if rep is not None:
                    diffs.append(abs(rep - base))
    return (statistics.fmean(diffs), len(diffs)) if diffs else (None, 0)


def gate_effect():
    """The gate's effect with the continuation confound removed.

    _bnd includes 100 epochs of whole-backbone fine-tuning; _cont is that same
    continuation without a gate. The difference is what the gate contributed.
    """
    d = []
    for s in SPLITS:
        bnd = best(f'busi_split{s}_bnd', 100)
        cont = best(f'busi_split{s}_cont', 100)
        if bnd is not None and cont is not None:
            d.append(bnd - cont)
    return d


def frozen_effect():
    """The gate's effect measured the clean way: backbone frozen, gates only."""
    d = []
    for s in SPLITS:
        bndf = best(f'busi_split{s}_bndf', 100)
        parent = best(f'busi_split{s}_aug', 400)
        if bndf is not None and parent is not None:
            d.append(bndf - parent)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()

    nf, n_nf = noise_floor()
    ge, fe = gate_effect(), frozen_effect()

    print('=== boundary gate: is the question settled? ===')
    print(f'init-only noise floor: '
          + (f'{nf:.4f} (mean |diff| over {n_nf} seed pairs)' if nf else 'not measurable yet'))

    settled, why = False, []

    for label, d in (('gate vs continuation control', ge),
                     ('gate alone (frozen backbone)', fe)):
        if len(d) < 2:
            print(f'{label}: {len(d)}/3 splits -- waiting')
            why.append(f'{label} incomplete')
            continue
        m = statistics.fmean(d)
        res = A.ttest_rel(d)
        p = res[1] if res else float('nan')
        print(f'{label}: mean {m:+.4f}, p={p:.3f} (n={len(d)})')
        # Settled when the effect is small next to the noise floor: not "no difference",
        # but "any difference is below what a different random seed already does".
        if nf and len(d) == 3 and abs(m) < nf:
            settled = True
            why.append(f'{label} |{m:+.4f}| < noise floor {nf:.4f}')

    print()
    if settled:
        print('DECISION: settled -- the gate effect is within init-only noise.')
        for w in why:
            print('  ', w)
        print('  -> remaining time is better spent on more seeds of the established')
        print('     conditions (aug / ftl), which tighten reported numbers.')
    else:
        print('DECISION: not settled -- keep the boundary-gate runs.')
        for w in why:
            print('  ', w)

    if a.apply:
        plan = os.path.join(HERE, 'queue_plan.txt')
        with open(plan, 'w', encoding='utf-8') as f:
            f.write('seeds\n' if settled else 'boundary\n')
        print(f'\nwrote {plan}: {"seeds" if settled else "boundary"}')


if __name__ == '__main__':
    main()
