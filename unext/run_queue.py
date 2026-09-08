"""Unattended experiment queue for the two new modifications.

Runs a priority-ordered list of training jobs, one at a time, and keeps going when an
individual job fails so that an overnight session yields as much usable data as possible.

Priority order matters: the queue is ordered so that if it is cut short, what completed
is still a coherent, reportable result (all three splits of one condition) rather than
a scatter of unrelated partial runs.

  stage 1   UNext_Wave      x3 splits   from scratch, 400 ep  -- the novelty result
  stage 2   UNext_Boundary  x3 splits   grafted, 100 ep       -- the safety result
  stage 2b  controls        x3 splits   no-gate continuation + frozen-backbone gate
  stage 3   both            x2 extra seeds each               -- statistical power

Stage 3 exists because three splits at p~0.08 cannot resolve a +0.005 effect; the extra
seeds hold the data split fixed and vary only the weight init, which is what the paired
comparison against the corresponding baseline run actually needs.

Usage:
    python run_queue.py                # run everything not already done
    python run_queue.py --dry-run      # print the plan and exit
    python run_queue.py --only wave    # restrict to matching job names

A job is considered done when models/<name>/model.pth exists, so the queue is safely
restartable: interrupted or repaired jobs simply resume on the next invocation.
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, 'queue_logs')
STATUS = os.path.join(LOG_DIR, 'queue_status.csv')

SPLITS = [41, 42, 43]
EXTRA_SEEDS = [101, 202]        # same split, different init


def job(name, arch, epochs, seed, split_seed, **extra):
    """One training invocation, as a name plus the argv that produces it."""
    cmd = [sys.executable, 'train.py',
           '--dataset', 'busi', '--arch', arch, '--name', name,
           '--epochs', str(epochs), '--aug', 'strong',
           '--seed', str(seed), '--split_seed', str(split_seed)]
    for k, v in extra.items():
        cmd += ['--' + k, str(v)]
    return {'name': name, 'arch': arch, 'cmd': cmd}


def build_queue():
    jobs = []

    # ---- stage 1: wavelet token mixer, from scratch -------------------------------
    # The mixer replaces the shifted MLP in all four Tok-MLP blocks, so there is no
    # meaningful baseline checkpoint to graft from: it trains from scratch, like the
    # baseline it will be compared against.
    for s in SPLITS:
        jobs.append(job(f'busi_split{s}_wave', 'UNext_Wave', 400, s, s))

    # ---- stage 2: boundary gate, grafted -------------------------------------------
    # Four small gates on an otherwise untouched backbone, and identity_init reproduces
    # the parent exactly at insertion -- so fine-tuning from the matching *_aug run both
    # costs ~3x less and isolates the gate's contribution from init noise.
    for s in SPLITS:
        parent = f'models/busi_split{s}_aug/model.pth'
        jobs.append(job(f'busi_split{s}_bnd', 'UNext_Boundary', 100, s, s,
                        init_from=parent, skip_identity_init='True'))

    # ---- stage 2b: the controls that make stage 2 interpretable ---------------------
    # The _bnd runs above fine-tune the WHOLE backbone for 100 more epochs, and the
    # parent checkpoints had not converged (split 41's peaked at epoch 391 of 400). So
    # "+0.0054 vs the parent" confounds the gate with 100 epochs of ordinary training.
    #
    #   _cont : identical continuation with NO gate (plain UNext). The difference
    #           between _bnd and _cont is what the gate actually bought.
    #   _bndf : gates only, backbone frozen, lr 1e-3 -- the protocol the earlier
    #           busi_split43_SkipFt run used, so the boundary gate can be compared
    #           against Skip-Fusion on equal terms rather than across protocols.
    for s in SPLITS:
        parent = f'models/busi_split{s}_aug/model.pth'
        jobs.append(job(f'busi_split{s}_cont', 'UNext', 100, s, s, init_from=parent))
        jobs.append(job(f'busi_split{s}_bndf', 'UNext_Boundary', 100, s, s,
                        init_from=parent, skip_identity_init='True',
                        freeze_except='fuse', lr=1e-3))

    # ---- stage 3: extra seeds for statistical power ---------------------------------
    for seed in EXTRA_SEEDS:
        for s in SPLITS:
            jobs.append(job(f'busi_split{s}_wave_s{seed}', 'UNext_Wave', 400, seed, s))
        for s in SPLITS:
            parent = f'models/busi_split{s}_aug/model.pth'
            jobs.append(job(f'busi_split{s}_bnd_s{seed}', 'UNext_Boundary', 100, seed, s,
                            init_from=parent, skip_identity_init='True'))
    return jobs


def done(name):
    return os.path.exists(os.path.join(HERE, 'models', name, 'model.pth'))


def record(row):
    os.makedirs(LOG_DIR, exist_ok=True)
    new = not os.path.exists(STATUS)
    with open(STATUS, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['name', 'arch', 'status', 'best_iou', 'minutes',
                        'finished_at', 'note'])
        w.writerow(row)


def best_iou(name):
    """Read the best val IoU straight from the run's own log."""
    p = os.path.join(HERE, 'models', name, 'log.csv')
    if not os.path.exists(p):
        return ''
    try:
        with open(p, encoding='utf-8') as f:
            vals = [float(r['val_iou']) for r in csv.DictReader(f) if r.get('val_iou')]
        return f'{max(vals):.4f}' if vals else ''
    except Exception:
        return ''


def run(j, retry_ok=True):
    """Run one job. Returns (status, note). Never raises."""
    log_path = os.path.join(LOG_DIR, j['name'] + '.log')
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        try:
            p = subprocess.run(j['cmd'], cwd=HERE, stdout=log,
                               stderr=subprocess.STDOUT, text=True)
        except Exception as e:                       # pragma: no cover
            return 'error', f'launch failed: {e}'
    mins = (time.time() - t0) / 60

    if p.returncode == 0 and done(j['name']):
        return 'ok', f'{mins:.1f} min'

    tail = ''
    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            tail = f.read()[-4000:]
    except Exception:
        pass

    # --- automatic repair for the one failure mode that is both likely and fixable ---
    # An 8 GB laptop GPU can OOM if anything else claims VRAM mid-session. Halving the
    # batch changes the optimisation slightly, so it is recorded in the note rather than
    # being silently equivalent to the other runs.
    if retry_ok and ('CUDA out of memory' in tail or 'CUBLAS_STATUS_ALLOC_FAILED' in tail):
        j2 = dict(j, cmd=j['cmd'] + ['-b', '4'])
        st, note = run(j2, retry_ok=False)
        return st, f'OOM -> retried at batch 4; {note}'

    err = 'unknown failure'
    for line in reversed(tail.splitlines()):
        if line.strip() and ('Error' in line or 'error' in line or 'Exception' in line):
            err = line.strip()[:200]
            break
    return 'failed', f'rc={p.returncode}: {err}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None, help='substring filter on job name')
    a = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    jobs = build_queue()
    if a.only:
        jobs = [j for j in jobs if a.only in j['name']]

    pending = [j for j in jobs if not done(j['name'])]
    print(f'{len(jobs)} jobs, {len(jobs) - len(pending)} already complete, '
          f'{len(pending)} to run')
    for j in pending:
        print('   ', j['name'])
    if a.dry_run:
        return

    t_start = time.time()
    for i, j in enumerate(pending, 1):
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'[{stamp}] ({i}/{len(pending)}) {j["name"]} ...', flush=True)
        st, note = run(j)
        record([j['name'], j['arch'], st, best_iou(j['name']), note.split(';')[-1].strip(),
                datetime.datetime.now().isoformat(timespec='seconds'), note])
        print(f'    -> {st}  iou={best_iou(j["name"]) or "n/a"}  {note}', flush=True)

    print(f'\nqueue finished in {(time.time() - t_start) / 3600:.2f} h')
    print(f'status: {STATUS}')


if __name__ == '__main__':
    main()
