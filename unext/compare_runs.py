"""Build the baseline-vs-modified comparison table and training curves for the slides.

Reads every models/<name>/{config,summary,eval}.yml + log.csv it can find and emits:
  * results_table.csv  -- one row per run, the numbers for the results slide
  * curves.png         -- val Dice and val loss per epoch, all runs overlaid

Usage:  python compare_runs.py                      # all runs
        python compare_runs.py --runs busi_UNext busi_UNext_SE
"""

import argparse
import os
from glob import glob

import pandas as pd
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='*', default=None,
                   help='run names under models/ (default: all)')
    p.add_argument('--out_csv', default='results_table.csv')
    p.add_argument('--out_png', default='figures/curves.png')
    return p.parse_args()


def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def main():
    args = parse_args()

    run_names = args.runs
    if not run_names:
        run_names = sorted(os.path.basename(os.path.dirname(p))
                           for p in glob(os.path.join('models', '*', 'config.yml')))
    if not run_names:
        raise SystemExit('no runs found under models/')

    rows = []
    curves = {}
    for name in run_names:
        d = os.path.join('models', name)
        config = load_yaml(os.path.join(d, 'config.yml'))
        summary = load_yaml(os.path.join(d, 'summary.yml'))
        evals = load_yaml(os.path.join(d, 'eval.yml'))

        log_path = os.path.join(d, 'log.csv')
        log = pd.read_csv(log_path) if os.path.exists(log_path) else None
        if log is not None and len(log):
            curves[name] = log

        row = {
            'run': name,
            'arch': config.get('arch'),
            'dataset': config.get('dataset'),
            'epochs': summary.get('epochs_run', config.get('epochs')),
            'params': evals.get('n_params', summary.get('n_params')),
            'gflops': evals.get('gflops'),
            'best_val_iou': summary.get('best_iou'),
            'best_epoch': summary.get('best_epoch'),
            'eval_iou': evals.get('iou_batch_aggregate'),
            'eval_dice_from_iou': evals.get('dice_from_iou'),
            'eval_iou_per_image': evals.get('iou_per_image'),
            'eval_dice_per_image': evals.get('dice_per_image'),
            'gpu_ms': evals.get('gpu_ms_per_image'),
            'cpu_ms': evals.get('cpu_ms_per_image'),
            'train_min': round(summary['train_seconds'] / 60, 1)
                         if 'train_seconds' in summary else None,
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(args.out_csv, index=False)
    print(table.to_string(index=False))
    print(f'\nwrote {args.out_csv}')

    if curves:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for name, log in curves.items():
                if 'val_dice' in log:
                    axes[0].plot(log['epoch'], log['val_dice'], label=name)
                if 'val_loss' in log:
                    axes[1].plot(log['epoch'], log['val_loss'], label=name)
                if 'loss' in log:
                    axes[1].plot(log['epoch'], log['loss'], '--', alpha=0.5,
                                 label=f'{name} (train)')

            axes[0].set_xlabel('epoch'); axes[0].set_ylabel('val Dice')
            axes[0].set_title('Validation Dice'); axes[0].grid(alpha=0.3); axes[0].legend()
            axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss')
            axes[1].set_title('Loss (dashed = train)'); axes[1].grid(alpha=0.3); axes[1].legend()

            fig.tight_layout()
            fig.savefig(args.out_png, dpi=150)
            print(f'wrote {args.out_png}')
        except ImportError:
            print('(matplotlib not installed - skipping curves; pip install matplotlib)')


if __name__ == '__main__':
    main()
