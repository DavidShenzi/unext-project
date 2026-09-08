"""Generate presentation figures. White background, black text, minimal styling.

Outputs to figures/:
  fig1_overfitting.png   train vs val IoU -- the diagnosis
  fig2_reproduction.png  our runs vs the paper's reported number
  fig3_augmentation.png  baseline vs augmented curves + per-split bars
  fig4_qualitative.png   prediction vs ground truth overlaid on the scan
  fig5_efficiency.png    params vs CPU latency, UNeXt vs TransUNet
  fig6_cumulative.png    both training-recipe changes stacked, per split
  fig7_precision_recall.png  what Focal Tversky traded

Every figure except fig4 builds from the committed models/*/log.csv alone.
fig4 additionally needs the BUSI dataset and a trained checkpoint (neither is in
the repository); it is skipped with a message if they are absent.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'text.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.color': 'gray', 'figure.dpi': 150,
})

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
PAPER_IOU, PAPER_DICE = 0.6695, 0.7937

# One colour per experimental condition, used identically in every figure, so the
# audience learns the mapping once: blue = the paper's recipe / baseline, orange =
# our augmentation fix, green = + Focal Tversky (our best), red = the paper's
# claimed number we are chasing. Chosen to stay distinguishable in greyscale print
# and under the common red-green colour-vision deficiencies.
C_BASE = '#1f77b4'     # paper recipe / baseline
C_AUG = '#ff7f0e'      # + strong augmentation
C_FTL = '#2ca02c'      # + Focal Tversky (best)
C_PAPER = '#d62728'    # the paper's reported target
C_TRAIN = '#7f7f7f'    # training curves (context, not a condition)
C_ALT = '#9467bd'      # other variants (SE, 512px, ...)
C_WAVE = '#17becf'     # + wavelet token mixer
C_BND = '#8c564b'      # + boundary-gated skip


def log(name):
    return pd.read_csv(f'models/{name}/log.csv')


def fig1_overfitting():
    """The single most important visual: train and val separating after ~ep100."""
    d = log('busi_split41')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d.epoch, d.iou, color=C_TRAIN, lw=1.8, label='Train IoU')
    ax.plot(d.epoch, d.val_iou, color=C_BASE, lw=1.8, label='Validation IoU')

    pk = int(d.val_iou.idxmax())
    # shade everything after the peak: this is the wasted portion of training
    ax.axvspan(pk, len(d) - 1, color=C_PAPER, alpha=0.06, zorder=0)
    ax.axvline(pk, color=C_BASE, lw=1.0, ls=':', alpha=0.8)
    ax.annotate(f'validation peaks\nat epoch {pk}', xy=(pk, d.val_iou.max()),
                xytext=(pk - 100, d.val_iou.max() + 0.12), fontsize=10, color=C_BASE,
                arrowprops=dict(arrowstyle='->', color=C_BASE, lw=1.0))
    # sits in the empty band between the two curves, clear of the legend
    ax.text((pk + len(d)) / 2, 0.70, 'validation flat — the model is\nmemorising the training set',
            fontsize=9.5, color=C_PAPER, ha='center', va='center', style='italic')

    last = len(d) - 1
    ax.annotate('', xy=(last, d.iou.iloc[-1]), xytext=(last, d.val_iou.iloc[-1]),
                arrowprops=dict(arrowstyle='<->', color=C_PAPER, lw=1.6))
    ax.text(last - 6, (d.iou.iloc[-1] + d.val_iou.iloc[-1]) / 2,
            f'gap\n+{d.iou.iloc[-1] - d.val_iou.iloc[-1]:.2f}',
            fontsize=10.5, ha='right', va='center', color=C_PAPER, fontweight='bold')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Baseline UNeXt overfits: validation peaks early, then declines',
                 fontsize=12, pad=12)
    ax.legend(frameon=False, loc='lower right')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_overfitting.png')
    plt.close(fig)


def fig2_reproduction():
    """Our six baseline runs against the paper's reported IoU."""
    runs = [('256px\n400ep', 0.5975), ('512px 500ep\n(authors\' cmd)', 0.5956),
            ('SE variant', 0.5953), ('seed 41', 0.6016),
            ('seed 42', 0.5692), ('seed 43', 0.6085)]
    names = [r[0] for r in runs]
    vals = [r[1] for r in runs]

    # purple = configuration variants, blue = the three-seed protocol the paper uses
    cols = [C_ALT, C_ALT, C_ALT, C_BASE, C_BASE, C_BASE]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(vals)), vals, color=cols, edgecolor='black', lw=0.9, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=9.5)

    # band spanning every run we measured -- shows they all sit far below the target
    ax.axhspan(min(vals), max(vals), color=C_BASE, alpha=0.10, zorder=0)
    ax.text(-0.62, (min(vals) + max(vals)) / 2 - 0.075, 'every run\n0.569 – 0.602',
            fontsize=9, color=C_BASE, va='center', ha='center', style='italic')

    ax.axhline(PAPER_IOU, color=C_PAPER, lw=1.8, ls='--')
    ax.text(-0.62, PAPER_IOU + 0.014, f'paper: {PAPER_IOU:.4f}',
            fontsize=10.5, ha='left', style='italic', color=C_PAPER, fontweight='bold')

    # gap arrow in the clear lane to the right of the last bar
    mean = np.mean([0.6016, 0.5692, 0.6085])
    ax.annotate('', xy=(5.62, PAPER_IOU), xytext=(5.62, mean),
                arrowprops=dict(arrowstyle='<->', color=C_PAPER, lw=1.6))
    ax.text(5.72, (PAPER_IOU + mean) / 2, f'gap\n{PAPER_IOU - mean:.3f}',
            fontsize=10, va='center', ha='left', color=C_PAPER, fontweight='bold')
    ax.set_xlim(-1.05, 6.35)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('IoU')
    ax.set_ylim(0, 0.78)
    ax.set_title('Six baseline runs, four configurations — none reach the reported number',
                 fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_reproduction.png')
    plt.close(fig)


def fig3_augmentation():
    """Left: curves showing the later peak. Right: per-split improvement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    b, a = log('busi_split43'), log('busi_split43_aug')
    ax1.plot(b.epoch, b.val_iou, color=C_BASE, lw=1.5, label='Paper augmentation')
    ax1.plot(a.epoch, a.val_iou, color=C_AUG, lw=1.7, label='Strong augmentation')
    ax1.axhline(PAPER_IOU, color=C_PAPER, lw=1.3, ls=':', alpha=0.85)
    ax1.text(8, PAPER_IOU + 0.014, 'paper', fontsize=9, style='italic', color=C_PAPER)
    pk = int(a.val_iou.idxmax())
    ax1.plot(pk, a.val_iou.max(), 'o', color=C_AUG, ms=7, zorder=4)
    ax1.annotate(f'peak @ep{pk}\n(baseline peaked ~ep100)',
                 xy=(pk, a.val_iou.max()), xytext=(140, 0.22), fontsize=9.5, color=C_AUG,
                 arrowprops=dict(arrowstyle='->', color=C_AUG, lw=1.0))
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation IoU')
    ax1.set_title('Augmentation keeps improving past ep100', fontsize=11.5)
    ax1.legend(frameon=False, loc='lower right', fontsize=9.5)
    ax1.set_ylim(0, 0.92)

    seeds = ['41', '42', '43']
    base = [0.6016, 0.5692, 0.6085]
    aug = [0.6215, 0.6306, 0.6464]
    x = np.arange(3); w = 0.34
    ax2.bar(x - w/2, base, w, color=C_BASE, edgecolor='black', lw=0.9, label='Paper aug')
    ax2.bar(x + w/2, aug, w, color=C_AUG, edgecolor='black', lw=0.9, label='Strong aug')
    for i in range(3):
        ax2.text(x[i] - w/2, base[i] + 0.010, f'{base[i]:.3f}', ha='center', fontsize=8.5)
        ax2.text(x[i] + w/2, aug[i] + 0.010, f'{aug[i]:.3f}', ha='center', fontsize=8.5)
        # per-split delta in text -- an arrow between bars this close reads as noise
        ax2.text(x[i], max(base[i], aug[i]) + 0.055, f'+{aug[i] - base[i]:.3f}',
                 ha='center', fontsize=9, color=C_AUG, fontweight='bold')
    ax2.axhline(PAPER_IOU, color=C_PAPER, lw=1.3, ls=':', alpha=0.85)
    ax2.text(-0.45, PAPER_IOU + 0.014, 'paper', fontsize=9, ha='left',
             style='italic', color=C_PAPER)
    ax2.set_xticks(x); ax2.set_xticklabels([f'seed {s}' for s in seeds])
    ax2.set_ylabel('Best IoU')
    ax2.set_ylim(0, 0.92)          # headroom so labels clear the paper line
    ax2.set_title('All three splits improve (+0.040 mean)', fontsize=11.5)
    # legend above the bars, where nothing overlaps it
    ax2.legend(frameon=False, fontsize=9.5, loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_augmentation.png')
    plt.close(fig)


def fig4_qualitative():
    """Image / ground truth / prediction for a good and a poor case."""
    import cv2
    import torch
    import albumentations as A
    import archs
    from sklearn.model_selection import train_test_split
    from glob import glob

    # This is the one figure that needs the dataset and a trained checkpoint,
    # neither of which is in the repository (see .gitignore). Skip it cleanly so
    # the remaining figures still build on a fresh clone.
    ckpt = 'models/busi_split43_aug/model.pth'
    if not os.path.exists(ckpt) or not os.path.isdir('inputs/busi/images'):
        print('    skipped: needs the BUSI dataset and a trained checkpoint'
              f' ({ckpt}); the committed figures/fig4_qualitative.png was built'
              ' from a completed run')
        return

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = archs.UNext(1, 3, False)
    m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    m = m.to(dev).eval()

    ids = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob('inputs/busi/images/*.png'))
    _, val = train_test_split(ids, test_size=0.2, random_state=43)
    tf = A.Compose([A.Resize(256, 256), A.Normalize()])

    scored = []
    for i in val:
        img = cv2.imread(f'inputs/busi/images/{i}.png')
        msk = cv2.imread(f'inputs/busi/masks/0/{i}.png', 0)
        x = tf(image=img, mask=msk[..., None])
        t = torch.tensor(x['image'].transpose(2, 0, 1)[None], dtype=torch.float32, device=dev)
        with torch.no_grad():
            p = (torch.sigmoid(m(t))[0, 0].cpu().numpy() > 0.5)
        g = cv2.resize(msk, (256, 256), interpolation=cv2.INTER_NEAREST) > 127
        inter = (p & g).sum()
        dice = 2 * inter / (p.sum() + g.sum() + 1e-7)
        scored.append((dice, i, cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB), g, p))
    scored.sort(reverse=True)
    picks = [scored[0], scored[len(scored) // 2], scored[-1]]

    from matplotlib.patches import Patch

    def overlay(img, gt, pr):
        """Colour-code agreement per pixel on top of the scan.

        green  = correctly found tumour (true positive)
        red    = missed tumour (false negative) -- the dangerous error
        yellow = false alarm (false positive)
        """
        o = img.astype(float).copy()
        tint = {(True, True): (60, 220, 60), (True, False): (235, 45, 45),
                (False, True): (250, 210, 40)}
        for (in_gt, in_pr), rgb in tint.items():
            m = (gt == in_gt) & (pr == in_pr)
            if in_gt or in_pr:                       # skip true-negative background
                o[m] = 0.45 * o[m] + 0.55 * np.array(rgb)
        return o.astype('uint8')

    fig, axes = plt.subplots(3, 3, figsize=(7.9, 8.2))
    for r, (dice, iid, img, gt, pr) in enumerate(picks):
        panels = [(img, 'Ultrasound', None),
                  (gt, 'Ground truth', 'gray'),
                  (overlay(img, gt, pr), 'Prediction vs truth', None)]
        for c, (im, ttl, cm) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im, cmap=cm)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color('black')
            if r == 0:
                ax.set_title(ttl, fontsize=11)
        col = C_FTL if dice > 0.8 else (C_AUG if dice > 0.3 else C_PAPER)
        axes[r, 0].set_ylabel(f'Dice {dice:.2f}', fontsize=10.5,
                              color=col, fontweight='bold')
    fig.suptitle('Best, median and worst cases (held-out split)', fontsize=12, y=0.985)
    fig.legend(handles=[Patch(facecolor='#3cdc3c', edgecolor='black', label='found (TP)'),
                        Patch(facecolor='#eb2d2d', edgecolor='black', label='missed (FN)'),
                        Patch(facecolor='#fad228', edgecolor='black', label='false alarm (FP)')],
               loc='lower center', ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(f'{OUT}/fig4_qualitative.png')
    plt.close(fig)


def fig6_cumulative():
    """The headline result: two training-recipe changes, stacked, per split."""
    seeds = ['41', '42', '43']
    base = [0.6016, 0.5692, 0.6085]      # paper augmentation
    aug = [0.6215, 0.6306, 0.6464]       # + strong augmentation
    ftl = [0.6397, 0.6359, 0.6629]       # + Focal Tversky

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={'width_ratios': [1.35, 1]})

    x = np.arange(3); w = 0.26
    ax1.bar(x - w, base, w, color=C_BASE, edgecolor='black', lw=0.9, label='Paper recipe')
    ax1.bar(x, aug, w, color=C_AUG, edgecolor='black', lw=0.9, label='+ strong augmentation')
    ax1.bar(x + w, ftl, w, color=C_FTL, edgecolor='black', lw=0.9, label='+ Focal Tversky')
    # value labels on a common line above the paper marker, so none collide with it
    lbl_y = PAPER_IOU + 0.022
    for i in range(3):
        for off, v in [(-w, base[i]), (0, aug[i]), (w, ftl[i])]:
            ax1.vlines(x[i] + off, v, lbl_y - 0.006, color='0.7', lw=0.7, zorder=1)
            ax1.text(x[i] + off, lbl_y, f'{v:.3f}', ha='center', fontsize=7.8,
                     rotation=90, va='bottom')
    ax1.axhline(PAPER_IOU, color=C_PAPER, lw=1.3, ls=':', alpha=0.85)
    # label the line on the left, below it, where no bar reaches
    ax1.text(-0.42, PAPER_IOU - 0.038, 'paper 0.6695', fontsize=9,
             style='italic', color=C_PAPER)
    ax1.set_xticks(x); ax1.set_xticklabels([f'seed {s}' for s in seeds])
    ax1.set_ylabel('Best IoU'); ax1.set_ylim(0, 1.02)   # room for rotated labels
    ax1.set_title('Per split — every seed improves at each stage', fontsize=11.5)
    ax1.legend(frameon=False, fontsize=9, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.005))

    means = [np.mean(base), np.mean(aug), np.mean(ftl)]
    errs = [np.std(base, ddof=1), np.std(aug, ddof=1), np.std(ftl, ddof=1)]
    labels = ['Paper\nrecipe', '+ strong\naug', '+ Focal\nTversky']
    ax2.errorbar(range(3), means, yerr=errs, color='0.35', lw=1.8, marker='none',
                 capsize=6, zorder=3)
    # colour each point to match its condition in the left panel
    for i, (m, col) in enumerate(zip(means, [C_BASE, C_AUG, C_FTL])):
        ax2.plot(i, m, 'o', color=col, ms=11, mec='black', mew=1.0, zorder=4)
    for i, (m, e) in enumerate(zip(means, errs)):
        # last point sits near the paper line -- put its label below instead
        below = (i == len(means) - 1)
        ax2.text(i, m - e - 0.016 if below else m + e + 0.011, f'{m:.4f}',
                 ha='center', va='top' if below else 'bottom',
                 fontsize=10, fontweight='bold')
    ax2.axhline(PAPER_IOU, color=C_PAPER, lw=1.3, ls=':', alpha=0.85)
    ax2.text(-0.35, PAPER_IOU + 0.002, 'paper', fontsize=9, ha='left',
             style='italic', color=C_PAPER)
    ax2.set_ylim(0.560, 0.685)
    ax2.annotate('', xy=(2.2, means[0]), xytext=(2.2, means[2]),
                 arrowprops=dict(arrowstyle='<->', color=C_FTL, lw=1.6))
    ax2.text(2.28, (means[0] + means[2]) / 2, f'+{means[2]-means[0]:.3f}',
             fontsize=10, va='center', fontweight='bold', color=C_FTL)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_xlim(-0.45, 2.65)
    ax2.set_ylabel('Mean IoU over 3 splits')
    ax2.set_title('Cumulative — neither change adds\nany parameter or FLOP', fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_cumulative.png')
    plt.close(fig)


def fig7_precision_recall():
    """What Focal Tversky actually traded: precision for recall."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    metrics = ['Recall\n(tumour pixels found)', 'Precision\n(predictions correct)']
    bce = [0.7882, 0.7928]
    ftl = [0.8536, 0.7510]
    x = np.arange(2); w = 0.32
    ax.bar(x - w/2, bce, w, color=C_AUG, edgecolor='black', lw=0.9, label='BCE + Dice')
    ax.bar(x + w/2, ftl, w, color=C_FTL, edgecolor='black', lw=0.9,
           label='BCE + Focal Tversky')
    for i in range(2):
        ax.text(x[i] - w/2, bce[i] + 0.012, f'{bce[i]:.3f}', ha='center', fontsize=10)
        ax.text(x[i] + w/2, ftl[i] + 0.012, f'{ftl[i]:.3f}', ha='center', fontsize=10)
    # green where we gained, red where we paid -- the trade is the whole story
    ax.text(0, 0.945, f'+{ftl[0]-bce[0]:.3f}', fontsize=13, ha='center',
            fontweight='bold', color=C_FTL)
    ax.text(1, 0.945, f'{ftl[1]-bce[1]:.3f}', fontsize=13, ha='center',
            fontweight='bold', color=C_PAPER)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.10)
    ax.set_title('Focal Tversky trades precision for recall — by design',
                 fontsize=12, pad=12)
    # legend above the bars; the plot area below is fully occupied
    ax.legend(frameon=False, fontsize=10, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig7_precision_recall.png')
    plt.close(fig)


def fig5_efficiency():
    """Params vs CPU latency — the claim that DID reproduce."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    pts = [('TransUNet', 105.32, 246, C_PAPER), ('UNeXt (paper)', 1.47, 25, C_BASE),
           ('UNeXt (ours)', 1.47, 12.5, C_FTL)]
    # arrow across the two orders of magnitude UNeXt buys you
    ax.annotate('', xy=(3.6, 47), xytext=(80, 230),
                arrowprops=dict(arrowstyle='->', color='0.55', lw=1.6, ls='--'))
    ax.text(15, 115, '72× fewer parameters\n10× faster', fontsize=10,
            color='0.35', ha='center', style='italic')
    for name, p, ms, col in pts:
        ax.scatter(p, ms, s=170, color=col, edgecolor='black', lw=1.2, zorder=3)
        ax.annotate(f'{name}\n{p:.2f}M, {ms:.0f}ms', (p, ms),
                    textcoords='offset points', xytext=(12, 6),
                    fontsize=9.5, color=col, fontweight='bold')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Parameters (millions, log scale)')
    ax.set_ylabel('CPU inference (ms, log scale)')
    ax.set_title('Efficiency reproduces exactly: 72× fewer parameters', fontsize=12, pad=12)
    ax.set_xlim(0.8, 260); ax.set_ylim(7, 500)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_efficiency.png')
    plt.close(fig)


def fig8_modifications():
    """Every architectural modification we tried, against what each one cost.

    The point of the figure is that the two axes disagree: the modification that helps
    most is not the one that costs most. Plotting delta-IoU against delta-GFLOPs puts
    "free" improvements on the left and expensive ones on the right, so a reader can see
    at a glance which changes were worth their compute.
    """
    import csv as _csv

    def best(run, min_epochs=None):
        """Best val IoU, but only for runs that actually finished.

        A run still in progress would otherwise be plotted against a completed baseline
        and read as a large regression, when it has simply not trained yet. `min_epochs`
        makes that exclusion explicit rather than leaving it to whoever reads the chart.
        """
        p = f'models/{run}/log.csv'
        if not os.path.exists(p):
            return None
        v = [float(r['val_iou']) for r in _csv.DictReader(open(p)) if r.get('val_iou')]
        if not v or (min_epochs and len(v) < min_epochs):
            return None
        return max(v) if v else None

    def paired(mod_pat, base_pat, min_epochs=None):
        """Mean over splits of (modified - baseline), pairing within each split."""
        ds = []
        for s in (41, 42, 43):
            m = best(mod_pat.format(s=s), min_epochs)
            b = best(base_pat.format(s=s))
            if m is not None and b is not None:
                ds.append(m - b)
        return (sum(ds) / len(ds), len(ds)) if ds else (None, 0)

    # (label, mean delta IoU, delta GFLOPs, colour). SE and Skip-Fusion are the two
    # modifications from the first round, measured on split 43 only.
    se = best('busi_split43_SEft')
    skip = best('busi_split43_SkipFt')
    b43 = best('busi_split43_aug')
    rows = []
    if se is not None and b43 is not None:
        rows.append(('SE attention', se - b43, 0.000, C_ALT, 1))
    if skip is not None and b43 is not None:
        rows.append(('Skip-Fusion', skip - b43, 0.641 - 0.577, C_BASE, 1))

    ftl, n_ftl = paired('busi_split{s}_ftl', 'busi_split{s}_aug')
    if ftl is not None:
        rows.append(('Focal Tversky', ftl, 0.000, C_FTL, n_ftl))
    wave, n_w = paired('busi_split{s}_wave', 'busi_split{s}_aug', min_epochs=400)
    if wave is not None:
        rows.append(('Wavelet mixer', wave, 0.525 - 0.577, C_WAVE, n_w))
    bnd, n_b = paired('busi_split{s}_bnd', 'busi_split{s}_aug', min_epochs=100)
    if bnd is not None:
        rows.append(('Boundary gate', bnd, 0.672 - 0.577, C_BND, n_b))

    if not rows:
        print('    skipped: no modification runs found yet')
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(0, color='black', lw=1, zorder=1)
    ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=1)

    for label, d_iou, d_flops, colour, n in rows:
        ax.scatter(d_flops, d_iou, s=190, color=colour, zorder=3,
                   edgecolor='black', linewidth=0.8)
        # Offset labels away from the axes so they never sit on the zero lines.
        # Offsets are in POINTS (textcoords='offset points'); using data units here
        # silently puts the text on top of its own marker.
        va = 'bottom' if d_iou >= 0 else 'top'
        dy = 14 if d_iou >= 0 else -14
        # Points close together in x collide when both labels are centred; nudge
        # each side outward instead. Skip-Fusion and the boundary gate sit within
        # 0.03 GFLOPs of each other and overlapped at the default centring.
        near = [o for o in rows if o is not (label, d_iou, d_flops, colour, n)
                and abs(o[2] - d_flops) < 0.045 and abs(o[1] - d_iou) < 0.004]
        ha = 'center'
        if near:
            ha = 'right' if d_flops <= near[0][2] else 'left'
        ax.annotate(f'{label}\n{d_iou:+.4f} ({n} split{"s" if n > 1 else ""})',
                    (d_flops, d_iou), xytext=(0, dy), textcoords='offset points',
                    ha=ha, va=va, fontsize=9.5, color=colour, weight='bold',
                    annotation_clip=False)

    ax.set_xlabel('change in GFLOPs vs baseline  (left = cheaper)')
    ax.set_ylabel('change in validation IoU')
    ax.set_title('What each modification bought, and what it cost', fontsize=13)

    # Pad the limits first so annotations have room, then shade. The claim is "no more
    # expensive and better", so the region includes x == 0 (free) as well as x < 0.
    xl, yl = ax.get_xlim(), ax.get_ylim()
    # Generous x-padding: labels are nudged left/right to avoid collisions, so they
    # need room beyond the outermost points or they clip at the axes.
    px, py = (xl[1] - xl[0]) * 0.30, (yl[1] - yl[0]) * 0.18
    ax.set_xlim(xl[0] - px, xl[1] + px)
    ax.set_ylim(yl[0] - py, yl[1] + py)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.add_patch(plt.Rectangle((xl[0], 0), -xl[0], yl[1], color='green', alpha=0.055,
                               zorder=0, linewidth=0))
    ax.text(xl[0] * 0.5, yl[1] * 0.95, 'no dearer, and better',
            ha='center', va='top', fontsize=9, style='italic', color='green', alpha=0.85)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig8_modifications.png')
    plt.close(fig)


if __name__ == '__main__':
    for fn in (fig1_overfitting, fig2_reproduction, fig3_augmentation,
               fig4_qualitative, fig5_efficiency, fig6_cumulative,
               fig7_precision_recall, fig8_modifications):
        fn()
        print(f'  {fn.__name__} ok')
    print(f'\nwrote {len(os.listdir(OUT))} figures to {OUT}/')
