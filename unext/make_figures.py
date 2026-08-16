"""Generate presentation figures. White background, black text, minimal styling.

Outputs to figures/:
  fig1_overfitting.png   train vs val IoU -- the diagnosis
  fig2_reproduction.png  our runs vs the paper's reported number
  fig3_augmentation.png  baseline vs augmented curves + per-split bars
  fig4_qualitative.png   image / ground truth / prediction examples
  fig5_efficiency.png    params vs CPU latency, UNeXt vs TransUNet
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


def log(name):
    return pd.read_csv(f'models/{name}/log.csv')


def fig1_overfitting():
    """The single most important visual: train and val separating after ~ep100."""
    d = log('busi_split41')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d.epoch, d.iou, color='black', lw=1.6, label='Train IoU')
    ax.plot(d.epoch, d.val_iou, color='gray', lw=1.6, ls='--', label='Validation IoU')

    pk = int(d.val_iou.idxmax())
    ax.axvline(pk, color='black', lw=0.9, ls=':', alpha=0.7)
    ax.annotate(f'validation peaks\nat epoch {pk}', xy=(pk, d.val_iou.max()),
                xytext=(pk + 28, d.val_iou.max() - 0.13), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.9))

    last = len(d) - 1
    ax.annotate('', xy=(last, d.iou.iloc[-1]), xytext=(last, d.val_iou.iloc[-1]),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.3))
    ax.text(last - 6, (d.iou.iloc[-1] + d.val_iou.iloc[-1]) / 2,
            f'gap\n+{d.iou.iloc[-1] - d.val_iou.iloc[-1]:.2f}',
            fontsize=10, ha='right', va='center')

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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(vals)), vals, color='white', edgecolor='black', lw=1.3, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=9.5)

    ax.axhline(PAPER_IOU, color='black', lw=1.6, ls='--')
    ax.text(len(vals) - 0.4, PAPER_IOU + 0.009,
            f'paper: {PAPER_IOU:.4f}', fontsize=10, ha='right', style='italic')

    mean = np.mean([0.6016, 0.5692, 0.6085])
    ax.annotate('', xy=(4.5, PAPER_IOU), xytext=(4.5, mean),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.3))
    ax.text(4.62, (PAPER_IOU + mean) / 2, f'gap\n{PAPER_IOU - mean:.3f}', fontsize=10, va='center')

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
    ax1.plot(b.epoch, b.val_iou, color='gray', lw=1.3, ls='--', label='Paper augmentation')
    ax1.plot(a.epoch, a.val_iou, color='black', lw=1.5, label='Strong augmentation')
    ax1.axhline(PAPER_IOU, color='black', lw=1.1, ls=':', alpha=0.65)
    ax1.text(8, PAPER_IOU + 0.014, 'paper', fontsize=9, style='italic')
    pk = int(a.val_iou.idxmax())
    ax1.plot(pk, a.val_iou.max(), 'o', color='black', ms=6)
    ax1.annotate(f'peak @ep{pk}\n(baseline peaked ~ep100)',
                 xy=(pk, a.val_iou.max()), xytext=(140, 0.22), fontsize=9.5,
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.9))
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation IoU')
    ax1.set_title('Augmentation keeps improving past ep100', fontsize=11.5)
    ax1.legend(frameon=False, loc='lower right', fontsize=9.5)
    ax1.set_ylim(0, 0.92)

    seeds = ['41', '42', '43']
    base = [0.6016, 0.5692, 0.6085]
    aug = [0.6215, 0.6306, 0.6464]
    x = np.arange(3); w = 0.34
    ax2.bar(x - w/2, base, w, color='white', edgecolor='black', lw=1.3, label='Paper aug')
    ax2.bar(x + w/2, aug, w, color='black', edgecolor='black', lw=1.3, label='Strong aug')
    for i in range(3):
        ax2.text(x[i] - w/2, base[i] + 0.010, f'{base[i]:.3f}', ha='center', fontsize=8.5)
        ax2.text(x[i] + w/2, aug[i] + 0.010, f'{aug[i]:.3f}', ha='center', fontsize=8.5)
    ax2.axhline(PAPER_IOU, color='black', lw=1.1, ls=':', alpha=0.65)
    ax2.text(-0.45, PAPER_IOU + 0.014, 'paper', fontsize=9, ha='left', style='italic')
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

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = archs.UNext(1, 3, False)
    m.load_state_dict(torch.load('models/busi_split43_aug/model.pth', map_location='cpu'))
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

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.8))
    for r, (dice, iid, img, gt, pr) in enumerate(picks):
        for c, (im, ttl, cm) in enumerate([(img, 'Ultrasound', None),
                                           (gt, 'Ground truth', 'gray'),
                                           (pr, 'Prediction', 'gray')]):
            ax = axes[r, c]
            ax.imshow(im, cmap=cm)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color('black')
            if r == 0:
                ax.set_title(ttl, fontsize=11)
        axes[r, 0].set_ylabel(f'Dice {dice:.2f}', fontsize=10)
    fig.suptitle('Best, median and worst cases (held-out split)', fontsize=12, y=0.98)
    fig.tight_layout()
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
    ax1.bar(x - w, base, w, color='white', edgecolor='black', lw=1.3, label='Paper recipe')
    ax1.bar(x, aug, w, color='0.65', edgecolor='black', lw=1.3, label='+ strong augmentation')
    ax1.bar(x + w, ftl, w, color='black', edgecolor='black', lw=1.3, label='+ Focal Tversky')
    for i in range(3):
        for off, v in [(-w, base[i]), (0, aug[i]), (w, ftl[i])]:
            ax1.text(x[i] + off, v + 0.008, f'{v:.3f}', ha='center', fontsize=7.8)
    ax1.axhline(PAPER_IOU, color='black', lw=1.1, ls=':', alpha=0.7)
    ax1.text(-0.42, PAPER_IOU + 0.013, 'paper 0.6695', fontsize=9, style='italic')
    ax1.set_xticks(x); ax1.set_xticklabels([f'seed {s}' for s in seeds])
    ax1.set_ylabel('Best IoU'); ax1.set_ylim(0, 0.93)
    ax1.set_title('Per split — every seed improves at each stage', fontsize=11.5)
    ax1.legend(frameon=False, fontsize=9, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.005))

    means = [np.mean(base), np.mean(aug), np.mean(ftl)]
    errs = [np.std(base, ddof=1), np.std(aug, ddof=1), np.std(ftl, ddof=1)]
    labels = ['Paper\nrecipe', '+ strong\naug', '+ Focal\nTversky']
    ax2.errorbar(range(3), means, yerr=errs, color='black', lw=1.8, marker='o',
                 ms=8, capsize=6, zorder=3)
    for i, (m, e) in enumerate(zip(means, errs)):
        # last point sits near the paper line -- put its label below instead
        below = (i == len(means) - 1)
        ax2.text(i, m - e - 0.016 if below else m + e + 0.011, f'{m:.4f}',
                 ha='center', va='top' if below else 'bottom',
                 fontsize=10, fontweight='bold')
    ax2.axhline(PAPER_IOU, color='black', lw=1.1, ls=':', alpha=0.7)
    ax2.text(-0.35, PAPER_IOU + 0.002, 'paper', fontsize=9, ha='left', style='italic')
    ax2.set_ylim(0.560, 0.685)
    ax2.annotate('', xy=(2.2, means[0]), xytext=(2.2, means[2]),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.4))
    ax2.text(2.28, (means[0] + means[2]) / 2, f'+{means[2]-means[0]:.3f}',
             fontsize=10, va='center', fontweight='bold')
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
    ax.bar(x - w/2, bce, w, color='white', edgecolor='black', lw=1.4, label='BCE + Dice')
    ax.bar(x + w/2, ftl, w, color='black', edgecolor='black', lw=1.4, label='BCE + Focal Tversky')
    for i in range(2):
        ax.text(x[i] - w/2, bce[i] + 0.012, f'{bce[i]:.3f}', ha='center', fontsize=10)
        ax.text(x[i] + w/2, ftl[i] + 0.012, f'{ftl[i]:.3f}', ha='center', fontsize=10)
    ax.annotate(f'+{ftl[0]-bce[0]:.3f}', xy=(0.42, 0.90), fontsize=11, fontweight='bold')
    ax.annotate(f'{ftl[1]-bce[1]:.3f}', xy=(1.42, 0.90), fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.0)
    ax.set_title('Focal Tversky trades precision for recall — by design',
                 fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig7_precision_recall.png')
    plt.close(fig)


def fig5_efficiency():
    """Params vs CPU latency — the claim that DID reproduce."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    pts = [('TransUNet', 105.32, 246, 'gray'), ('UNeXt (paper)', 1.47, 25, 'gray'),
           ('UNeXt (ours)', 1.47, 12.5, 'black')]
    for name, p, ms, col in pts:
        ax.scatter(p, ms, s=130, color='white' if col == 'gray' else 'black',
                   edgecolor='black', lw=1.6, zorder=3)
        ax.annotate(f'{name}\n{p:.2f}M, {ms:.0f}ms', (p, ms),
                    textcoords='offset points', xytext=(12, 6), fontsize=9.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Parameters (millions, log scale)')
    ax.set_ylabel('CPU inference (ms, log scale)')
    ax.set_title('Efficiency reproduces exactly: 72× fewer parameters', fontsize=12, pad=12)
    ax.set_xlim(0.8, 260); ax.set_ylim(7, 500)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_efficiency.png')
    plt.close(fig)


if __name__ == '__main__':
    for fn in (fig1_overfitting, fig2_reproduction, fig3_augmentation,
               fig4_qualitative, fig5_efficiency, fig6_cumulative,
               fig7_precision_recall):
        fn()
        print(f'  {fn.__name__} ok')
    print(f'\nwrote {len(os.listdir(OUT))} figures to {OUT}/')
