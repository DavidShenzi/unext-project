# UNeXt on BUSI — a reproduction study

Final project, Deep Learning (83882).

**Paper:** [UNeXt: MLP-based Rapid Medical Image Segmentation Network](https://arxiv.org/abs/2203.04967)
— Valanarasu & Patel, Johns Hopkins University, MICCAI 2022
**Original code:** https://github.com/jeya-maria-jose/UNeXt-pytorch
**This repository:** https://github.com/DavidShenzi/unext-project

**Group:** Ze'ev Gastevert · Ido Leibowitz · David Sheinenzon
**Presentation video:** `<UNLISTED YOUTUBE LINK>`

*(ID numbers are on the presentation submitted to the course, not in this public repository.)*

---

## Summary of findings

UNeXt claims near-state-of-the-art medical image segmentation at 1/72 the parameters
of TransUNet. We reproduced it on the BUSI breast-ultrasound dataset.

**The efficiency claims reproduce exactly.** 1,471,921 parameters against the paper's stated
1.47 M; 0.577 GFLOPs against 0.57; 12.5 ms CPU inference. The architecture is what the paper
describes.

**The accuracy did not.** Running the authors' own command we measured **IoU 0.5931 ± 0.0210**
against the reported **0.6695** — across six runs and four configurations, none above 0.602.

We eliminated five explanations (resolution, mixed precision, decision threshold, split
composition, and split averaging — the last at 3.6σ), and found that three other published
groups independently report baseline UNeXt numbers close to ours rather than the paper's.

**Diagnosis: severe overfitting.** Validation peaks near epoch 100 and then declines for 300+
epochs, ending with a **+0.31** train/validation gap. The paper's augmentation (90° rotation
and flips only) is too weak for 518 training images.

**Fix — two changes to the training recipe, neither adding a single parameter:**

| stage | mean IoU (3 splits) | Δ |
|---|---|---|
| Paper recipe | 0.5931 ± 0.0210 | — |
| + strong augmentation | 0.6328 ± 0.0126 | +0.040 |
| + Focal Tversky loss | **0.6461 ± 0.0146** | +0.013 |
| | | **+0.053 total** |

That recovers ~70% of the gap. By contrast the two *architectural* modifications we ported from
follow-up papers gave +0.006 (Skip-Fusion, at +11% compute) and exactly 0.000 (SE attention).

**Takeaway: for this network on this dataset, the training recipe mattered far more than the
architecture.**

Full detail: [`project_notes.md`](project_notes.md) §7 · per-run log: [`unext/EXPERIMENTS.md`](unext/EXPERIMENTS.md)

---

## Repository contents

```
├── project_notes.md            paper theory, all results, slide narrative
├── enhancement_research.md     follow-up literature review
├── presentation_outline.md     slide-by-slide outline
├── UNeXt_presentation.pptx     the deck (speaker notes in every slide)
└── unext/
    ├── archs.py                UNext, UNext_S, UNext_SE, UNext_SkipFusion
    ├── train.py                training loop
    ├── val.py                  evaluation + GFLOPs/latency benchmarks
    ├── losses.py               BCEDice, FocalTversky, BCEFocalTversky
    ├── dataset.py  metrics.py  utils.py
    ├── prepare_busi.py         raw BUSI → loader layout
    ├── prepare_isic.py         raw ISIC 2018 → loader layout
    ├── smoke_test.py           architecture checks, no data needed
    ├── compare_runs.py         results table + curves
    ├── make_figures.py         all presentation figures
    ├── make_pptx.py            builds the deck
    ├── EXPERIMENTS.md          per-run log: what was run, what it scored
    ├── models/<run>/           config.yml + log.csv per run (weights gitignored)
    └── figures/                generated figures
```

**Not in git** (see `.gitignore`): the BUSI dataset (redistribution + 460 MB),
model weights (262 MB — every result is reproducible from the committed `log.csv`
files), raw stdout logs (53 MB of progress-bar spam), and the upstream clone.

---

## Reproducing

```bash
pip install -r unext/requirements.txt        # torch not pinned; see the file
cd unext
python smoke_test.py                         # verifies the architecture, no data needed
```

Download `Dataset_BUSI_with_GT` (registration required —
[Cairo University](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) or the Kaggle mirror), then:

```bash
python prepare_busi.py --raw /path/to/Dataset_BUSI_with_GT

# baseline — the authors' recipe
python train.py --dataset busi --arch UNext --name base --epochs 400 --split_seed 41

# our best configuration
python train.py --dataset busi --arch UNext --name best --epochs 400 --split_seed 41 \
                --aug strong --loss BCEFocalTverskyLoss

python val.py --name best
python compare_runs.py
```

`prepare_busi.py` excludes the 133 `normal` (empty-mask) images, giving 437 benign + 210
malignant = **647**, matching the paper's stated count.

---

## Changes to the original code

Required to run at all on Python 3.12 / PyTorch 2.11:
- removed an unused `mmcv` import (never called; a painful Windows build)
- `timm.models.layers` → `timm.layers`; albumentations v1 → v2 API
- deleted a dead module-level `shift()` that referenced undefined names

Correctness fixes:
- **Double normalization** — `dataset.py` divided by 255 *after* `A.Normalize()` had already
  applied ImageNet statistics, squashing inputs from ±2.5 into **±0.01**. Present upstream;
  materially affects training.
- `shiftmlp.forward` reused the input channel count after `fc1`; correct only because the paper
  sets `mlp_ratio=1`.
- `val.py` hard-coded the split seed to 41, so evaluating a run trained with another seed scored
  partly **on training images**.

Additions: `--aug {paper,strong}`, `--loss BCEFocalTverskyLoss`, `--stop_after` (halt without
changing the cosine `T_max`), `--resume`, full-state checkpoints, per-image metrics alongside
upstream's aggregate convention, seeding, GFLOPs/latency benchmarks, and the BUSI/ISIC
preprocessing scripts the original repo does not ship.

**Metric note:** upstream derives Dice as `2·IoU/(IoU+1)`, which equals the *aggregate* Dice, not
the mean per-image Dice usually reported. Both are logged (`val_dice`, `val_dice_per_image`).

---

## License / attribution

Original UNeXt code © Jeya Maria Jose Valanarasu, under the licence in the
[upstream repository](https://github.com/jeya-maria-jose/UNeXt-pytorch). Our modifications are
provided for academic coursework.

BUSI: Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief, 2020. Not
redistributed here.
