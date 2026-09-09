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
against the reported **0.6695** — across six runs and four configurations, the best of which
reached 0.6085.

We eliminated five explanations (resolution, mixed precision, decision threshold, split
composition, and split averaging — the last at 3.6σ), and found that three other published
groups independently report baseline UNeXt numbers close to ours rather than the paper's.

**Diagnosis: severe overfitting.** Validation peaks near epoch 100 and then declines for 300+
epochs, ending with a **+0.31** train/validation gap at the final epoch (**+0.16** at the
selected checkpoint). The paper's augmentation (90° rotation and flips only) is too weak for
518 training images.

**Fix — stronger augmentation, adding not a single parameter:**

| stage | mean IoU (3 splits) | Δ |
|---|---|---|
| Paper recipe | 0.5931 ± 0.0210 | — |
| + strong augmentation | **0.6328 ± 0.0126** | **+0.040** |

At a *matched* training budget the gain is **+0.023 (p = 0.24)** — augmentation does not converge
faster, it removes the ceiling that stalls the baseline near epoch 100.

**No architectural modification cleared our own noise floor.** Rerunning identical
configurations with only the random seed changed moves IoU by **0.0107** on average (6 pairs,
range 0.003–0.019). Every modification we tried is smaller than that:

| modification | params | GFLOPs | Δ IoU | paired runs |
|---|---|---|---|---|
| SE attention | +0.55% | — | 0.000 | 1 |
| Skip-Fusion | +5.90% | +11% | +0.002 | 1 |
| Focal Tversky loss | — | — | +0.005 (p = 0.32) | 6 |
| Boundary gate | +8.84% | +16% | +0.004 (p = 0.23) | 3 |
| **Wavelet mixer** | +1.43% | **−9%** | −0.001 (p = 0.72) | 3 |

The wavelet mixer is the one worth reporting: identical accuracy for **9% fewer FLOPs**, and on
the 133 held-out `normal` scans it leaves far more of them completely clean (**14.8%** mean over
4 runs, versus **5.5%** over 15 baseline runs). Only 2 of those 4 pairs match on seed, so we
quote no p-value — but the effect is the same size in both groups (+0.113 vs +0.105).

**Takeaway: the training recipe mattered more than the architecture — and most reported gains of
this size are indistinguishable from the random seed.**

Full detail: [`project_notes.md`](project_notes.md) §7 · per-run log: [`unext/EXPERIMENTS.md`](unext/EXPERIMENTS.md)

---

## Repository contents

```
├── project_notes.md            paper theory, all results, slide narrative
├── enhancement_research.md     follow-up literature review
├── UNeXt_presentation.pptx     the deck, notes-free (speaker notes are kept locally, not committed)
└── unext/
    ├── archs.py                UNext, UNext_S, UNext_SE, UNext_SkipFusion,
    │                           UNext_Wave, UNext_Boundary
    ├── train.py                training loop
    ├── val.py                  evaluation + GFLOPs/latency benchmarks
    ├── losses.py               BCEDice, FocalTversky, BCEFocalTversky
    ├── dataset.py  metrics.py  utils.py
    ├── prepare_busi.py         raw BUSI → loader layout
    ├── prepare_isic.py         raw ISIC 2018 → loader layout
    ├── smoke_test.py           architecture checks, no data needed
    ├── compare_runs.py         results table + curves
    ├── analyze_mods.py         paired comparisons + paired t-tests
    ├── normal_eval.py          false positives on the 133 healthy scans
    ├── threshold_sweep.py      false positives vs lesion IoU across thresholds
    ├── boundary_eval.py        boundary F1 + Hausdorff-95
    ├── boundary.csv            boundary metrics per run
    ├── run_queue.py            unattended job queue
    ├── make_figures.py         all presentation figures
    ├── EXPERIMENTS.md          per-run log: what was run, what it scored
    ├── results_table.csv       every run, every metric
    ├── normals.csv             false-positive burden per run
    ├── models/<run>/           config.yml + log.csv + eval.yml (weights gitignored)
    └── figures/                generated figures
```

**Not in git** (see `.gitignore`): the BUSI dataset (redistribution + 460 MB),
model weights (262 MB — every result is reproducible from the committed `log.csv`
files), raw stdout logs (53 MB of progress-bar spam), and the upstream clone.

---

## Running everything

### 1. Install

```bash
pip install -r unext/requirements.txt        # torch not pinned; see the file
cd unext
python smoke_test.py                         # architecture checks — no data needed
```

`smoke_test.py` instantiates `UNext`, `UNext_S` and `UNext_SE`, checking output shapes, a
backward pass, and the parameter count against the paper's stated 1.47 M. If it passes, the
install works even without the dataset. (It predates the wavelet/boundary architectures.)

### 2. Get the data

Download `Dataset_BUSI_with_GT` (registration required —
[Cairo University](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) or the Kaggle mirror),
then:

```bash
python prepare_busi.py --raw /path/to/Dataset_BUSI_with_GT
```

This writes `inputs/busi/{images,masks}` and excludes the 133 `normal` (empty-mask) images,
giving 437 benign + 210 malignant = **647**, matching the paper's stated count. Keep the raw
archive: the false-positive evaluation reads the `normal` cases directly from it.

### 3. Train, evaluate, analyze

```bash
python train.py --dataset busi --arch UNext --name base --epochs 400 --split_seed 41
python val.py --name base            # -> models/base/eval.yml (IoU/Dice, GFLOPs, latency)
python compare_runs.py               # every run -> results_table.csv + curves.png
python analyze_mods.py               # paired per-split comparisons + t-tests
```

Every run writes `models/<name>/` (`config.yml`, `log.csv`, `summary.yml`, `model.pth`) and is
reproducible from its own `config.yml`. Run `python train.py --help` for the full flag list —
architectures (`--arch`), augmentation strength, loss, checkpoint grafting (`--init_from`,
`--freeze_except`), and resume/stop controls.

**Pair runs correctly.** A comparison is only clean when the two runs share **both** `--seed`
and `--split_seed` and differ in exactly one factor — the single easiest mistake to make here
(two of our four wavelet pairs violate it; see the caveat above).

### 4. Extra analyses and figures

```bash
python normal_eval.py --csv normals.csv        # false positives on the 133 healthy scans
python threshold_sweep.py                      # false positives vs lesion IoU across thresholds
python boundary_eval.py --runs base --csv boundary.csv   # boundary F1 + Hausdorff-95
python make_figures.py                          # regenerate every presentation figure
```

### Running many jobs unattended

`run_queue.py` runs a priority-ordered job list one at a time, retries once at a smaller batch
size on CUDA OOM, and skips jobs already complete — safe to restart.

```bash
python run_queue.py --dry-run    # print the plan, run nothing
python run_queue.py              # run everything not already done
```

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
