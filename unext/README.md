# UNeXt — course project code

Adapted from [jeya-maria-jose/UNeXt-pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch)
(UNeXt, MICCAI 2022) for Python 3.12 / PyTorch 2.11 / CUDA 12.8.

Baseline (`UNext`) plus the SE-attention modification (`UNext_SE`) described in
`../project_notes.md` §4.1.

---

## 1. Install

```powershell
pip install -r requirements.txt
```

Torch is deliberately **not** in `requirements.txt` — you already have
`2.11.0+cu128`, which is the build that supports the RTX 5060. Installing torch
from a requirements file risks downgrading you to a CPU-only wheel.

## 2. Smoke test (no dataset needed)

```powershell
python smoke_test.py
```

Checks both architectures forward/backward at 256² and 512², prints parameter
counts, and reports the SE overhead vs. baseline. Run this first — it catches
environment problems before you spend time downloading data.

## 3. Get the data

Neither dataset is downloadable without registration; both need manual download.

### BUSI (recommended first — 647 images, 256², fast to train)

Breast Ultrasound Images Dataset, Al-Dhabyani et al. 2020.
Download `Dataset_BUSI_with_GT.zip` from the
[Cairo University dataset page](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
(also mirrored on Kaggle as "Breast Ultrasound Images Dataset").

Unzip anywhere, then:

```powershell
python prepare_busi.py --raw "C:\path\to\Dataset_BUSI_with_GT"
```

This writes `inputs/busi/images/` and `inputs/busi/masks/0/`.

> The `normal` class (133 empty-mask cases) is **excluded** by default, giving
> 437 benign + 210 malignant = **647 images**, matching the paper's count. Pass
> `--include_normal` for all 780 if you want that ablation — but state it in the report.

### ISIC 2018 (2594 images, 512² — the second dataset)

ISIC 2018 Task 1 (lesion boundary segmentation). Register at
[challenge.isic-archive.com](https://challenge.isic-archive.com/data/#2018) and
download:
- `ISIC2018_Task1-2_Training_Input.zip` (images)
- `ISIC2018_Task1_Training_GroundTruth.zip` (masks)

```powershell
python prepare_isic.py `
  --images "C:\path\to\ISIC2018_Task1-2_Training_Input" `
  --masks  "C:\path\to\ISIC2018_Task1_Training_GroundTruth"
```

## 4. Train

Defaults follow the **paper** (Adam, lr 1e-4, batch 8, 400 epochs, cosine anneal
to 1e-5, BCE+Dice), not the upstream repo's generic CLI defaults.

```powershell
# baseline
python train.py --dataset busi --arch UNext    --name busi_UNext    --input_h 256 --input_w 256

# SE modification
python train.py --dataset busi --arch UNext_SE --name busi_UNext_SE --input_h 256 --input_w 256
```

Both use `--seed 41`, so they see the **identical** train/val split — the ablation
is clean. To reproduce the paper's 3-split averaging, rerun with
`--split_seed 41 / 42 / 43` and average.

Start with a short run to check timing before committing to 400 epochs:

```powershell
python train.py --dataset busi --arch UNext --name smoke --epochs 5
```

ISIC at 512² will need a smaller batch on a laptop GPU:

```powershell
python train.py --dataset isic --arch UNext --name isic_UNext --input_h 512 --input_w 512 -b 4
```

Outputs land in `models/<name>/`: `config.yml`, `log.csv` (per-epoch metrics),
`model.pth` (best val IoU), `summary.yml` (best score, epoch, wall-clock time).

### If you hit CUDA out-of-memory

In order of preference: lower `-b` (8 → 4 → 2), then drop `--input_h/w` to 256,
then `--amp true` (already the default). AMP is on by default and roughly halves
activation memory.

## 5. Evaluate

```powershell
python val.py --name busi_UNext
python val.py --name busi_UNext_SE
```

Writes predicted masks to `outputs/<name>/0/` and metrics to
`models/<name>/eval.yml`: IoU, Dice, parameter count, GFLOPs, and GPU + **CPU**
per-image latency (the paper's table reports CPU inference time).

## 6. Build the comparison table + curves

```powershell
python compare_runs.py
```

Emits `results_table.csv` (one row per run — the results slide) and `curves.png`
(val Dice + train/val loss per epoch, all runs overlaid — useful for the
overfitting discussion in §5 of the notes).

---

## Changes from upstream, and why

These matter for the "challenges encountered" slide.

**Made it run at all on a modern stack**
- Upstream imports `mmcv` in `archs.py` but never calls it. Dropped — `mmcv` is a
  painful source build on Windows and buys nothing here.
- `timm.models.layers` → `timm.layers` (upstream path deprecated in timm ≥ 0.9).
- `albumentations.augmentations.transforms.Flip` → `A.HorizontalFlip` /
  `A.VerticalFlip` (the v1→v2 API moved these and removed the combined `Flip`).
- Removed a dead module-level `shift()` function in `archs.py` that referenced
  undefined names (`xs`, `self`) and would have raised if ever called.
- Hard-coded `.cuda()` → a resolved `device`, so CPU smoke tests work.

**Correctness fixes**
- `shiftmlp.forward` reused the input channel count `C` for the second shift, but
  after `fc1` the tensor has `hidden_features` channels. With the paper's
  `mlp_ratio=1` these are equal so upstream happens to work, but the code breaks
  for any other ratio. Now reads the actual channel count.
- `val.py` hard-coded `random_state=41` for the split regardless of what the run
  was trained with — evaluating on training images if you'd trained with another
  seed. It now reads `split_seed` back from the run's `config.yml`.
- `Dataset.__getitem__` raised an opaque `TypeError` deep in the loader on a
  missing/corrupt file; now raises `FileNotFoundError` naming the path.

**Metrics — read this before reporting numbers**
- Upstream derives Dice from IoU as `2·IoU/(IoU+1)`. That identity holds for a
  single set, so applied to a whole batch it is **not** the mean per-image Dice
  the literature reports — it's a monotone transform of batch-aggregate IoU.
- `metrics.py` keeps that function unchanged (as `iou_score`) so baseline numbers
  stay comparable to the paper's table, and adds `iou_dice_per_image` for the
  standard per-image mean. Both are logged. **Say which one you report.**

**Additions for the report**
- `--seed` / `--split_seed`, so baseline and modified runs are directly comparable.
- AMP (on by default), pinned memory, `num_workers=0` default (safe on Windows).
- Wall-clock training time, parameter count, GFLOPs, GPU + CPU latency.
- `prepare_busi.py` / `prepare_isic.py` — upstream ships no preprocessing at all.
- `smoke_test.py`, `compare_runs.py`.

## Files

| file | purpose |
|---|---|
| `archs.py` | `UNext`, `UNext_S`, `UNext_SE` |
| `train.py` | training loop, paper defaults |
| `val.py` | evaluation, latency/FLOP benchmarks, mask dumps |
| `dataset.py` | image/mask loader |
| `losses.py` | `BCEDiceLoss` (0.5·BCE + Dice) |
| `metrics.py` | IoU/Dice, both conventions |
| `utils.py` | seeding, param counting, meters |
| `prepare_busi.py` | raw BUSI → loader layout |
| `prepare_isic.py` | raw ISIC 2018 → loader layout |
| `smoke_test.py` | no-data sanity checks |
| `compare_runs.py` | results table + training curves |
