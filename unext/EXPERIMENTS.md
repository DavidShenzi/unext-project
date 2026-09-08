# Experiment Log — UNeXt on BUSI

All runs: BUSI (647 images, 437 benign + 210 malignant, `normal` excluded to match the
paper's count), 80/20 split, `UNext` arch (1,471,921 params), Adam lr 1e-4,
CosineAnnealingLR → 1e-5, weight decay 1e-4, BCEDiceLoss, batch 8, RTX 5060.

Paper's reported BUSI numbers: **IoU 0.6695 / F1-Dice 0.7937** (mean of 3 random splits).

---

## Headline result

| | baseline (paper aug) | strong aug | Δ |
|---|---|---|---|
| mean IoU | 0.5931 ± 0.0210 | **0.6328 ± 0.0126** | **+0.040** |
| mean Dice | 0.7232 ± 0.0202 | **0.7718 ± 0.0013** | **+0.049** |
| gap to paper (IoU) | −0.076 | **−0.037** | — |
| typical train/val gap | +0.31 | +0.12 | −60% |
| peak epoch | ~ep100 | ep282–390 | — |

Paired t-test on the three splits: t=3.31, **p=0.080** (n=3) — consistent in direction
across all three splits, but not conventionally significant at this sample size.

---

## 1. Baseline reproduction

The authors' exact README command reproduces to **~0.59**, not the published 0.6695.
Six runs across four configurations all land in 0.569–0.602.

| run | config | best IoU | peak ep | log dir |
|---|---|---|---|---|
| `busi_UNext` | 256px, 400ep, AMP | 0.5975 | 80 | `models/busi_UNext/` |
| `busi_UNext_512` | 512px, 500ep, no AMP — **authors' exact command** | 0.5956 | 229 | `models/busi_UNext_512/` |
| `busi_split41` | 256px, 400ep, no AMP | 0.6016 | 107 | `models/busi_split41/` |
| `busi_split42` | seed 42, stopped @150 | 0.5692 | 95 | `models/busi_split42/` |
| `busi_split43` | seed 43, stopped @150 | 0.6085 | 93 | `models/busi_split43/` |

**3-split baseline mean: 0.5931 ± 0.0210.**

### Hypotheses tested and eliminated

| hypothesis | verdict | evidence |
|---|---|---|
| Input resolution (256 vs 512) | **ruled out** | 512 scored 0.5956 vs 256's 0.5975 — slightly worse, 2.2× the compute |
| Mixed precision (AMP) | **ruled out** | AMP vs no-AMP statistically tied at every 50-epoch window; see `log_512_amp_stopped_ep173.csv` |
| Decision threshold | **ruled out** | sweep 0.3–0.6 is flat; 0.5 already optimal |
| Split composition | **ruled out** | seeds 41/42/43 give near-identical benign/malignant balance and lesion fraction |
| Split averaging (paper reports mean of 3) | **ruled out** | spread 0.039 vs gap 0.076 → **3.6σ short**; even the best split (0.6085) falls well below 0.6695 |

### Pretrained weights do not exist
- Official repo ships none. GitHub issues [#11](https://github.com/jeya-maria-jose/UNeXt-pytorch/issues/11)
  (Apr 2022) and #21 (Jul 2022) request them — **both still open and unanswered**.
- Third-party [MedOtter/UNeXt](https://huggingface.co/MedOtter/UNeXt) `BUSI.pth` loads
  `strict=True` into our architecture (1,472,952 params) but scores **IoU 0.105** on our
  split. Tried [0,1], ImageNet, and our normalization — ImageNet was best and still 0.105.
  Its embedded config (`base_lr: 0.01`, 300 epochs, own train/val files) shows a different
  training codebase. Cause of the low score not established; not usable as a reference.

---

## 2. Diagnosis: overfitting

Every baseline run shows the same failure:

- Peak val IoU around **ep93–107**, then flat-to-declining for the remaining 300+ epochs
- Final train/val gap **+0.31** (e.g. `busi_split41`: train 0.87 vs val 0.56)
- Val loss minimum coincides with the IoU peak, then rises monotonically
- Train loss falls to ~0.07 — near-memorization of 518 training images

Root cause: the paper's augmentation is **rotate90 + flips only**. Measured cost
1.70 ms/image. That is insufficient regularization for 518 training images.

---

## 3. Modification: strong augmentation

`--aug strong` adds Affine (shift/scale/rotate), RandomBrightnessContrast, RandomGamma,
and ElasticTransform on top of the paper's rotate+flip. Cost 16.57 ms/image (9.7× more).
Implemented in `build_train_transform()` in `train.py`.

| run | best IoU | peak ep | best Dice | log dir |
|---|---|---|---|---|
| `busi_split41_aug` | 0.6215 | 390 | 0.7732 | `models/busi_split41_aug/` |
| `busi_split42_aug` | 0.6306 | 282 | 0.7709 | `models/busi_split42_aug/` |
| `busi_split43_aug` | **0.6464** | 331 | 0.7711 | `models/busi_split43_aug/` |

**All three splits improved** (+0.020, +0.061, +0.038).

### Important caveat on the mechanism

At a **matched 100-epoch budget** augmentation shows *no* gain:

| seed | baseline @100ep | aug @100ep | Δ |
|---|---|---|---|
| 41 | 0.5763 | 0.5844 | +0.008 |
| 42 | 0.5692 | 0.5881 | +0.019 |
| 43 | 0.6085 | 0.5800 | **−0.029** |
| mean | 0.5847 ± 0.0209 | 0.5842 ± 0.0041 | **−0.0005** |

The benefit is **not** faster convergence. Augmentation removes the overfitting ceiling
that stops baselines improving past ~ep100, letting training continue productively to
ep282–390. Any comparison at a short budget will miss this entirely — and an earlier
draft of this analysis overstated the gain by comparing 400-epoch augmented runs against
150–250-epoch baselines.

Secondary finding: **Dice variance collapses 15×** (±0.0202 → ±0.0013). All three
augmented runs land within 0.002 of each other on Dice — augmentation makes results
markedly more reproducible, plausibly because baseline spread partly reflected how
severely each split overfit.

---

## 4. SE attention (first modification attempt)

`busi_UNext_SE` — SE blocks after every conv block, encoder and decoder.
+8,160 params (+0.55%). See `UNext_SE` in `archs.py`.

**Result: 0.5953 vs baseline 0.5975 — no measurable benefit** (single split, 256px,
400ep). Difference is well inside split noise (±0.021), so the honest claim is
"no effect detected", not "SE is worse". Never re-tested under strong augmentation.

---

## 5. Where everything lives

```
unext/
├── models/<run>/          config.yml, log.csv (per-epoch metrics), model.pth (best),
│                          last.pth (resumable: model+optimizer+scheduler+epoch),
│                          summary.yml, eval.yml (where val.py was run)
├── outputs/busi_UNext/0/  predicted masks (qualitative figures)
├── inputs/busi/           647 preprocessed image/mask pairs
├── train_*.log            stdout per run (*_r2, *_r3 = resumed segments)
├── results_table.csv      consolidated table, all 9 runs
├── curves.png             val Dice + train/val loss, all runs overlaid
└── log_512_amp_stopped_ep173.csv   AMP-vs-noAMP comparison data
```

Run inventory (9 runs, ~9 h GPU, 170 MB checkpoints):

| run | epochs | eval.yml? | note |
|---|---|---|---|
| `busi_UNext` | 400 | ✓ | first baseline |
| `busi_UNext_SE` | 400 | ✓ | SE modification |
| `busi_UNext_512` | 500 | ✓ | authors' exact command |
| `busi_split41` | 251 | — | stopped; peak ep107 |
| `busi_split42` | 150 | — | partial, resumable |
| `busi_split43` | 150 | — | partial, resumable |
| `busi_split41_aug` | 400 | ✓ | |
| `busi_split42_aug` | 400 | — | |
| `busi_split43_aug` | 400 | ✓ | best overall |

Gaps: `busi_split41/42/43` and `busi_split42_aug` have no `eval.yml` (never run through
`val.py`) — the numbers above come from `log.csv`. Running `val.py --name <run>` would
fill these in for GFLOPs/latency/per-image metrics.

---

## 6. Code changes from upstream

Fixes required to run at all (Python 3.12 / torch 2.11 / CUDA 12.8):
- dropped unused `mmcv` import; `timm.models.layers` → `timm.layers`
- `albumentations` v1 → v2 API (`Flip` → `HorizontalFlip`/`VerticalFlip`, `Affine`)
- removed dead module-level `shift()` referencing undefined names

Correctness fixes:
- **Double normalization**: `dataset.py` divided by 255 *after* `A.Normalize()`, squashing
  inputs from ±2.5 into **±0.01**. Present in upstream. Fixed — this materially affects training.
- `shiftmlp.forward` reused input channel count after `fc1`; only correct because the
  paper sets `mlp_ratio=1`. Fixed to read the actual channel count.
- `val.py` hard-coded `random_state=41`, so evaluating a run trained with another seed
  would score partly **on training images**. Now reads `split_seed` from the run config.

Additions: `--aug {paper,strong}`, `--stop_after` (halt without changing cosine `T_max`),
`--resume`, `last.pth` full-state checkpoints, per-image metrics alongside upstream's
aggregate convention, seeding, GFLOPs/latency benchmarks, BUSI/ISIC preprocessing scripts.

**Metric note:** upstream derives Dice as `2·IoU/(IoU+1)`, which equals the *aggregate*
Dice, not the mean per-image Dice the literature usually reports. Both are logged
(`val_dice` vs `val_dice_per_image`). State which convention you report.

---

## 7. Two further modifications (queued 2026-09-07)

Two architectures aimed at the same weakness -- UNeXt's handling of blurred, infiltrative
lesion margins -- but attacking it at different points in the network.

### 7.1 `UNext_Wave` -- wavelet token mixer

Replaces the shifted-MLP token mixer in all four Tok-MLP blocks. A parameter-free Haar
DWT splits each feature map into four sub-bands; the low band (LL) runs through the
existing MLP at half resolution, and the three detail bands each get a dilated depthwise
3x3. The inverse transform recombines them, so the block is a drop-in for `shiftmlp`.

The motivation is that `torch.roll` is frequency-blind: one rigid operator handles both
the smooth interior of a lesion and its margin, averaging sharp boundary evidence
together with low-frequency texture. Routing the bands separately lets the margin path
have its own operator.

Because the MLP then sees a quarter of the tokens, the block is **cheaper** than the one
it replaces:

| | params | GFLOPs |
|---|---|---|
| `UNext` (baseline) | 1,471,921 | 0.577 |
| `UNext_Wave` | 1,493,041 (+1.4%) | **0.525 (-9%)** |

Trained from scratch (400 ep), since the mixer has no counterpart in a baseline
checkpoint to graft from.

**Risk to watch:** BUSI is ultrasound, and speckle noise is high-frequency. The HH band
may carry mostly speckle rather than boundary. A null result here is still informative,
in the same way the SE result was.

### 7.2 `UNext_Boundary` -- boundary-gated skip

Extends `SkipFusion`. That gate predicts how much encoder detail to admit from the
*values* of the encoder and decoder features; this one additionally feeds it a fixed
Laplacian edge response computed from the encoder feature, so it can key on spatial
gradient directly instead of inferring boundaries from values:

    e   = |laplacian(encoder)|                    # fixed kernel, no parameters
    g   = sigmoid(conv1x1([decoder, encoder, e]))
    out = decoder + 2*g * encoder

Keeps `SkipFusion`'s `identity_init` contract, so it grafts onto a trained checkpoint and
fine-tunes (100 ep from the matching `*_aug` run). Grafting is both ~3x cheaper than
training from scratch and a cleaner ablation: the comparison against `busi_split43_SkipFt`
isolates the edge signal, since gate topology and backbone are otherwise identical.

1,602,049 params / 0.672 GFLOPs, against `UNext_SkipFusion`'s 1,558,785 / 0.641.

### 7.3 Pre-flight verification

Checked before committing GPU time, since an unattended queue cannot diagnose itself:

- Haar DWT/IWT reconstruct to max abs error 3e-7, including odd spatial sizes and under
  AMP float16; 99.98% of a smooth signal's energy lands in LL as theory requires.
- `UNext_Boundary` with `identity_init` matches the baseline **bit-exactly** (max diff
  0.000e+00); with random gate init it differs, confirming the gate is live.
- No dead gradients in either model. The detail convs and gates start at zero but receive
  nonzero gradient -- i.e. not the failure mode that made SE attention a null result.
- Graft loads 101 backbone tensors, 12 new, 0 unused; resumes at IoU 0.608, matching the
  parent checkpoint.

One performance defect was found and fixed here: expanding the DWT filter bank with
`repeat()` on every call cost 0.134 ms against the transform's own 0.100 ms. Caching it
per (channels, dtype) took `UNext_Wave` from 1.8x baseline step time to 0.96x.

### 7.4 Queue

`run_queue.py` runs the jobs in priority order, skipping any whose `model.pth` already
exists (so it is restartable) and continuing past individual failures. Order is chosen so
that a queue cut short still yields a coherent result -- all three splits of one condition
-- rather than scattered partial runs:

1. `busi_split{41,42,43}_wave` -- 400 ep from scratch (~3.4 h each)
2. `busi_split{41,42,43}_bnd` -- 100 ep grafted (~1.0 h each)
3. the same six at seeds 101 and 202 (split held fixed, init varied)

Measured throughput settled at ~17 s/epoch once the GPU left its initial throttled
window (the first 100 epochs suggested ~31 s/epoch, which overestimated the total by
almost 2x). A 400-epoch wavelet run takes ~1.9 h, so stages 1-2 are ~7.3 h and the full
18-job queue is ~22 h. Stage 3 exists because three splits at p~0.08 cannot resolve a +0.005 effect;
whether it completes depends on available time.

Note the ~31 s/epoch is data-loader bound, not GPU bound (`num_workers=0`, a deliberate
choice for Windows -- see the note in `train.py`). Raw GPU step time would predict roughly
half that.

### 7.5 Wavelet mixer: result (3 splits, complete)

| split | baseline (aug) | wavelet | delta |
|---|---|---|---|
| 41 | 0.6215 | 0.6246 | +0.0031 |
| 42 | 0.6306 | 0.6226 | -0.0080 |
| 43 | 0.6464 | 0.6471 | +0.0007 |

**Mean -0.0014, p = 0.72.** No effect on aggregate IoU in either direction. Given the
block is 9% cheaper in GFLOPs, "matches the baseline at lower cost" is the fair summary
of the headline metric -- not an improvement in accuracy.

Boundary-localised metrics are consistently worse, on all three splits:

| split | boundary F1 | HD95 (px) |
|---|---|---|
| 41 | 0.4671 -> 0.4578 (-0.0093) | 33.39 -> 34.92 (+1.52) |
| 42 | 0.4946 -> 0.4564 (-0.0382) | 33.12 -> 36.83 (+3.71) |
| 43 | 0.4602 -> 0.4569 (-0.0033) | 31.42 -> 32.66 (+1.25) |

Mean boundary F1 -0.0169 (p = 0.26), mean HD95 +2.16 px (p = 0.11). Three out of three in
the same direction on both metrics, but with n = 3 that is **not** statistically
significant -- an exact sign test on 3/3 gives p = 0.25 at best. The defensible claim is
"no benefit, with a consistent tendency toward worse boundaries", not "makes boundaries
worse".

**A pattern that did not survive the third split.** After splits 41 and 42 the penalty on
small lesions looked strikingly reproducible (-0.0356 and -0.0361, terciles of
ground-truth mask area). Split 43 came out +0.0078:

| split | small | medium | large |
|---|---|---|---|
| 41 | -0.0356 | -0.0124 | -0.0006 |
| 42 | -0.0361 | +0.0147 | -0.0213 |
| 43 | +0.0078 | +0.0122 | +0.0015 |

Mean -0.0213, p = 0.28. Two runs agreeing to three decimal places was coincidence, not
signal -- a useful reminder of what n = 2 is worth on this dataset.

**Note the two IoU conventions disagree in sign on split 41.** The checkpoint is selected
on *aggregate* val IoU, where large lesions dominate and the wavelet run wins (+0.0031);
per image it loses (-0.016). This is the discrepancy flagged in the metric note at the end
of section 6, showing up in practice.

**Mechanism.** The detail convolutions did learn -- they start at exactly zero and reach
mean |w| ~0.012, comparable to the MLP's own ~0.015, so the path is not dead. But
`detail.2` (the HH diagonal band) ends up roughly *half* the magnitude of the two oriented
bands in every block. In B-mode ultrasound, HH is dominated by speckle rather than
anatomy, so the network appears to have learned to suppress a band carrying mostly noise
-- the risk anticipated in section 7.1 before the runs started. This is an observation
about learned weights, not a controlled experiment; an ablation that removes the HH branch
entirely would be the way to test it properly.
