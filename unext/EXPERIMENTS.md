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
| mean Dice | 0.7344 ± 0.0208 | **0.7670 ± 0.0114** | **+0.033** |
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
| `busi_split41_aug` | 0.6215 | 390 | 0.7581 | `models/busi_split41_aug/` |
| `busi_split42_aug` | 0.6306 | 282 | 0.7631 | `models/busi_split42_aug/` |
| `busi_split43_aug` | **0.6464** | 331 | 0.7798 | `models/busi_split43_aug/` |

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

Secondary finding: **the split-to-split Dice spread narrows 4.4x** (per-image, ±0.0242 → ±0.0054; on the aggregate convention 1.8x, ±0.0208 → ±0.0114). An earlier revision quoted 15x against ±0.0013, a figure that came from the 8-run partial eval set and does not survive full 33/33 coverage. The three augmented
runs span 0.0107 on per-image Dice, against 0.0421 for the baselines —
augmentation makes results more reproducible, plausibly because baseline spread partly
reflected how severely each split overfit.

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

### 7.6 Boundary gate: the gate learned the opposite of the hypothesis

Aggregate results after two splits (third still running):

| split | parent (aug) | + boundary gate | delta |
|---|---|---|---|
| 41 | 0.6215 | 0.6269 | +0.0054 |
| 42 | 0.6306 | 0.6314 | +0.0009 |

**These numbers are not yet interpretable.** The runs fine-tune the *whole* backbone for
100 further epochs (`--freeze_except` was not set), starting from parents that had not
converged -- split 41's peaked at epoch 391 of 400. Both runs drop about 0.06 below their
parent at epoch 1 (0.6215 -> 0.5575, 0.6306 -> 0.5723) and then climb back, which is the
backbone being disturbed and re-converging, not gates learning on a stable base. The
`_cont` control (identical continuation, no gate) is queued to separate the two.

**What the gate actually learned.** The design assumed the gate would *open* on
boundaries, admitting encoder detail where the margin is. Probing the learned gates on
validation data shows the opposite. Mean gate value by edge-magnitude decile of the
encoder feature (lowest two deciles are exactly zero -- flat background -- so those bins
are empty):

| split | low edge ....................... high edge |
|---|---|
| 41 | 0.521 0.530 0.529 0.526 0.517 0.510 0.499 **0.478** |
| 42 | 0.502 0.475 0.472 0.468 0.464 0.460 0.455 **0.429** |

Monotone decreasing on both splits, independently. The gate *closes* as edge strength
rises: it admits less encoder detail at boundaries, not more.

The weight inspection agrees that the edge signal was not privileged. Across all four
gates on both splits the Laplacian channel's share of gate weight magnitude is 0.335-0.361
-- essentially exactly the 1/3 that undifferentiated weights over three equal input groups
would give. The one consistent departure is `fuse1`, the highest-resolution skip, where
the edge share is slightly elevated (0.361, 0.345); that is the level where boundary
detail would matter most, but the margin is small.

A plausible reading: at these feature scales a strong Laplacian response marks *speckle
and texture*, not lesion margin, so suppressing high-gradient encoder pixels is a sensible
denoising policy for the network to adopt -- and the gate found it. That is a real
mechanism, but it is not the one the module was designed around, and it means the module's
name over-claims. Whether the small aggregate gain survives the `_cont` control is a
separate question from whether the stated mechanism is what produced it.

### 7.7 Boundary gate: complete aggregate result, and boundary metrics

| split | parent (aug) | + boundary gate | delta |
|---|---|---|---|
| 41 | 0.6215 | 0.6269 | +0.0054 |
| 42 | 0.6306 | 0.6314 | +0.0009 |
| 43 | 0.6464 | 0.6447 | -0.0017 |

**Mean +0.0015, p = 0.54.** A second null on aggregate IoU. Still confounded with 100
epochs of whole-backbone fine-tuning until the `_cont` control lands (see 7.6).

Boundary metrics, in contrast to the wavelet mixer, lean positive:

| split | boundary F1 | HD95 (px) |
|---|---|---|
| 41 | 0.4671 -> 0.4803 (+0.0132) | 33.39 -> 32.14 (-1.25) |
| 42 | 0.4946 -> 0.4946 (-0.0000) | 33.12 -> 35.69 (+2.56) |
| 43 | 0.4602 -> 0.4636 (+0.0035) | 31.42 -> 30.73 (-0.69) |

Mean boundary F1 **+0.0056** (p = 0.29); mean HD95 +0.21 px (p = 0.88, dominated by split
42). Set against the wavelet mixer's -0.0169 on the same metric, the two modifications
move boundary quality in opposite directions -- but at n = 3 neither is significant, so
this is a direction worth reporting, not a result worth claiming.

The gate improves boundary F1 while *closing* on high-gradient pixels (7.6). Those are
consistent if the encoder's high-Laplacian pixels are mostly speckle: suppressing them
cleans up the contour even though the mechanism is denoising rather than the intended
boundary-admission.

### 7.8 Status at the end of the unattended session

Six of six queued jobs completed with no failures. Both modifications are aggregate nulls:

| modification | mean delta IoU | p | GFLOPs vs baseline |
|---|---|---|---|
| wavelet mixer | -0.0014 | 0.72 | **0.525 (-9%)** |
| boundary gate | +0.0015 | 0.54 | 0.672 (+16%) |

**Resolved: the controls ran on 8 Sep, 08:29-10:49.** All six completed the full
100-epoch schedule. They change the reading of the boundary gate materially, because the
`_bnd` figure above compares against the *graft parent*, which confounds the gate with 100
epochs of ordinary fine-tuning.

| comparison | what it isolates | mean delta | p |
|---|---|---|---|
| `_cont` vs `_aug` | fine-tuning alone, no gate | -0.0026 | 0.099 |
| **`_bnd` vs `_cont`** | **the gate itself** | **+0.0041** | **0.232** |
| `_bndf` vs `_cont` | gate only, backbone frozen | -0.0009 | 0.675 |

So the gate's effect roughly triples once the fine-tuning penalty is removed -- and is still
a null: +0.0041 sits below the 0.0107 seed-noise floor established in 7.9, and all three
pairs match on both seed and AMP, so this is the cleanest measurement in the section.

### 7.9 Validity check on the modification runs

Auditing every run's config before trusting the numbers turned up one defect worth
stating plainly.

**Seed confound on splits 42 and 43.** Every baseline run (`_aug`, `_ftl`) uses
`seed = 41` and varies only `split_seed`. The modification runs queued in this session
used `seed = split_seed`, i.e. 41/42/43:

| split | baseline seed | modification seed | comparison |
|---|---|---|---|
| 41 | 41 | 41 | clean |
| 42 | 41 | 42 | confounded with initialisation |
| 43 | 41 | 43 | confounded with initialisation |

So on two of three splits the architecture change is entangled with an initialisation
change. This does not overturn either conclusion -- both modifications were nulls, and
adding init noise makes a null easier to obtain, not harder -- but the effect sizes are
noisier than a clean paired design would give, and the p-values are correspondingly
weaker. Any future run should pass `--seed 41` explicitly to match the baselines.

**AMP confound, previously undisclosed.** The same audit found a second difference the
seed table above does not capture: every `_wave`, `_bnd`, `_bndf` and `_cont` run sets
`amp: true`, while their `_aug` graft parents ran `amp: false`. So those comparisons vary
mixed-precision as well as architecture and seed. The `_ftl` comparison is unaffected
within pairs (seed 41 runs are `amp: false`, seed 101 runs are `amp: true`, matched on both
sides) -- but that means the seed-41-positive / seed-101-negative pattern in the Focal
Tversky results is *perfectly confounded with AMP*, and cannot be attributed to the seed
with the runs that exist.

**Eval coverage was incomplete until 8 Sep.** `eval.yml` existed for only 8 of 33 runs, and
Dice, GFLOPs and latency are stored nowhere else -- so every Dice figure quoted in earlier
revisions of this document came from those 8 runs. In particular the Focal Tversky Dice
column rested on a single run (`busi_split43_ftl`). `val.py` has since been run on the
remaining 25; coverage is now 33/33, and the aggregate-convention means are 0.7344 ± 0.0208
(baseline), 0.7670 ± 0.0114 (strong augmentation), 0.7759 ± 0.0132 (Focal Tversky).

**The noise floor matters more than the effect sizes.** Comparing `busi_split41_wave`
(seed 41) against `busi_split41_wave_s101` (seed 101) at matched epoch 178 gives a
difference of **0.0062** from the initialisation alone. Both modifications' measured
effects -- -0.0014 for the wavelet mixer, +0.0015 for the boundary gate -- are *smaller
than the spread produced by changing the random seed*. At n = 3 splits, effects this size
are not measurable with this protocol, whatever the architecture does.

That is the honest headline for both modifications, and it is a stronger statement than
either individual p-value: not "we failed to find an effect", but "an effect of this size
is below the resolution of the experiment".

**Also incomplete on disk** (pre-existing, unrelated to this session): `busi_split41`
(251/400), `busi_split42` and `busi_split43` (150/400), `busi_split43_SEft` (31/100).
These are the runs already flagged in section 5 as lacking `eval.yml`; they are truncated
in the logs too, and any figure using them should say so.

### 7.10 Independent verification of the analysis

Two audits were run against the raw logs with no access to the conclusions. They confirmed
most claims and corrected three. The corrections are recorded here because two of them
were errors that would have been presented.

**Retracted: "IoU and false-positive rate trade off against each other."** Across all 31
scored runs, best IoU and clean rate correlate at r = -0.559 (p = 0.0011), which was read
as a general accuracy/hallucination tradeoff. It is not. Controlling for a single binary
"strong augmentation" indicator:

| | r | p |
|---|---|---|
| raw | -0.5585 | 0.0011 |
| partial, controlling for aug group | **-0.0364** | 0.85 |
| within strong-aug runs only (n = 25) | **+0.0545** | 0.80 |

There are two clusters -- paper-recipe runs with low IoU and high clean rate, strong-aug
runs with the reverse -- and the "correlation" is the line joining their centroids. The
correct statement is about the augmentation change specifically, not about IoU in general.

**Corrected: checkpoint selection is 26 of 32, not 27.** Selecting on min val_loss rather
than max val_iou loses IoU in 26 runs, ties in 7, and **gains in none** (Wilcoxon
p = 3.0e-08). The "never gains" half is the stronger claim and is exact.

**Corrected: the overfitting gap figures.** The quoted +0.33-0.36 vs +0.08-0.13 were
measured at the final epoch, not at the selected checkpoint. At the checkpoint epoch, and
restricted to runs that completed their schedule, the separation still holds with no
overlap (paper-aug 0.1969; strong-aug -0.0080 to 0.1182), but an audit using truncated
runs found an overlap of 0.001. Any version of this claim must state that it excludes
incomplete runs.

**Strengthened: the wavelet false-positive result.** Adding the `split41` seed-101 pair
gives four paired comparisons, all positive:

| pair | aug clean | wave clean | delta |
|---|---|---|---|
| split41 seed41 | 0.0150 | 0.1579 | +0.1429 |
| split42 seed41 | 0.0226 | 0.0902 | +0.0677 |
| split43 seed41 | 0.0451 | 0.1880 | +0.1429 |
| split41 seed101 | 0.0752 | 0.1579 | +0.0827 |

**Mean +0.1090, t = 5.513, p = 0.0117.** This is the only result in the project that
strengthens as data is added, and the only one that clears p = 0.05. It is measured on an
evaluation set no model was trained on, and it is invisible to the metric the study
otherwise reports: the same comparison on IoU is -0.0014 (p = 0.72) and on Dice -0.0001
(p = 0.94).

**Open question flagged by the audit.** Every false-positive number uses a fixed 0.5
threshold. If the wavelet model is simply less confident rather than better behaved, a
threshold sweep will show its advantage collapsing; if the advantage holds across
thresholds, it is architectural. That sweep is inference-only and is the highest-value
outstanding analysis.

### 7.11 Threshold sweep: is the wavelet advantage architectural or calibration?

The false-positive result in 7.10 is measured at a fixed 0.5 threshold, which leaves the
obvious objection open: a model whose outputs simply sit lower would raise fewer alarms on
everything, normal scans included, and that advantage would vanish the moment anyone moved
the threshold. `threshold_sweep.py` tests this by sweeping 0.1-0.9 and measuring both
sides at each threshold -- false positives on the 133 normals, and lesion IoU on the run's
own validation split.

**At matched lesion IoU** (baseline at its best threshold, wavelet at whatever threshold
reproduces that IoU):

| pair | baseline | wavelet at matched IoU | false positives |
|---|---|---|---|
| split41 | 0.6239 @ thr 0.4, meanFP 4239 | 0.6247 @ thr 0.5, meanFP 2585 | **-39%** |
| split43 | 0.6491 @ thr 0.5, meanFP 2635 | 0.6512 @ thr 0.7, meanFP 2143 | **-19%** |
| split42 | 0.6341 @ thr 0.4 | never reaches it (best 0.6292) | no match exists |
| split41 s101 | 0.6376 @ thr 0.4 | never reaches it (best 0.6191) | no match exists |

Two pairs show a real reduction at equal segmentation quality. Two cannot be matched at
all, because the wavelet run's IoU ceiling on that split sits below the baseline's -- which
is itself informative, and is not visible in the 0.5-threshold table.

**Two arguments against the pure-calibration explanation:**

1. Interpolating the wavelet curve to each baseline threshold's IoU, the wavelet model has
   fewer false positives at **15 of 20 matched-IoU points (75%)**.
2. A less-confident model would peak at a *lower* threshold. The wavelet run peaks at
   **0.5 against the baseline's 0.4** -- the opposite direction.

**But the statistics do not support a strong claim.** The naive binomial on 15/20 gives
p = 0.041, and that number should not be reported: the twenty points are nine correlated
thresholds within each of four pairs, not twenty independent trials. At the honest unit of
analysis the tally is 3 pairs favouring the wavelet mixer, 1 split evenly, sign test
**p = 0.625**.

**Verdict.** The advantage is not *merely* calibration -- the threshold-peak direction and
the matched-IoU reductions on two splits both argue against that. But it is not
established as architectural either, and on two of four pairs the wavelet model simply
cannot reach the baseline's segmentation quality at any threshold. The defensible claim is
narrower than 7.10's p = 0.0117 suggests: **at the operating point both models are trained
and selected for, the wavelet mixer raises far fewer false alarms on lesion-free tissue;
whether that survives arbitrary re-thresholding is not resolved by n = 4.**
