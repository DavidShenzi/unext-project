# Deep Learning Final Project — Consolidated Notes

Selected paper: **UNeXt: MLP-based Rapid Medical Image Segmentation Network**
Published: MICCAI 2022, Valanarasu & Patel (Johns Hopkins)
Paper: https://arxiv.org/abs/2203.04967
Code: https://github.com/jeya-maria-jose/unext-pytorch

---

## 1. Course Project Demands

Source: `Deep Learning - final project (2).pdf`

- Groups of 2–3 students. Paper must be pre-approved via the groups' table by **2026-06-30**.
- Paper must **not** be directly related to your undergraduate final project.
- Core requirement: get the code running on a dataset and **explain the results** you obtain. Running the code (training/eval) is mandatory; modifying the architecture is optional but recommended.
- Final submission deadline: **2026-08-31**.

### Deliverables
1. **10–15 slide presentation** (ppt/pdf), written in your own words, English recommended even if presenting in Hebrew. Optional appendix slides allowed beyond the 15.
2. **First slide must contain** (penalty if missing):
   - Paper name, venue, publication date
   - All group members' names + IDs
   - Link to the code used (GitHub or Google Drive)
   - Link to a 15–20 minute video presentation
3. **Video presentation**:
   - ~15 min, max 20 min
   - Every group member presents a portion, on camera while speaking
   - Recorded via Zoom/Teams/Meet/OBS, uploaded **unlisted/private** to YouTube
   - Link pasted in the comment section of the first slide; test the link before submitting

### Recommended slide structure
1. Motivation and problem background
2. Solution proposed in the paper
3. Brief review of the paper's own reported results (keep short — leave room for your own experiments)
4. **Your own experiments and results** (this is the graded core, not optional)
5. *(Optional, recommended)* Limitations of the paper's solution + modifications you tried
6. *(Optional, recommended)* Challenges encountered

---

## 2. The Paper: UNeXt

### Core architecture
UNeXt keeps a U-Net-shaped encoder–decoder with skip connections but splits it into two stages:

- **Convolutional stage** (shallow levels): standard Conv3x3 → BN → ReLU blocks, narrower channel widths than U-Net (32/64/128/160/256 across 5 levels). Encoder uses max-pool downsampling; decoder uses bilinear upsampling instead of transpose convolutions (fewer parameters).
- **Tokenized MLP (Tok-MLP) stage** (bottleneck, deepest 2 levels): conv features are tokenized (projected to an embedding dimension via a stride/kernel-3 conv), then passed through:
  1. Shifted MLP (width-axis shift)
  2. Depth-wise convolution (DWConv) — acts as a cheap, resolution-robust positional encoding, replacing ViT-style learned positional embeddings
  3. GELU activation
  4. Shifted MLP (height-axis shift)
  5. Residual connection + LayerNorm

### Shifted MLP mechanism
Inspired by Swin Transformer's windowed attention and axial attention: features are split into partitions along one spatial axis and cyclically shifted by a fixed offset before tokenizing/mixing, done once for width and once for height. This is a **pure indexing operation** — it costs zero extra FLOPs or parameters — while injecting a local/windowed inductive bias into an otherwise fully global (all-tokens-mixed) MLP layer. Without this shift, MLP token-mixing has no notion of spatial locality; the shift lets nearby tokens interact more directly across the mixing operation without needing convolutions or attention.

### Why this design choice
Pure MLP-based bottlenecks are computationally very cheap (matrix multiplies, no attention's quadratic cost) but lack any built-in spatial/positional structure. UNeXt's answer is to inject structure two ways: (1) DWConv for position information, (2) axis-shifting for locality — both essentially free in compute, unlike self-attention's cost of building context.

### Results (paper's own benchmarks)
- Datasets: **ISIC 2018** (skin lesion, 2594 images, 512×512) and **BUSI** (breast ultrasound, 647 images, 256×256).
- Metrics: F1/Dice, IoU, parameter count, GFLOPs, CPU inference time.
- UNeXt: 1.47M params, 0.57 GFLOPs, 25ms inference; ISIC F1/IoU = 89.70/81.70; BUSI F1/IoU = 79.37/66.95.
- Vs. TransUNet (105.32M params, 38.52 GFLOPs, 246ms): UNeXt achieves **72× fewer params, 68× less compute, 10× faster inference**, with a statistically significant accuracy improvement (p < 10⁻⁵).
- Loss: 0.5·BCE + Dice loss. Optimizer: Adam, lr=0.0001, cosine annealing down to 0.00001. Batch size 8.

### Repo practicalities
- Small PyTorch codebase (`archs.py`, `train.py`, `val.py`, `dataset.py`, `losses.py`, `metrics.py`).
- No pretrained weights shipped — every run trains from scratch (satisfies the course's "run the code" requirement by default).
- Old pinned dependencies (Python 3.6, torch 1.7.1) — expect to need updated versions for current hardware.
- Only ISIC 2018 and BUSI wired up natively; extending to new datasets requires writing a new dataset loader following `dataset.py`'s pattern.

---

## 3. Known Limitations (from follow-up literature)

Concrete weaknesses that later papers identified and addressed:

1. **Semantic gap between encoder and decoder features** is not fully resolved by simple skip connections — addressed by *G-UNeXt*.
2. **Static/fixed token mixing** — the shift pattern in shift-MLP is a fixed, data-independent offset; it doesn't adapt to image content — addressed by a *Wave-MLP hybrid* replacing Tok-MLP with dynamic phase-based token aggregation.
3. **No multi-scale feature fusion** — a single-scale conv+MLP pipeline struggles with lesions of varying size/shape — addressed by *IS-UNeXt* (Inception-style multi-scale blocks + SE attention).
4. **Single-step prediction** — no mechanism to reuse information across training epochs/iterations — addressed by *FA-UNext* (feedback attention from prior epoch's prediction).
5. **Limited global context in lightweight conv+MLP designs generally** — *CMUNeXt* argues this class of architecture (including UNeXt) under-captures global context compared to attention, and that conv inductive bias (via large kernels) fits scarce medical data better than MLP tokenization.

Broader survey positioning: MLP-based segmentation networks like UNeXt sit as a favorable **efficiency/accuracy middle ground** between pure CNNs and Transformers — near-Transformer accuracy at a fraction of the compute — rather than a strict improvement over either family.

---

## 4. Proposed Modifications — Theoretical Explanations

For each modification, this is why it works and precisely what it changes structurally in UNeXt.

### 4.1 Squeeze-and-Excitation (SE) attention (from IS-UNeXt)

**Theoretical basis:** Standard convolutions treat every output channel equivalently once computed — there's no explicit mechanism for the network to weigh "this channel's feature map is more informative for this input" against another. SE blocks address this by learning a **channel-wise attention/gating** signal: global average pooling collapses each channel's spatial map to a single scalar (global context), a small MLP (fc → ReLU → fc → sigmoid) maps this vector of per-channel scalars to a set of weights in [0,1], and each channel's feature map is rescaled by its learned weight before being passed on. This is a form of soft, learned feature selection — channels correlated with the segmentation target get amplified, irrelevant/noisy channels get suppressed.

**How it changes the network:** Insert an SE block after each conv block in the conv stage (and optionally after Tok-MLP blocks). Structurally this adds: 1 global-avg-pool + 2 small FC layers + 1 sigmoid + 1 channel-wise multiply, per block — a small parameter/compute overhead (the FC layers are on a squeezed C-dimensional vector, not the full feature map, so cost is negligible relative to the conv itself). It does not change spatial resolution or the overall U-shape; it only re-weights channels at each stage. Expected effect: better discrimination of small/faint lesions where relevant signal is confined to a subset of channels, at very low compute cost — consistent with IS-UNeXt's reported parameter *reduction* alongside accuracy gains (the SE gating allowed shrinking other parts of the network without losing accuracy).

### 4.2 Multi-scale (Inception-style) fusion blocks (from IS-UNeXt)

**Theoretical basis:** A single 3×3 convolution has a fixed receptive field per layer, so the network's notion of "local context" is scale-rigid. Medical lesions vary hugely in size (a few pixels to filling half the image), so a fixed-scale feature extractor is a mismatch for at least some lesions in any dataset. Inception-style blocks run several convolutions of different kernel sizes (e.g., 1×1, 3×3, 5×5, or dilated variants) **in parallel** on the same input and concatenate their outputs, giving the network simultaneous access to multiple receptive field sizes at each layer, and letting subsequent layers (or attention) decide which scale's features matter for a given input.

**How it changes the network:** Replace (or augment) one or more of the conv-stage's plain Conv3x3 blocks with a multi-branch block: e.g., parallel 1×1, 3×3, 5×5 (or 3×3 with dilation 1/2/3) convs, each followed by BN+ReLU, concatenated channel-wise, then a 1×1 conv to project back to the target channel width. This increases parameter count and compute per block (multiple conv paths instead of one) but can be offset by reducing channel widths elsewhere (as IS-UNeXt does, netting an overall parameter *reduction* via this trade). The U-shape and Tok-MLP bottleneck are unaffected; only the conv-stage's internal receptive-field structure changes.

### 4.3 Dynamic/Wave-MLP-style token mixing (replacing fixed shift-MLP)

**Theoretical basis:** UNeXt's shift-MLP applies a **fixed, content-independent** cyclic shift before mixing tokens — every image gets the same shift offset regardless of what's in it. This gives locality (nearby tokens interact) but no adaptivity: the network cannot decide, per-image or per-region, how far or in what direction context should be aggregated from. Wave-MLP treats each token as a wave with an amplitude and a **phase term**; the phase is predicted from the token's own content, and mixing is done via phase-aware aggregation (roughly, tokens whose phases align contribute more to each other). This makes the aggregation *learned and input-dependent* rather than a fixed geometric shift — closer in spirit to what self-attention achieves (content-dependent context aggregation) but far cheaper computationally (no quadratic attention matrix).

**How it changes the network:** Replace the shift + MLP + DWConv block inside Tok-MLP with a Wave-MLP block: each token gets a learned amplitude and phase (via lightweight linear projections), and the token-mixing operation combines neighboring tokens weighted by phase similarity instead of by fixed shift-then-linear-mix. Parameter count increases modestly (extra projections for phase/amplitude), but the fixed-shift operation (previously free) is replaced by a learned, still relatively cheap mechanism. The conv stage and overall U-shape are untouched; only the internal token-mixing rule of the Tok-MLP block changes from "fixed shift + linear" to "content-adaptive phase-weighted mixing."

### 4.4 Feedback attention using prior epoch's prediction (from FA-UNext)

**Theoretical basis:** Standard segmentation training is single-step per input: the network sees the image once per forward pass and produces one prediction, with no mechanism to refine that prediction using its own prior output as a hint. Iterative refinement (used in pose estimation, some segmentation literature) argues that feeding a coarse/prior prediction back into the network as an auxiliary input lets the network attend more precisely to regions where it was previously uncertain (e.g., boundaries), functioning like a soft attention mask over its own errors — analogous to how residual/recurrent refinement improves upon single-pass estimates.

**How it changes the network:** After each epoch (or forward pass), the model's own output prediction mask is stored, then concatenated or attention-fused with the input at the start of the next epoch's forward pass — typically as an extra channel or as a gating signal on early conv features. This requires: (a) persisting predictions across epochs, (b) a small fusion module (e.g., concatenate + 1×1 conv to merge channels back to the expected width). It adds a modest parameter cost (the fusion layer) and changes the *training loop*, not just the architecture — the model must be able to consume its own previous prediction as an input, which also changes what a single "forward pass" means at inference time (may require an initial dummy/zero prediction map for epoch 1 / first inference pass).

### 4.5 Skip-Fusion / semantic-gap reduction in skip connections (from G-UNeXt / CMUNeXt)

**Theoretical basis:** Plain U-Net-style skip connections copy encoder features directly to the decoder at the matching resolution. But encoder features at a given depth are typically **more local/low-level** (edges, textures) while decoder features at the same nominal resolution have already been influenced by much deeper, more semantic/global context from the bottleneck — this mismatch is the "semantic gap." Directly concatenating mismatched-abstraction features can dilute the decoder's finer signal with noisy low-level encoder detail, or vice versa. A fusion mechanism that first transforms/aligns the encoder feature (rather than passing it through raw) reduces this mismatch before combining.

**How it changes the network:** Instead of `concat(encoder_features, decoder_features)` at each skip, insert a small transformation on the encoder side before fusion — e.g., a lightweight conv block or attention gate that reweights the encoder features conditioned on the decoder's (more global) features, then fuses (concat + conv, or additive gating). This adds a small number of parameters per skip connection (one gating/conv block × number of skip connections, typically 3–5 in UNeXt's structure) but does not change the overall depth, channel widths, or Tok-MLP internals — it is a localized change to how skip connections are combined, aimed specifically at closing the encoder/decoder abstraction mismatch rather than at capacity or receptive field.

---

## 5. How UNeXt Trains Effectively on Small Datasets

UNeXt is trained from scratch (no pretraining) on genuinely small datasets — ISIC 2018 (2594 images) and BUSI (647 images), 80/20 splits. This works for a combination of structural and loss-design reasons, not through heavy regularization.

### 5.1 Architectural capacity is small and locality-biased
1.47M parameters is tiny relative to transformer-based alternatives (TransUNet ≈105M — 72× larger). More important than the raw count is *where* the capacity sits: most of the network is still convolutional, and convolution carries a built-in prior (locality + translation-equivariance) that doesn't need to be learned from data — the network only has to learn *what patterns matter*, not *that nearby pixels are related*. This is precisely why pure self-attention/ViT-style bottlenecks (MedT, TransUNet) typically require large-scale pretraining: self-attention has no spatial prior at all, so it must learn locality itself, which needs data volume this setting doesn't have.

### 5.2 The shift-MLP injects locality into the one part that would otherwise have none
The Tok-MLP bottleneck replaces convolution with `Linear` layers, which are normally spatially blind — a plain MLP over flattened tokens has zero notion of "adjacent pixel." UNeXt's fix (in `archs.py`): before each Linear layer, feature-map chunks are cyclically shifted (`torch.roll`, shift window ≈ ±2 pixels) along height then width, with a depthwise convolution sitting between the two Linear layers. This forces each token's computation to mix in its actual spatial neighbors before the MLP projects it — approximating a convolution's locality bias cheaply, right at the one place (the bottleneck) where the model would otherwise be most exposed to overfitting on a small dataset. It functions as an inductive-bias substitute for data volume: the mechanistic reason a full/unshifted MLP bottleneck would be far more data-hungry than this one, even though the paper doesn't frame it explicitly as an anti-overfitting claim.

### 5.3 The loss function handles foreground/background imbalance directly
`loss = 0.5·BCE + Dice` (confirmed in `losses.py`, Dice smooth=1e-5). Medical lesion masks are heavily background-dominated; plain BCE alone can get dragged toward trivially predicting "background everywhere" on a small, imbalanced dataset. Dice loss directly rewards region overlap regardless of class imbalance, while BCE supplies stable per-pixel gradients early in training when Dice's gradient is close to undefined on near-empty predictions. This combination is standard practice for small, imbalanced medical segmentation data generally — it isn't UNeXt-specific, but it is load-bearing here.

### 5.4 Regularization itself is minimal
- **Augmentation** (confirmed in `train.py`): only random 90° rotation + horizontal/vertical flip (via albumentations `RandomRotate90` and `Flip`). No elastic deformation, no color jitter, no scaling/cropping — notably lighter than typical medical pipelines (e.g., nnU-Net's heavy elastic/gamma/noise augmentation).
- **Optimizer**: Adam with weight_decay=1e-4, lr=1e-4 (paper), cosine annealing schedule down to 1e-5.
- **Dropout** defaults to 0 in the `shiftmlp`/`shiftedBlock` classes and is not enabled in the actual training configuration.
- **Normalization**: LayerNorm inside the Tok-MLP blocks (per-token normalization, appropriate since inputs there are tokens rather than spatial feature maps); standard BatchNorm elsewhere in the conv stage.
- **Early stopping** exists as a CLI flag but defaults to disabled (`-1`) and isn't used in the paper's reported runs.
- No explicit train/val overfitting curves were found in accessible paper text (the arxiv PDF wasn't fully text-extractable during research) — flagged as an open gap rather than assumed.

> **RESOLVED BY OUR EXPERIMENTS (§7).** This open question now has a measured answer, and it
> contradicts the reading below. On BUSI (518 training images) the paper's configuration
> **overfits severely**: final train IoU 0.87–0.91 vs val 0.56–0.58, a **+0.31 generalization
> gap**, with validation peaking around epoch 95–107 and then declining for the remaining
> 300 epochs. Adding standard medical-imaging augmentation cuts that gap by ~60% and raises
> mean IoU by +0.040. So the structural argument below is real but *insufficient* — the
> architecture's inductive bias does not, on its own, substitute for regularization at this
> dataset size.

### 5.5 Bottom line
UNeXt doesn't fight the small-data problem with heavy regularization or augmentation — it fights it structurally. Keeping most of the network convolutional gives locality "for free," and smuggling a locality bias into the MLP bottleneck via the shift trick means there's very little spatial structure left that actually needs to be learned from ~2000 training images. The BCE+Dice loss then handles the remaining small-dataset problem (class imbalance) that architecture alone can't solve.

**Caveat from our runs:** this reasoning explains why UNeXt trains *at all* without pretraining,
but our measurements (§7) show it still overfits badly on BUSI. The structural prior reduces
how much must be learned from data; it does not eliminate the need for augmentation.

---

## 6. Experiment Plan (as executed)

1. ✅ **Baseline** — reproduce UNeXt on BUSI from the official repo, compare to the paper.
2. ✅ **SE attention** (§4.1) — implemented, trained from scratch, and later re-tested as a
   fine-tuning graft on a converged model. **No effect in either setting.**
3. ✅ **Skip-Fusion** (§4.5) — implemented as a gated fusion, fine-tuned with a frozen
   backbone. **Marginal gain (+0.006 IoU), below split noise.**
4. ✅ **Augmentation** (unplanned, emerged from diagnosis) — the intervention that actually
   worked: **+0.040 IoU across all three splits.**

ISIC 2018 was not run — BUSI alone produced enough material, and the reproduction gap became
the project's main finding.

---

## 7. Results

All runs: BUSI (647 images = 437 benign + 210 malignant; `normal` excluded to match the
paper's count), 80/20 split, `UNext` (1,471,921 params — matches the paper's 1.47M),
Adam lr 1e-4, cosine → 1e-5, BCE+Dice, batch 8, RTX 5060 Laptop. ~10 h GPU total.

Full experiment log with per-run detail: `unext/EXPERIMENTS.md`.

### 7.1 Headline

| | baseline (paper aug) | + strong augmentation | paper |
|---|---|---|---|
| mean IoU (3 splits) | 0.5931 ± 0.0210 | **0.6328 ± 0.0126** | 0.6695 |
| mean Dice (3 splits) | 0.7232 ± 0.0202 | **0.7718 ± 0.0013** | 0.7937 |
| train/val gap | +0.31 | +0.12 | — |
| peak epoch | ~ep100 | ep282–390 | — |

Params 1,471,921 and **0.577 GFLOPs** (paper: 0.57) confirm the architecture reproduction is
faithful. Paired t-test on augmentation: t=3.31, **p=0.080** (n=3) — consistent in direction
across all three splits, but not conventionally significant at this sample size.

### 7.2 We could not reproduce the paper's number

The authors' **exact README command** (`--input_w 512 --input_h 512 --epochs 500 -b 8
--lr 0.0001`) yields **0.5956 IoU**, not 0.6695. Six runs across four configurations all land
in **0.569–0.602**.

Five candidate explanations were tested and eliminated:

| hypothesis | verdict |
|---|---|
| Input resolution (256 vs 512) | ruled out — 512 scored *worse* (0.5956 vs 0.5975) at 4× compute |
| Mixed precision (AMP) | ruled out — tied with FP32 at every 50-epoch window |
| Decision threshold | ruled out — sweep 0.3–0.7 flat; 0.5 already optimal |
| Split composition | ruled out — seeds 41/42/43 balanced on class and lesion area |
| Split averaging (paper reports mean of 3) | ruled out — spread 0.039 vs gap 0.076 → **3.6σ** |

**No official weights exist to check against.** The repo ships none; GitHub issues #11 (Apr 2022)
and #21 (Jul 2022) request them and remain **open and unanswered**. The one public third-party
checkpoint (HuggingFace `MedOtter/UNeXt`) loads `strict=True` into our architecture but scores
**IoU 0.105** on our split under three different normalizations.

### 7.3 Diagnosis: the paper's augmentation under-regularizes

Every baseline run shows the same failure — validation peaks at **epoch 93–107**, then declines
for 300+ epochs while training loss falls to ~0.07 (near-memorization of 518 images). Final gap
**+0.31**.

Cause: the paper augments with **rotate90 + flips only** (measured 1.70 ms/image).

### 7.4 The fix: stronger augmentation

Adding Affine, brightness/contrast, gamma and elastic deformation (16.57 ms/image, 9.7× cost):

| seed | baseline | augmented | Δ |
|---|---|---|---|
| 41 | 0.6016 | 0.6215 | +0.020 |
| 42 | 0.5692 | 0.6306 | +0.061 |
| 43 | 0.6085 | **0.6464** | +0.038 |

**All three splits improved.** Gap to the paper narrowed 0.076 → 0.037 (IoU) and 0.071 → 0.022
(Dice). Dice variance collapsed **15×** (±0.0202 → ±0.0013) — augmentation makes results far
more reproducible.

> **Methodological caveat worth stating in the talk.** At a *matched* 100-epoch budget,
> augmentation shows **no gain** (mean −0.0005; seed 43 was −0.029). The benefit is not faster
> convergence — augmentation removes the overfitting ceiling that halts baselines at ~ep100,
> letting training continue productively to ep282–390. An earlier version of this analysis
> overstated the gain by comparing 400-epoch augmented runs against 150–250-epoch baselines.

### 7.5 Architectural modifications: both largely negative

Both were grafted onto the best converged model (`busi_split43_aug`, IoU 0.6464) and fine-tuned
with the **backbone frozen**, so any change is attributable to the new layers alone. Insertion
was verified lossless (Skip-Fusion bit-exact, max logit diff 0.00e+00).

| | SE attention (§4.1) | Skip-Fusion (§4.5) |
|---|---|---|
| new params | 8,160 (0.55%) | 86,864 (5.57%) |
| GFLOPs | 0.577 (unchanged) | 0.641 (**+11%**) |
| gradient from real loss | **0.000000** | 1e-3 – 4e-3 |
| weights after training | never moved | \|w\|max 0.08–0.12 |
| Δ IoU | **0.000** | **+0.006** |

Skip-Fusion's +11% compute for +0.006 IoU is a poor trade on a network whose entire selling
point is efficiency — worth saying plainly in the talk.

**The mechanism is the interesting result.** On a *converged* UNeXt, per-channel attention
receives essentially **no gradient signal** — the network has no use for channel reweighting at
that operating point, which explains why SE also did nothing when trained from scratch
(0.5953 vs 0.5975). Spatial gating on skip connections *does* get signal, because the baseline's
bare `torch.add` genuinely discards information about *where* encoder detail should be admitted.

Skip-Fusion's +0.006 IoU held across all thresholds 0.3–0.7 (so it is a genuine improvement, not
an operating-point shift), but it is **half the split-to-split std (±0.013)** — a single run
cannot establish significance, and three splits likely still would not.

### 7.6 Efficiency (reproduced faithfully)

| | ours | paper |
|---|---|---|
| params | 1,471,921 | 1.47 M ✓ |
| GFLOPs @256 | 0.577 | 0.57 ✓ |
| CPU latency | 12.5 ms | 25 ms |
| GPU latency | 3.6–4.6 ms | — |

The efficiency claims — the paper's central contribution — reproduce essentially exactly. It is
only the *accuracy* figure we could not match.

---

## 8. Suggested slide narrative

1. **Motivation** — medical segmentation needs point-of-care speed; U-Net/TransUNet are heavy.
2. **Solution** — UNeXt: conv stage + tokenized shifted-MLP bottleneck, 1.47M params, 0.57 GFLOPs.
3. **Paper's results** (keep brief) — 72× fewer params, 68× less compute, 10× faster than TransUNet.
4. **Our experiments** — efficiency reproduces exactly; **accuracy does not** (0.593 vs 0.669),
   and here are the five hypotheses we eliminated.
5. **Limitations + modifications** — diagnosed severe overfitting; augmentation fixes 60% of the
   gap; SE and Skip-Fusion give ~nothing, and we can show *why* (gradient signal).
6. **Challenges** — double-normalization bug in upstream `dataset.py`; frozen-BatchNorm drift;
   dead upstream dependencies; no released weights.

**The honest framing is the strong one:** this is a systematic reproduction study that identified
a real limitation and fixed most of it — not a lucky number match.
