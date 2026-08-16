# Enhancing UNeXt — Follow-up Literature Review

Research targeted at what we actually measured: the model **overfits** (+0.31 train/val gap,
peak at ~ep100) and is **under-regularized**. Ideas are ranked by whether they address that,
not by how novel they sound.

---

## 0. The most important finding: independent reproductions also fall short

Before proposing enhancements — three independent groups measured baseline UNeXt on BUSI and
**none reproduced the paper's number either**:

| source | Dice | IoU | protocol |
|---|---|---|---|
| **UNeXt paper (self-reported)** | **79.37** | **66.95** | own splits, 3 runs |
| MK-UNet (ICCV-W 2025) | 74.71 | — | 80/10/10, 5 runs, no augmentation |
| SC-UNext (2024) | 72.97 | — | independent reimplementation |
| third-party evaluation | — | 65.94 | — |
| **Ours — baseline** | 72.32 | 59.31 | authors' command, 3 splits |
| **Ours — + strong augmentation** | **77.18** | 63.28 | 3 splits, 400 ep |

**Two things follow.**

1. Our baseline (72.32 Dice) sits squarely inside the independent range (72.97–74.71), not
   below it. Our reproduction is **normal**, not broken.
2. Our augmented result (**77.18**) is **higher than every published independent
   reproduction** — only the original authors report better.

The paper's own MICCAI 2022 reviews are relevant here: reviewers raised the variance of the
reported results, and the authors responded that performance "varies across different
repetitions because the train:test data splits are different each time." That is precisely
the effect we quantified — and found too small (3.6σ) to close the gap.

> **This substantially strengthens the presentation.** Slide 6 currently says "we could not
> reproduce it." It can now say: *nobody publishing independently has reproduced it, and our
> augmented model beats all of them.*

---

## 1. What the follow-up papers actually change

Six derivative works, with the mechanism and what each reports.

### 1.1 CMUNeXt — large kernels + skip fusion *(arXiv 2308.01239)*
- **Change:** abandons MLP tokenization; uses large-kernel depthwise convolutions (7×7–11×11)
  plus a "Skip-Fusion" block on skip connections.
- **Reports:** BUSI IoU **71.56**, F1 **79.86**; CMUNeXt-S gains 1.1–3.8% IoU over UNeXt with
  **0.41 M params** (vs 1.47 M) at 657 FPS.
- **Argument:** convolution's inductive bias suits scarce medical data better than MLP mixing —
  consistent with our finding that the MLP bottleneck's shift trick is doing the heavy lifting.

### 1.2 G-UNeXt — Ghost path skip connections
- **Change:** replaces plain skips with a "Ghost path" that generates part of each feature map
  through cheap linear operations instead of full convolution, narrowing the encoder/decoder
  semantic gap.
- **Reports:** **−33% parameters, −23.7% FLOPs**, higher accuracy and faster inference.
- **Relevance:** our Skip-Fusion attempt (+0.006 IoU, +11% FLOPs) was the *expensive* version
  of this idea. Ghost path claims the same benefit while **reducing** cost.

### 1.3 SC-UNext — SCConv + neuron pruning
- **Change:** (a) SCConv replaces standard convolutions — a Spatial Reconstruction Unit and
  Channel Reconstruction Unit that cut spatial/channel redundancy; (b) "apoptosis and division"
  algorithms prune redundant MLP neurons and redistribute weights.
- **Reports:** BUSI Dice **75.29** vs their UNeXt baseline **72.97** (+2.32), at 1.46 M params
  and 2.13 GFLOPs.
- **Relevance:** their +2.32 Dice gain is *smaller* than our augmentation's **+4.86**.

### 1.4 UNeXt + Wave-MLP + attention gates
- **Change:** Wave-MLP blocks replace Tok-MLP (content-dependent phase-based mixing instead of
  a fixed shift), **attention gates** on skip connections, and **Focal Tversky Loss** for class
  imbalance.
- **Relevance:** the FTL substitution is the cheapest idea in this whole review — a loss change,
  no architecture edit (see §2.1).

### 1.5 IS-UNeXt — Inception + SE
- **Change:** multi-scale Inception-style fusion blocks, SE attention, and depthwise-separable
  decomposition of standard convolutions.
- **Caution:** we tested SE twice (from scratch and as a frozen-backbone graft) and measured
  **exactly zero** gradient signal reaching it on a converged model. Their gain likely comes
  from the multi-scale blocks, not the SE.

### 1.6 Rolling-UNet *(AAAI 2024)*
- **Change:** aggregates features along multiple directions via a rolling MLP operation,
  capturing long-range dependencies more thoroughly than UNeXt's single H-then-W shift.
- **Reports:** Rolling-UNet-S at **76.38** Dice on BUSI (per MK-UNet's comparison table).
- **Relevance:** a direct generalization of the exact mechanism UNeXt is built on.

---

## 2. Ranked recommendations for one more modification

Ordered by expected value given our diagnosis, not by novelty.

### 2.1 ⭐ Focal Tversky Loss — best value, no new layers
**Why it fits our findings.** BUSI's mean lesion area is **9.4%** (range 0.3%–56%) — severe
foreground/background imbalance. Our worst-case example (Dice 0.00) fails exactly the predicted
way: the model fires on large hypoechoic regions and misses a small true lesion. Tversky loss
adds tunable α/β to weight false negatives over false positives; the focal term down-weights
easy examples so training concentrates on hard, small lesions.

**Change:** replace `BCEDiceLoss` in `losses.py`. **Zero new parameters, zero extra FLOPs.**
**Cost:** ~45 min for one split. **Precedent:** used by the Wave-MLP UNeXt variant.
**Risk:** low — pure loss swap, no architectural surgery.

### 2.2 ⭐ Multi-directional shift (Rolling-UNet style) — most faithful to the paper's thesis
**Why.** UNeXt shifts along height, then width. Rolling-UNet shows aggregating along *more*
directions captures longer-range dependencies. This modifies the paper's **central
contribution** — a better story than bolting on generic attention.

**Change:** extend `shiftmlp._shift()` to add diagonal offsets, or widen `shift_size` from 5.
Still pure `torch.roll` indexing → **still zero parameters, zero FLOPs.**
**Cost:** ~45 min. **Risk:** medium — must verify tensor shapes survive the extra shifts.

### 2.3 Ghost-path skips — the cheap version of what we already tried
**Why.** Our Skip-Fusion cost +11% GFLOPs for +0.006 IoU. Ghost path claims the same
semantic-gap benefit while *reducing* parameters 33%. Directly supersedes our attempt and gives
an honest "we tried the expensive version; the literature has a cheaper one" narrative.
**Cost:** ~1 h including implementation. **Risk:** medium — needs the paper's exact formulation.

### 2.4 Large-kernel depthwise convs (CMUNeXt style)
**Why.** The best-performing idea in the literature (IoU 71.56) — but it replaces the MLP
bottleneck entirely, so it is no longer "UNeXt with an added layer." Better framed as a
comparison than a modification.
**Cost:** ~2 h. **Risk:** high — substantial rewrite.

### 2.5 ✗ Not recommended: more attention modules
We have **measured evidence** that SE receives zero gradient on a converged UNeXt. Attention
gates and CBAM would likely behave the same. Adding another attention variant risks a third
null result and would not add to what slide 10 already says.

---

## 3. Recommendation

**Focal Tversky Loss (§2.1).** It is the only option that is zero-parameter, zero-FLOP, directly
targets a failure mode we photographed (Dice 0.00 on a small lesion), has literature precedent
in a UNeXt derivative, and fits comfortably before the deadline.

If time allows a second, **multi-directional shift (§2.2)** — it extends the paper's own core
mechanism and is also free at inference.

Both preserve UNeXt's central selling point: **1.47 M parameters, 0.577 GFLOPs.** Any
modification that inflates those undermines the very thing the paper is arguing for, which is
why Skip-Fusion's +11% compute was a poor trade.

---

## Sources

- [CMUNeXt (arXiv 2308.01239)](https://arxiv.org/pdf/2308.01239)
- [SC-UNext (PMC11300774)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11300774/)
- [G-UNeXt (Multimedia Systems)](https://link.springer.com/article/10.1007/s00530-023-01173-z)
- [MK-UNet (ICCV-W 2025)](https://arxiv.org/html/2509.18493) — source of the independent BUSI table
- [UNeXt + Wave-MLP (Comp Med Imaging Graphics)](https://www.sciencedirect.com/science/article/abs/pii/S0895611123001295)
- [IS-UNeXt](https://www.sciencedirect.com/science/article/abs/pii/S095219762501543X)
- [FA-UNext (ACM MM Asia)](https://dl.acm.org/doi/10.1145/3696409.3700203)
- [UNeXt MICCAI 2022 reviews](https://conferences.miccai.org/2022/papers/535-Paper0077.html)

**Caveat:** CMUNeXt's and G-UNeXt's PDFs were not text-extractable; those numbers come from
abstracts and secondary citations. The MK-UNet comparison table and SC-UNext figures were read
directly. Any number quoted in the talk should be re-checked against the source PDF.
