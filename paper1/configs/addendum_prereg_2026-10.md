# Paper 1 addendum — pre-registration (written 2026-10-07, before any run)

Three additions to the JSTARS manuscript, frozen here before training or inference.
Nothing below has been run. Commit this file before the first job is submitted so the
git timestamp precedes the results.

Existing locked protocol still applies: selection on validation only, held-out sets
read once, no hyperparameter change. All new runs reuse the training configuration in
`experiments_rerun.yaml` unchanged.

## A. Random split vs buffered spatial split (controlled)

**Question.** On the same labels, code and training-set size, how much does a random
tile-level split inflate held-out accuracy, and does it change which conclusions the
factorial would support?

**Design.**
- Split `R`: tiles of pool A (2,354) drawn at random, seed 401, into train/val/test of
  exactly the spatial-split sizes (1,286 / 304 / 399); the remainder is unused. Tiles in
  pixel contact with any external (Scene 2) tile are excluded before drawing, so the
  external set stays clean for both designs. Built by
  `paper1/scripts/build_random_split.py`, written to
  `paper1/analysis/results/random_splits.json`.
- Architectures: `unetpp_effb3`, `segformer_b2`, `upernet_convnextt` (CNN,
  transformer, modern-CNN control). Conditions: `b0_rgb_plain`, `b1_rgb_aug`,
  `b4_rgbnir_plain`, `b5_rgbnir_aug` (class weighting off). Seeds 0–4.
  3 × 4 × 5 = **60 runs**.
- Checkpoint chosen on the `R` validation split (what a practitioner using `R` does).
- Each `R` run is evaluated on its own `R` test split and on the external Scene 2 set.
  The matching spatial runs (same arch × condition × seed) already exist.

**Hypotheses and tests.** Paired by arch × condition × seed (n = 60); 95% intervals by
bootstrap over seeds within architecture (10,000 resamples).
- **A-H1 (inflation).** gap = mIoU(own test) − mIoU(external). gap_R − gap_S > 0, interval
  excludes 0.
- **A-H2 (NIR reversal).** NIR effect = mean of (b4 − b0, b5 − b1). Under `R`-test it is
  more positive than under spatial test (difference > 0, interval excludes 0). Note: the
  direction was suggested by the pre-rerun round (`paper1/_archive/README.md`), so this
  is a confirmatory test of a hypothesis generated from earlier, non-comparable data.
- **A-H3 (augmentation).** Augmentation effect (mean of b1 − b0, b5 − b4) is smaller
  under `R`-test than under spatial test. Directional; interval reported.
- **A-E1 (ranking, exploratory).** Kendall τ between the ranking of the 12
  arch × condition cells on own-test and on external, for `R` and for spatial.

A null or opposite result is reported as found.

## B. Variance decomposition of the existing 360 runs (descriptive)

Share of variance in test and external mIoU attributable to architecture, each training
factor and their interactions, and to seed (residual), from a fixed-effects ANOVA on
the 9 × 8 × 5 grid. Descriptive only; no hypothesis. Computed from
`rerun_eval_{test,external}.json`.

## C. Cross-region transfer probe: Roi Et (post-registration of Paper 2 data)

**Status.** Roi Et imagery and the 445 server masks belong to Paper 2. No Paper 1
model has been applied to Roi Et. The scene differs from Lamphun in region, season and
year at once, and carries an unrecorded resampling step (Paper 2 caveat B1), so this is a
transfer *probe*, not a generalization claim.

**Design.** Apply all 360 rerun checkpoints and the locked validation-selected ensemble,
unchanged, to Roi Et tiles cut on the Paper 1 grid (512 px, step 448) from
`paper2/data/roi_et/roiet_utm48n_0p5m.tif` and the co-registered
`roiet_labels_utm48n_0p5m.tif` (built from the 445 server masks; same 8-class codes
0–7, checked 2026-10-07), remapped to the 7-class scheme (grass → background) and
converted to tile files with the same DN-to-tile transform as `prepare_tiles.py` /
`make_nir_tiles.py`. Only tiles with ≥ 50% labelled pixels are kept. Two input variants, both reported: `raw` DN and `zscene` (per-band moments of
Roi Et matched to Lamphun Scene 1 training tiles). Inference unit: Roi Et spatial blocks
at 1,500 m.

**Hypotheses.**
- **C-H1.** Ensemble mIoU on Roi Et is below its Lamphun external mIoU (58.94%).
  Reported with a block-bootstrap interval; the size of the drop is the result.
- **C-H2.** The augmentation effect estimated on Roi Et (all 9 architectures, factorial
  contrast as in the main analysis) stays positive, interval excluding 0, under
  `zscene`.
- **C-H3.** `zscene` mIoU > `raw` mIoU for the ensemble (paired by block).

Roi Et is read once per variant after this file is committed. If the class remap or
tile alignment fails a check, the probe is dropped and reported as not done rather than
repaired after seeing accuracy.

## D. Removing background as a prediction target (added 2026-10-07, before any run)

**Why.** Section 4.6 finds that every audited background point is a real land-cover
class, and the confusion analysis shows every class losing pixels to background. The
manuscript recommends masking background from the loss; D tests that recommendation
and a no-retraining alternative.

**Arms** (architectures `unetpp_effb3`, `segformer_b2`, `upernet_convnextt`; condition
`b1_rgb_aug`; seeds 0–4; spatial split unchanged):
- `std` — existing rerun checkpoints, ordinary 7-class argmax.
- `sup` — the same checkpoints, argmax over the six real classes only (background logit
  suppressed at inference). No retraining.
- `ign` — **15 new runs** trained with background pixels excluded from the loss
  (`--ignore-background`: cross-entropy and Dice computed over classes 1–6 on
  non-background reference pixels only); checkpoint chosen on validation mIoU over the
  six real classes; background never predicted.

**Metrics.** Computed on reference pixels whose class is not background (background
reference pixels are dropped, not scored). For each real class c:
IoU6_c = TP / (TP + FP + FN), where a background prediction on a class-c pixel counts as
FN for c and as FP for no class. mIoU6 is the mean over the six classes. Also reported:
share of all pixels (including background reference) predicted as background.
Evaluated on val, test and external; test and external read once.

**Hypotheses** (paired by architecture × seed, n = 15; 95% interval by bootstrap over
seeds within architecture, 10,000 resamples):
- **D-H1.** `ign` mIoU6 > `std` mIoU6 on external.
- **D-H2.** `sup` mIoU6 > `std` mIoU6 on external.
- **D-E1 (exploratory).** `ign` versus `sup`, and per-class changes, on all splits.

The predicted-background share is descriptive. A null or opposite result is reported.
Conventional 7-class mIoU of `std` remains the headline metric of the paper; D is
reported as an addendum and does not replace it.

## E. Remote-sensing-pretrained encoder (added 2026-10-08, before any run)

**Why.** All nine architectures use ImageNet-pretrained encoders. A reviewer will ask
whether pretraining on overhead imagery changes the picture.

**Design.** `satlas` = SatlasPretrain `Aerial_SwinB_SI` (Swin-B + FPN, pretrained on
0.5–2 m aerial RGB; the same weights file and construction as Paper 2's
`train_unet_rspretrained.py`) with a light conv head. Condition `b1_rgb_aug` only
(the model is RGB), seeds 0–4, spatial split, training recipe of
`experiments_rerun.yaml` unchanged (batch 8, lr 1e-4, combo loss, patience 20).
**5 runs.** The ensemble stays as locked on 2026-08-21; `satlas` is not added to it.

**Comparator.** Among the nine ImageNet architectures in `b1_rgb_aug`, the one with the
highest mean *validation* mIoU over its five seeds (chosen on validation only).

**Hypotheses** (two-sample over seeds, 95% bootstrap interval, 10,000 resamples):
- **E-H1.** `satlas` mean external mIoU > comparator mean external mIoU.
- **E-E1 (exploratory).** Same contrast on the internal test split; on Roi Et
  (`zscene`) if the C files are complete.
A null or opposite result is reported.

### E amendment (2026-10-08, before any E run was submitted)

A second remote-sensing encoder is added, and the fine-tuning recipe of both is fixed
before either runs:
- `dinov3` = DINOv3-SAT ViT-L/16 (`vit_large_patch16_dinov3.sat493m`, pretrained on
  SAT-493M, Maxar RGB at 0.6 m) with a ViTDet-style simple pyramid (blocks 6/12/18/24)
  and FPN fusion head (`paper1/models/dinov3_seg.py`), input normalised with the
  SAT-493M statistics.
- Both `satlas` and `dinov3` are fine-tuned end to end with the backbone learning rate at
  0.1× the head rate (`--backbone-lr-mult 0.1`; head 1e-4), the usual practice for large
  pretrained backbones. All other settings as in `experiments_rerun.yaml`.
- 5 seeds each, `b1_rgb_aug`: **10 runs**. E-H1 is tested separately for each encoder
  against the same validation-chosen comparator; the two tests are reported together
  without correction and labelled as such.

## F–H. Transfer of design decisions across evaluation regimes (added 2026-10-09, before any F–H computation or run)

Source of the plan: `reports/มุมวิจัยใหม่ THEOS2 segmentation.md`. Paper framing: which
design decisions (bands, augmentation, class weighting, architecture, pretraining,
background handling) keep their conclusion across evaluation regimes — random split,
buffered test, adjacent held-out scene, Roi Et.

**Status of evidence.** Hypotheses formed after the 360-run results were seen are
confirmatory only on new runs or new reads. Re-analyses of test/external numbers that
were already read (F0, F1 on existing files) are labelled exploratory/diagnostic.

**Practical-equivalence margin:** 0.5 mIoU points, fixed here, used for every
non-inferiority statement below.

**Unified evaluation pass (one read per checkpoint).** All F2–F6 quantities are computed
from a single pass, on the H100 partition, writing per-tile confusion matrices so every
later analysis is a computation on saved files: for every checkpoint of the 360 rerun
runs, the 60 random-split runs, the 15 background-ignored runs, the 10 foundation runs
and the new G/H runs, on val, test and external (and Roi Et `zscene` for the 360 + G/H):
(i) 7-class and background-suppressed confusion per tile; (ii) the same split into the
tile's 32-px edge strip (overlap zone with neighbours: 64 px / 2) versus interior;
(iii) max-softmax confidence histograms (20 bins) per class; (iv) for RGB+NIR runs,
NIR perturbed at test time: shuffled across tiles within the split, low-passed to 2 m
(4×4 mean then bilinear), and replaced by its training-set mean. Code is developed and
checked on val before the pass is launched; the pass is run once.

- **F0 (diagnostic, existing files).** NIR main effect on test recomputed leaving out
  each test block in turn (block-mean mIoU, since per-block confusion matrices of the
  first read were not stored), and per class. Decision rule fixed here: if the sign of
  the pooled NIR effect flips when any single block is removed, NIR is treated as a
  secondary topic and G is reduced to one architecture (`segformer_b2`).
- **F1 (exploratory).** Variance shares (B) and Kendall τ of the 72 cells, per regime
  (val, test, external, Roi Et zscene, random-test), with block-bootstrap intervals.
- **F2-H1.** On matched random-split vs buffered runs (A design), the edge-minus-interior
  mIoU difference on the own test split is larger under the random split (paired by
  arch × condition × seed, seed-within-arch bootstrap, interval excludes 0).
- **F3-H1.** For RGB+NIR runs on external, mIoU loss under NIR shuffling is smaller than
  the margin (0.5 points) — i.e. the models make little use of NIR — tested by the upper
  bound of the 95% interval. Low-pass and mean-replacement reported descriptively.
- **F4 (exploratory).** Re-score all runs under P7 (7-class, background scored), P6-eval
  (background reference pixels dropped, background predictions counted wrong) and P6-sup
  (background logit suppressed); report whether factor-effect signs and the architecture
  ranking change.
- **F5 (descriptive).** ECE, Brier score and risk–coverage of the locked ensemble and of
  seed ensembles; temperature fitted on val only; on test, external and Roi Et zscene.
- **F6 (exploratory).** Scores against the original labels versus the audited points;
  cross-seed/architecture disagreement as a detector of audited label errors
  (precision@k and AUPRC versus random ranking).

**G. NIR mechanism (new runs).** Architectures `unetpp_effb3`, `segformer_b2`,
`upernet_convnextt`; augmentation on, class weighting off; seeds 0–4; buffered split.
- **G1 `nrg`:** NIR, R, G as three channels through the unchanged ImageNet 3-channel
  stem (15 runs). **N-H2:** `nrg` is non-inferior to `b1_rgb_aug` on external (lower
  bound of the paired difference > −0.5).
- **G2 `rgb_nirlp`:** R, G, B + NIR low-passed to 2 m (4×4 mean, bilinear back to 0.5 m),
  same 4-channel stem as `b5` (15 runs). **N-H3:** `rgb_nirlp` is non-inferior to
  `b5_rgbnir_aug` on external.
Comparators are the existing `b1`/`b5` runs of the same architectures and seeds.

**H. Pretraining controls (new runs).** `b1_rgb_aug`, seeds 0–4, buffered split.
- **H1:** `satlas` and `dinov3` with the backbone frozen (head trained only), same head
  and recipe as E (10 runs). Reported alongside E; expectation (not a test): frozen <
  fine-tuned on external.
- **H2:** ImageNet-pretrained Swin-V2-B in the identical Satlas construction (same FPN,
  head, backbone LR ×0.1), 5 runs. **P-H1:** the test-minus-external gap of `satlas` is
  smaller than that of ImageNet Swin-V2-B (paired by seed, interval excludes 0).

G and H add 45 runs, queued after the registered A and E runs. Tier-3 extensions (block
split without buffer, G under the random split, DINOv3 web ViT-L) are not registered
here and will be registered separately before any of them runs.

## I. RGB factorial for the DINOv3-SAT encoder (added 2026-10-12, before any I run)

**Why.** The factorial estimates the effects of augmentation and class weighting for nine
ImageNet-pretrained architectures. Addendum E trained `dinov3` in one condition only
(`b1_rgb_aug`), so it cannot say whether those effects hold for a large encoder
pretrained on satellite imagery. This addition was requested after the E/H results had
been read; it is registered here before any of its runs is submitted.

**Design.** `dinov3` exactly as in the E amendment (DINOv3-SAT ViT-L/16, same head,
backbone learning rate ×0.1, all other settings of `experiments_rerun.yaml`), buffered
spatial split, seeds 0–4, in the three RGB conditions it has not been trained in:
`b0_rgb_plain`, `b2_rgb_wt`, `b3_rgb_aug_wt`. **15 new runs.** Together with the five
existing `b1_rgb_aug` runs of E these form a complete 2×2 (augmentation × class
weighting) with five matched seeds. The near-infrared conditions are **not** run: the
encoder is an RGB model and a fourth input channel would have no pretrained weights, so
its near-infrared effect would not be comparable with that of the nine architectures.
`satlas` is not extended.

**Estimator.** As in `analyze_rerun.py`: within a seed, a main effect is the mean of the
two cells at the high level minus the mean of the two at the low level, and the
interaction uses the product of the ±1 codes; five per-seed estimates are summarised by
their mean and a 95% t interval. Effects are reported on validation mIoU (the scale of
the main factorial) and, from one evaluation pass per split, on the internal test split
and the held-out scene; Roi Et (`zscene`) is exploratory.

**Hypotheses.**
- **I-H1.** The augmentation effect of `dinov3` on validation mIoU is positive with an
  interval excluding zero, as for the nine architectures.
- **I-H2.** The class-weighting effect of `dinov3` on validation mIoU has an interval
  including zero.
- **I-E1 (exploratory).** The same two effects on the internal test split, the held-out
  scene and Roi Et; the augmentation × weighting interaction.
A null or opposite result is reported. The main factorial, the ensemble (locked
2026-08-21) and every number already reported are unchanged; `dinov3` is reported as a
supplement to the factorial, not pooled into it. The existing `b1_rgb_aug` evaluations
of E are reused, not re-read.

## J. Six-class score of the locked ensemble (added 2026-10-13, before the J evaluation)

**Why.** The headline accuracy is the seven-class mIoU of the validation-selected
ensemble, and background is one of the seven classes although every audited background
point is a real land-cover class. Addendum D measured what removing background as a
prediction target does for three architectures; it did not score the ensemble, so the
headline number has no six-class counterpart. No new labels and no retraining are
involved.

**Design.** The ensemble as locked on 2026-08-21 (`ensemble_selection.json`, same
members and weights), softmax averaged as in `eval_ensemble.py`.
- `std` — ordinary seven-class argmax. Its confusion matrices on test and external
  already exist (`ensemble_eval_{test,external}.json`); the six-class score of `std` is
  computed from them without a new read.
- `sup` — argmax over the six real classes (background probability excluded). One new
  evaluation pass on validation, test and external writes the `std` and `sup` confusion
  matrices, pooled and per spatial block, to `ensemble_bg_eval_<split>.json`; the `std`
  matrix of that pass must equal the existing one.

**Metric.** mIoU6 exactly as defined in section D (reference pixels that are not
background; a background prediction on a real-class pixel is a false negative for that
class). The share of all pixels predicted as background is descriptive.

**Hypothesis.**
- **J-H1.** `sup` mIoU6 > `std` mIoU6 on the held-out scene, with a 95% interval of the
  difference excluding zero (block bootstrap over spatial blocks, 10,000 resamples,
  confusion matrices summed within each resample).
- **J-E1 (exploratory).** The same difference on the internal test split; per-class IoU6.
A null or opposite result is reported. The seven-class mIoU of `std` remains the headline
metric; J is reported next to it and does not replace it.
