# Reproducibility package — THEOS-2 land-cover segmentation (Paper 1)

Code, configuration, split definitions and result files behind the article

> C. Srimanee and S. Homhuan, *Split, scene and reference definition outweigh training choices in
> deep land-cover segmentation: evidence from 360 controlled runs on THEOS-2 imagery*
> (submitted to the International Journal of Remote Sensing).

The package lets a reader do three things, in increasing order of effort:

1. **Check every number in the article** against the result file it comes from (table below; no GPU,
   no imagery needed).
2. **Recompute the statistics** (factorial effects, block bootstrap, variance shares, six-class
   scores, effect-size summary) and the result figures from the stored evaluation files.
3. **Re-run the whole pipeline** (tiling, training 360 runs, evaluation, addenda), which needs the
   THEOS-2 imagery from GISTDA and a GPU cluster.

## What is here and what is not

| Included | Not included, and why |
|---|---|
| Model code (`paper1/models`), data loading, losses and metrics (`paper1/utils`) | THEOS-2 imagery and image tiles — property of GISTDA, not redistributable; request from GISTDA |
| Pipeline scripts (`paper1/scripts`) and the Slurm jobs actually used (`hpc/`) | Trained checkpoints (360 runs, ~100 MB each) — too large for git |
| Configs and the pre-registration of the additional experiments (`paper1/configs`) | Figures that show image pixels (article Figs. 1 and 6, S1, S2) and the 47 MB scene grid (Fig. S3) |
| Split definition, leakage audit and split geometry (`spatial_splits.json`, `paper1/analysis/shapes`) | The 2020 national land-use layer (LDD) and administrative boundaries — third-party data |
| All evaluation and analysis results (`paper1/analysis/results`) | Manuscript sources, translations and submission files |
| Result figures without imagery (`paper1/analysis/figure`); the 360 per-run training logs (`analysis/results/run_logs`: best validation metrics and per-epoch history) | Legacy experiments not reported in the article (YOLO, Mask R-CNN, Colab notebooks) |

## Where each result in the article comes from

Paths are relative to `paper1/`. "Script" is the program that wrote the file.

| Article | Quantity | Result file (`analysis/results/`) | Script (`scripts/`) |
|---|---|---|---|
| §2.2, Table 1 | tiles per split, class shares, leakage audit (132/340 → 0/304 …) | `spatial_splits.json` (`leakage_audit`), `spatial_splits_summary.csv` | `build_paper1_spatial_splits.py` |
| §2.2 | distance of external tiles to training (median 924 m, max 10,545 m) | `external_distance.json` | `external_distance.py` |
| §2.3, Fig. 3 | 1500 m blocks, one-tile buffer (365 tiles dropped), seed 401 | `spatial_splits.json`, `../analysis/shapes/blocks.geojson`, `tiles.geojson` | `build_paper1_spatial_splits.py`, `export_split_shapes.py` |
| §2.5, Table 2 | nine architectures, 8 conditions × 5 seeds = 360 runs | `rerun_tasklist.json`; configs in `../configs/experiments_rerun.yaml` | `build_rerun_tasklist.py`, `train_gpu.py` |
| §2.6 | validation-only ensemble selection (7 members) | `ensemble_selection.json` | `build_ensemble.py` |
| §2.6 | single reads of the test split and the external scene (timestamps, unlock reason) | `ensemble_eval_test.json`, `ensemble_eval_external.json`, `rerun_eval_test.json`, `rerun_eval_external.json` | `eval_ensemble.py`, `eval_rerun.py` |
| §2.8, §3.1, Fig. 4 | factorial effects per architecture with 95% CI; interactions | `rerun_effects_mean_iou.json`, `rerun_cells_mean_iou.json` | `analyze_rerun.py`, `fig_results.py` |
| §3.1 | overhead-pretrained encoders (SatlasPretrain, DINOv3-SAT) | `foundation_eval_*.json`, `dinov3rgb_eval_*.json`, `dinov3_factorial.json` | `eval_rerun.py`, `analyze_dinov3_factorial.py` |
| §3.2, Fig. 5 | every run on both held-out sets, seven and six classes | `rerun_eval_test.json`, `rerun_eval_external.json`, `six_class_summary.json` | `six_class_summary.py` |
| §3.2 | random vs buffered split (+2.83 points gap) | `random_splits.json`, `randomsplit_eval_*.json`, `addendum_results.json` | `build_random_split.py`, `analyze_addendum.py` |
| §3.2 | share of variance by factor and regime | `variance_decomposition.json` | `variance_decomposition.py` |
| §3.2, Fig. 6 | Roi Et transfer, raw and moment matched (11.11% → 39.11%), calibration | `roiet_splits.json`, `ensemble_eval_roiet_raw.json`, `ensemble_eval_roiet_zscene.json`, `calibration.json` | `build_roiet_tiles.py`, `eval_ensemble.py`, `eval_calibration.py` |
| §3.3, Fig. 7 (left) | blinded point audit, two reviewers, kappa | `label_review_result.json`, `label_review_result_round2.json`, `label_review_key.json` | `build_label_review.py`, `analyze_label_review.py`, `fig_results.py` |
| §3.3, Fig. 7 (right) | parcel adjudication against the 2020 layer (80 parcels; 13.3% change, 33.3% reference error) | `from_paper3/g1_adjudication_pilot.json` (reviewer sheets `g1_pilot_r1/r2.json`, key `g1_pilot_key.json`) | see *Known gaps* |
| §3.3, Table 3 | background removed at inference or in training (+5.33 / +6.39 points) | `background_eval_*.json`, `addendum_results.json` | `eval_background.py`, `analyze_addendum.py` |
| §3.3 | ensemble, seven vs six classes (58.94% → 72.38% / 80.29%) | `ensemble_bg_results.json`, `ensemble_bg_eval_*.json` | `analyze_ensemble_bg.py` |
| §3.3, Fig. 8 | land-cover maps of both scenes; class composition | `map_composition.json` (maps not included) | `predict_ensemble_scene.py`, `qgis_make_maps.py` |
| §3.4, Fig. 9 | effect sizes of the three groups on one axis | `effect_sizes.json` | `fig_effect_sizes.py` |
| App. A, Table S1 | pre-registered hypotheses and outcomes | `addendum_results.json`, `unified_results.json`; plan in `../configs/addendum_prereg_2026-10.md` | `analyze_addendum.py`, `analyze_unified.py` |
| App. B | ~394 GPU-hours, 364 jobs (4 resubmitted), per-run time and memory; inference 378 s for 154 km² | `compute_budget.json`, `inference_cost.json` | `compute_budget.py`; inference: see *Known gaps* |
| App. C | paired spatial-block bootstrap (10 test blocks, 10,000 resamples) | `block_bootstrap_test_mean_iou.json`, `block_bootstrap_external_mean_iou.json` | `block_bootstrap_paper1.py` |
| App. D | example tiles, all-methods tiles, scene grid agreement | `fig_examples_tiles.json`, `method_tiles_agreement.json`, `scene_grid_agreement.json` | `fig_examples.py`, `fig_methods.py`, `scene_grid_agreement.py` |
| App. E | how the networks use the near-infrared band (shuffle, low-pass, NIR-R-G inputs) | `unified_results.json`, `f0_nir_diagnostic.json` | `eval_unified.py`, `analyze_unified.py`, `f0_nir_diagnostic.py` |

Figure files in `analysis/figure/`: Fig. 2 `fig_workflow_mermaid.png` (from `fig_workflow.mmd`,
`render_mermaid.sh`), Fig. 3 `manual/fig_split_map_tiles.jpg`, Fig. 4 `fig_effects.png`, Fig. 5
`fig_models_7v6.png`, Fig. 7 `fig_audit.png`, Fig. 8 `manual/fig_map_qgis.png`, Fig. S4
`manual/fig_map7_qgis.png`, Fig. 9 `fig_effect_sizes.png`. The QGIS maps are produced by
`qgis_make_maps.py`.

## Running it

Python 3.11. `requirements.txt` pins the versions of the cluster environment used for every run;
install with `pip install -r requirements.txt`
(training also needs a CUDA build of PyTorch; the QGIS map script runs inside QGIS's Python).

**Level 2 — recompute statistics from the stored results** (no imagery, no GPU), from the
repository root:

```bash
# factorial effects and cell means (Fig. 4) from the 360 per-run training logs
python paper1/scripts/analyze_rerun.py --results paper1/analysis/results/run_logs
python paper1/scripts/block_bootstrap_paper1.py --eval paper1/analysis/results/rerun_eval_test.json \
    --a "*__b4_rgbnir_plain__*" --b "*__b0_rgb_plain__*"
python paper1/scripts/variance_decomposition.py
python paper1/scripts/analyze_addendum.py
python paper1/scripts/analyze_ensemble_bg.py
python paper1/scripts/six_class_summary.py
python paper1/scripts/fig_effect_sizes.py
python paper1/scripts/fig_results.py
```

**Level 3 — full pipeline** (needs the imagery and a Slurm cluster; job scripts in `hpc/` were
written for the CMU ERAWAN system and use `/project/gisnorth/...` paths that must be adapted):

1. Tiles and masks: `prepare_tiles.py`, `create_labels.py`, `make_nir_tiles.py`
2. Split: `build_paper1_spatial_splits.py` (seed 401, 1500 m blocks, one-tile buffer); the result
   is `spatial_splits.json` — use the stored file to reproduce the exact partition
3. Training: `build_rerun_tasklist.py` then `hpc/paper1_rerun.slurm` (`train_gpu.py`)
4. Selection and single held-out reads: `hpc/p1_eval_val.slurm`, `hpc/p1_select_and_test.slurm`,
   `hpc/p1_eval_external.slurm`
5. Pre-registered additions: `hpc/chain_paper1_addendum*.sh`, `hpc/p1_addendum_eval.slurm`,
   `hpc/p1_unified_eval.slurm`, `hpc/p1_calibration.slurm`

## Known gaps (being closed before publication)

- **Reference masks** are not yet in the package; they will be deposited with the release.
- `inference_cost.json` and `map_composition.json` were produced by commands run on the cluster
  that are not yet saved as scripts.
- The parcel adjudication summary (`from_paper3/g1_adjudication_pilot.json`) was computed in a
  companion project; the reviewer sheets and key are included, the summarising script is not yet.

## Licence and citation

Licence: to be chosen by the authors before release. Citation: see `CITATION.cff`.
