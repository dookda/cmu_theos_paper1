#!/bin/bash
#SBATCH --job-name=p1-addm-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-addm-eval-%j.out

# Paper 1 addendum evaluation, pre-registered in paper1/configs/addendum_prereg_2026-10.md.
# Every output goes to its own file; the main rerun_eval_* / ensemble_eval_* files are
# never written here.
#   P1_EVAL=DC  addendum D (std/sup/ign on val, test, external) and C (Roi Et raw + zscene)
#   P1_EVAL=C   addendum C only (rerun after DC timed out; use -p gpu-h100 --time=2-00:00:00)
#   P1_EVAL=A   addendum A (random-split runs on their own test and on external)
#   P1_EVAL=E   addendum E (Satlas + DINOv3 runs on val, test, external, Roi Et zscene)
#   P1_EVAL=I   addendum I (DINOv3 RGB conditions b0, b2, b3 on the same four sets)
#   P1_EVAL=J   addendum J (locked ensemble, std and background-suppressed, val/test/external)
set -euo pipefail
module purge
module load anaconda3
source activate "${THEOS_CONDA_ENV:-theos}"
export PYTHONUNBUFFERED=1

P=/project/gisnorth/paper1
R=$P/analysis/results
cd "$P"
PRE="pre-registered in addendum_prereg_2026-10.md (commits 0c8511d, 096baba, 8c90c75, fe54b9b); read once"

case "${P1_EVAL:?set P1_EVAL=DC, C, A, E, I or J}" in
DC)
  STD=()
  for a in unetpp_effb3 segformer_b2 upernet_convnextt; do
    for s in 0 1 2 3 4; do STD+=("rerun_output/${a}__b1_rgb_aug__s${s}"); done
  done
  for split in val test external; do
    unlock=(); [[ $split == external ]] && unlock=(--unlock "addendum D: $PRE")
    python3 scripts/eval_background.py --split $split --runs "${STD[@]}" \
      --results-root rerun_output_ignbg "${unlock[@]}" \
      --out $R/background_eval_${split}.json
  done
  for v in raw zscene; do
    D=/project/gisnorth/paper1_roiet/$v
    python3 scripts/eval_rerun.py --results rerun_output --split external \
      --splits-file $R/roiet_splits.json --rgb-root $D --nir-root $D \
      --unlock "addendum C Roi Et ($v): $PRE" --out $R/rerun_eval_roiet_${v}.json
    python3 scripts/eval_ensemble.py --results rerun_output --split external \
      --splits-file $R/roiet_splits.json --data-root $D \
      --unlock "addendum C Roi Et ($v): $PRE" --out $R/ensemble_eval_roiet_${v}.json
  done
  ;;
A)
  python3 scripts/eval_rerun.py --results rerun_output_randomsplit --split test \
    --splits-file $R/random_splits.json --out $R/randomsplit_eval_test.json
  python3 scripts/eval_rerun.py --results rerun_output_randomsplit --split external \
    --splits-file $R/random_splits.json --unlock "addendum A: $PRE" \
    --out $R/randomsplit_eval_external.json
  ;;
C)
  # C alone: job 82304 (DC) finished D but hit its 12 h limit inside the Roi Et raw
  # pass (~2.2 min per run x 360 runs per variant) and wrote no C file. Same commands,
  # rerun on gpu-h100 with a longer limit:
  #   P1_EVAL=C sbatch -p gpu-h100 --time=2-00:00:00 hpc/p1_addendum_eval.slurm
  for v in ${P1_C_VARIANTS:-raw zscene}; do
    D=/project/gisnorth/paper1_roiet/$v
    python3 scripts/eval_rerun.py --results rerun_output --split external \
      --splits-file $R/roiet_splits.json --rgb-root $D --nir-root $D \
      --unlock "addendum C Roi Et ($v): $PRE" --out $R/rerun_eval_roiet_${v}.json
    python3 scripts/eval_ensemble.py --results rerun_output --split external \
      --splits-file $R/roiet_splits.json --data-root $D \
      --unlock "addendum C Roi Et ($v): $PRE" --out $R/ensemble_eval_roiet_${v}.json
  done
  ;;
E)
  for split in val test; do
    python3 scripts/eval_rerun.py --results rerun_output_foundation --split $split \
      --splits-file $R/spatial_splits.json --out $R/foundation_eval_${split}.json
  done
  python3 scripts/eval_rerun.py --results rerun_output_foundation --split external \
    --splits-file $R/spatial_splits.json --unlock "addendum E: $PRE" \
    --out $R/foundation_eval_external.json
  D=/project/gisnorth/paper1_roiet/zscene
  python3 scripts/eval_rerun.py --results rerun_output_foundation --split external \
    --splits-file $R/roiet_splits.json --rgb-root $D --nir-root $D \
    --unlock "addendum E Roi Et (zscene): $PRE" --out $R/foundation_eval_roiet_zscene.json
  ;;
I)
  # addendum I: DINOv3-SAT in b0, b2, b3 (prereg commit 0979fbe); the b1 runs of E are not re-read
  PRE_I="pre-registered in addendum_prereg_2026-10.md section I (commit 0979fbe); read once"
  for split in val test; do
    python3 scripts/eval_rerun.py --results rerun_output_dinov3rgb --split $split \
      --splits-file $R/spatial_splits.json --out $R/dinov3rgb_eval_${split}.json
  done
  python3 scripts/eval_rerun.py --results rerun_output_dinov3rgb --split external \
    --splits-file $R/spatial_splits.json --unlock "addendum I: $PRE_I" \
    --out $R/dinov3rgb_eval_external.json
  D=/project/gisnorth/paper1_roiet/zscene
  python3 scripts/eval_rerun.py --results rerun_output_dinov3rgb --split external \
    --splits-file $R/roiet_splits.json --rgb-root $D --nir-root $D \
    --unlock "addendum I Roi Et (zscene): $PRE_I" --out $R/dinov3rgb_eval_roiet_zscene.json
  ;;
J)
  # addendum J: six-class score of the locked ensemble (prereg commit 355eb1f)
  PRE_J="pre-registered in addendum_prereg_2026-10.md section J (commit 355eb1f); read once"
  for split in val test; do
    python3 scripts/eval_ensemble.py --results rerun_output --split $split \
      --background-modes --out $R/ensemble_bg_eval_${split}.json
  done
  python3 scripts/eval_ensemble.py --results rerun_output --split external \
    --background-modes --unlock "addendum J: $PRE_J" --out $R/ensemble_bg_eval_external.json
  ;;
*) echo "unknown P1_EVAL=$P1_EVAL"; exit 1 ;;
esac
echo "=== addendum eval $P1_EVAL done ==="
