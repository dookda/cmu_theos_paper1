#!/bin/bash
#SBATCH --job-name=p1-reeval-nir-rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-reeval-nir-rerun-%j.out
# Same correction as p1_reeval_nir_unified.slurm, for the per-run/per-block files that feed the
# factorial, block bootstrap and variance decomposition: NIR runs only, scored on the RGB masks.
# RGB runs are not re-read; their existing results are merged with these afterwards.
#   sbatch hpc/p1_reeval_nir_rerun.slurm
set -euo pipefail
module purge; module load anaconda3; source activate theos
export PYTHONUNBUFFERED=1
cd /project/gisnorth/paper1
R=analysis/results
L=/project/gisnorth/cmu_theos_ai_data_fixed
WHY="re-score of the NIR arm against the RGB reference masks (the NIR data root held the pre-fix masks; found 2026-10-21). Scoring labels only; no selection, checkpoint or metric changes."
for SPLIT in val test external; do
  python3 scripts/eval_rerun.py --results rerun_output --split $SPLIT --splits-file $R/spatial_splits.json \
    --nir-only --labels-root $L --unlock "$WHY" --out $R/rerun_eval_${SPLIT}_nir_rgblabels.json
done
for SPLIT in test external; do
  python3 scripts/eval_rerun.py --results rerun_output_randomsplit --split $SPLIT --splits-file $R/random_splits.json \
    --nir-only --labels-root $L --unlock "$WHY" --out $R/randomsplit_eval_${SPLIT}_nir_rgblabels.json
done
echo "=== NIR re-score (eval_rerun) done ==="
