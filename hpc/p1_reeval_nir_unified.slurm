#!/bin/bash
#SBATCH --job-name=p1-reeval-nir-unified
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-reeval-nir-unified-%j.out
# Re-score every run that reads the NIR data root (main factorial, random split, G/H) against the
# RGB reference masks. Found 2026-10-21: /project/gisnorth/cmu_theos_ai_data_nir_only/semantic_labels
# holds the pre-fix masks (bare soil -> background in 463 of 2642 tiles), so the NIR arm was trained
# and scored on a different reference. This pass changes only the scoring labels; checkpoints and
# predictions are unchanged. Output goes to a separate folder; the original unified/ is kept.
#   sbatch hpc/p1_reeval_nir_unified.slurm
set -euo pipefail
module purge; module load anaconda3; source activate theos
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd /project/gisnorth/paper1
python3 scripts/eval_unified.py \
  --roots rerun_output rerun_output_randomsplit rerun_output_gh \
  --nir-only --labels-root /project/gisnorth/cmu_theos_ai_data_fixed \
  --out /project/gisnorth/paper1/analysis/results/unified_rgblabels
echo "=== NIR re-score (unified) done ==="
