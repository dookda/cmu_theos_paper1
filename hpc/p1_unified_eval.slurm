#!/bin/bash
#SBATCH --job-name=p1-unified
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-unified-%j.out
# Paper 1 unified evaluation pass (prereg F, commit 25eb5c1): one read per checkpoint.
# Skips runs that already have an output, so the second pass (after G/H and A finish)
# covers only the new checkpoints.
#   sbatch hpc/p1_unified_eval.slurm
set -euo pipefail
module purge; module load anaconda3; source activate theos
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd /project/gisnorth/paper1
python3 scripts/eval_unified.py \
  --roots rerun_output rerun_output_ignbg rerun_output_foundation rerun_output_randomsplit rerun_output_gh \
  --roiet-roots rerun_output rerun_output_gh \
  --out /project/gisnorth/paper1/analysis/results/unified
echo "=== unified pass done ==="
