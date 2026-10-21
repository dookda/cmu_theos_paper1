#!/bin/bash
#SBATCH --job-name=p1-unif-dev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-unif-dev-%j.out
# Dev check of eval_unified.py on VAL only, into a scratch folder (not the F pass):
# must reproduce rerun_eval_val.json mIoU for the same runs.
set -euo pipefail
module purge; module load anaconda3; source activate theos
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd /project/gisnorth/paper1
rm -rf /project/gisnorth/paper1/unified_dev
python3 scripts/eval_unified.py --roots rerun_output --splits val --limit 1 \
  --out /project/gisnorth/paper1/unified_dev
mkdir -p dev_one && rm -f dev_one/* && ln -sf ../rerun_output/deeplabv3_r101__b4_rgbnir_plain__s0 dev_one/
python3 scripts/eval_unified.py --roots dev_one --splits val --out /project/gisnorth/paper1/unified_dev
