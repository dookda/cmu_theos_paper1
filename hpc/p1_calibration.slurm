#!/bin/bash
#SBATCH --job-name=p1-calib
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-calib-%j.out
# F5 (prereg 25eb5c1): calibration and risk-coverage of the locked ensemble.
set -euo pipefail
module purge; module load anaconda3; source activate theos
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd /project/gisnorth/paper1
python3 scripts/eval_calibration.py --out analysis/results/calibration.json
