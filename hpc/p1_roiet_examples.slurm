#!/bin/bash
#SBATCH --job-name=p1-roiet-fig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-roiet-fig-%j.out
# Roi Et figure: ensemble prediction maps of four fixed-rule tiles (not an evaluation).
set -euo pipefail
module purge; module load anaconda3; source activate theos
cd /project/gisnorth/paper1
python3 scripts/predict_roiet_examples.py
