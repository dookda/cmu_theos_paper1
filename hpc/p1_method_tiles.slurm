#!/bin/bash
#SBATCH --job-name=p1-tiles
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-tiles-%j.out
# Predictions of every method on four held-out tiles, for the visual comparison figure.
# Tiles: per held-out set, the two not used in Fig. 9 with the most mixed labelled classes.
# Not an evaluation: no score is computed and nothing is selected.
set -euo pipefail
module purge; module load anaconda3; source activate theos
export PYTHONUNBUFFERED=1
cd /project/gisnorth/paper1
R=rerun_output; F=rerun_output_foundation; I=rerun_output_ignbg
python3 scripts/predict_tiles.py --tiles tile_00705 tile_01830 tile_04327 tile_04317 \
  --runs $R/unet_mobilenet__b1_rgb_aug__s0 $R/unet_resnet50__b1_rgb_aug__s0 \
         $R/unet_densenet121__b1_rgb_aug__s0 $R/unet_effb3__b1_rgb_aug__s0 \
         $R/unetpp_effb3__b1_rgb_aug__s0 $R/deeplabv3_r101__b1_rgb_aug__s0 \
         $R/hrnet_w48__b1_rgb_aug__s0 $R/segformer_b2__b1_rgb_aug__s0 \
         $R/upernet_convnextt__b1_rgb_aug__s0 \
         $F/satlas_swinb__b1_rgb_aug__s0 $F/dinov3_vitl_sat__b1_rgb_aug__s0 \
         $I/segformer_b2__b1_rgb_aug_ignbg__s0 \
  --ensemble --out analysis/results/method_tiles.npz
