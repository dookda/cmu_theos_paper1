#!/bin/bash
#SBATCH --job-name=p1-eval-val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-eval-val-%j.out

# ขั้นที่ 1 ของการปิดงาน paper1 แยกออกมาเป็นงานของตัวเอง
#
# finalize.slurm เดิมทำสี่ขั้นในงานเดียวภายใต้ --time=20:00:00 ซึ่งพอเป็นแฟกทอเรียล
# เต็ม 360 รันก็ไม่พอ: eval หนึ่งรันใช้ ~100 วินาที ดังนั้น val อย่างเดียว ~10 ชม.
# และ test อีก ~10 ชม. งานจะโดนตัดกลางขั้นที่ 3 แล้วผล test หายทั้งก้อน เพราะ
# eval_rerun.py เขียนไฟล์ผลครั้งเดียวตอนจบ ไม่มีการทำต่อจากของเดิม
set -uo pipefail
module purge; module load anaconda3; source activate theos
cd /project/gisnorth/paper1
export PYTHONUNBUFFERED=1

python3 scripts/eval_rerun.py --results rerun_output --split val \
  --splits-file analysis/results/spatial_splits.json
echo "exit=$?"
