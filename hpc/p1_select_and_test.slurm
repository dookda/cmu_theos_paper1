#!/bin/bash
#SBATCH --job-name=p1-sel-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-sel-test-%j.out

# ขั้นที่ 2-4 ของการปิดงาน paper1 ต่อจาก p1_eval_val.slurm
#
# ลำดับนี้ห้ามสลับ: การเลือก ensemble เห็นเฉพาะ val แล้ว test จึงถูกอ่านครั้งเดียว
# ส่งด้วย --dependency=afterok:<jobid ของ eval val> เพื่อให้การเลือกเกิดหลัง val ครบ
set -uo pipefail
module purge; module load anaconda3; source activate theos
cd /project/gisnorth/paper1
export PYTHONUNBUFFERED=1
R=analysis/results
S=$R/spatial_splits.json

echo "########## 1/3 เลือก ensemble จาก val เท่านั้น (pool = seed 0)"
python3 scripts/build_ensemble.py --results rerun_output --cache ens_cache \
  --splits-file $S --pool '*__s0' --max-members 12
echo "exit=$?"

echo "########## 2/3 eval test — อ่านครั้งเดียวหลังตัดสินใจครบ"
python3 scripts/eval_rerun.py --results rerun_output --split test --splits-file $S
echo "exit=$?"

echo "########## 3/3 block bootstrap บน test (10 บล็อก)"
for pair in "b1_rgb_aug b0_rgb_plain" "b4_rgbnir_plain b0_rgb_plain" "b2_rgb_wt b0_rgb_plain"; do
  set -- $pair
  echo "--- $1 vs $2"
  python3 scripts/block_bootstrap_paper1.py --eval $R/rerun_eval_test.json \
    --a "*__$1__*" --b "*__$2__*"
done
echo "########## select+test done"
