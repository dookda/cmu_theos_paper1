#!/bin/bash
#SBATCH --job-name=p1-ext360
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-ext360-%j.out

# อ่าน external (Scene 2) ซ้ำให้ครบทั้ง 360 รัน
#
# hpc/external.slurm เดิมรันตอนแฟกทอเรียลยังมีแค่ 220 รัน ผลที่ได้จึงครอบคลุม
# 5 สถาปัตยกรรมเต็ม ส่วนอีก 4 ตัว (deeplabv3_r101, unet_resnet50, unet_densenet121,
# unet_mobilenet) มีแค่ b0_rgb_plain 5 seed เท่านั้น ทำให้ Fig. 6 เอาแท่งที่มาจาก
# 40 รัน 8 เงื่อนไข ไปวางเทียบกับแท่งที่มาจาก 5 รันเงื่อนไขเดียว
#
# ตอนนี้แฟกทอเรียลครบ 360/360 แล้ว จึงอ่านซ้ำครั้งเดียวให้ทุกสถาปัตยกรรมถูกให้คะแนน
# บนเงื่อนไขชุดเดียวกัน — ไม่มีการเลือกอะไรใหม่ ไม่เปลี่ยนเมตริก
#
# --time=16:00:00 เพราะ eval หนึ่งรันใช้ ~100 วิ; 360 รัน ~10 ชม. (ของเดิม 220 รัน
# ตั้งไว้ 12 ชม.) และ eval_rerun.py เขียนไฟล์ครั้งเดียวตอนจบ ถูกตัดกลางทาง = ผลหายหมด
set -uo pipefail
module purge; module load anaconda3; source activate theos
cd /project/gisnorth/paper1
export PYTHONUNBUFFERED=1
R=analysis/results
STAMP=$(date +%Y%m%d_%H%M)

echo "########## 0/2 สำรองผล external ชุด 220 รันไว้ก่อน"
# block_bootstrap_paper1.py เขียนแบบ append ถ้าไม่ย้ายของเก่าออก จะได้ 6 รายการ
# โดยสามอันแรกเป็นเลขของยุค 220 รัน ปนอยู่กับของใหม่แบบแยกไม่ออก
for f in rerun_eval_external.json block_bootstrap_external_mean_iou.json; do
  [[ -f $R/$f ]] && mv -v "$R/$f" "$R/${f%.json}.220runs_$STAMP.json"
done

echo "########## 1/2 eval external ครบ 360 รัน"
python3 scripts/eval_rerun.py --results rerun_output --split external \
  --splits-file $R/spatial_splits.json \
  --unlock "selection frozen 2026-08-29: factorial complete 360/360; ensemble re-selected on val alone returned the same 7 members chosen on 2026-08-21 (val mIoU 0.6016); test read once on 2026-08-28. This re-read covers the 140 runs added after the 2026-08-21 external read so that all nine architectures are scored on the same eight conditions. No selection or metric changes."
rc=$?
echo "exit=$rc"
if (( rc != 0 )); then
  echo "ABORT: eval external ล้มเหลว — คืนไฟล์เดิมกลับ"
  for f in rerun_eval_external.json block_bootstrap_external_mean_iou.json; do
    [[ -f $R/${f%.json}.220runs_$STAMP.json ]] && mv -v "$R/${f%.json}.220runs_$STAMP.json" "$R/$f"
  done
  exit 1
fi

echo "########## 2/2 block bootstrap บน external — คู่เดียวกับที่ทำบน test"
for pair in "b1_rgb_aug b0_rgb_plain" "b4_rgbnir_plain b0_rgb_plain" "b2_rgb_wt b0_rgb_plain"; do
  set -- $pair
  echo "--- $1 vs $2"
  python3 scripts/block_bootstrap_paper1.py --eval $R/rerun_eval_external.json \
    --a "*__$1__*" --b "*__$2__*"
done
echo "########## external 360 done"
