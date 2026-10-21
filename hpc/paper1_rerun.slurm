#!/bin/bash
#SBATCH --job-name=p1-rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/project/gisnorth/paper1/slurm_logs/p1-rerun-%A_%a.out
#SBATCH --array=0-139

# paper1 re-run: full 2x2x2 factorial (bands x augmentation x class weighting)
# x 7 architectures x 5 seeds, on spatially valid splits.
#
# Every number in the first round is unusable: the split was random at tile level
# while tiles overlap by 64 px, so 100% of val/test tiles shared pixels with
# training tiles. This array reruns the study on spatial_splits.json, where that
# figure is zero and Scene 2 is held out entirely as an external test.
#
# One array task = one run. The task index selects a row of the task list, so the
# grid lives in a file that can be diffed and audited, not in bash arithmetic:
#   python paper1/scripts/build_rerun_tasklist.py --phase 1
#
# gpu QOS: 2 running / 3 submitted, cpu <= 4 per job. Submit through
# hpc/submit_queue_paper1_rerun.sh, never sbatch the whole array.

set -euo pipefail
module purge
module load anaconda3
source activate "${THEOS_CONDA_ENV:-theos}"

export PYTHONUNBUFFERED=1

P1_ROOT="${P1_ROOT:-/project/gisnorth/paper1}"
cd "$P1_ROOT"
mkdir -p slurm_logs

PHASE="${P1_PHASE:-1}"
TASKLIST="${P1_TASKLIST:-$P1_ROOT/analysis/results/rerun_tasklist_phase${PHASE}.json}"
SPLITS="${P1_SPLITS:-$P1_ROOT/analysis/results/spatial_splits.json}"
OUT_ROOT="${P1_OUT_ROOT:-$P1_ROOT/rerun_output}"

# RGB and RGB+NIR live in separate dataset trees: SegmentationDataset picks up the
# fourth band by looking for <tile>_nir.png next to the tile, and only the
# _nir_only tree carries those. Labels are byte-identical in both trees (md5
# checked), so the arm only changes which tile directory is read.
RGB_ROOT="${P1_RGB_ROOT:-/project/gisnorth/cmu_theos_ai_data_fixed}"
NIR_ROOT="${P1_NIR_ROOT:-/project/gisnorth/cmu_theos_ai_data_nir_only}"

for f in "$TASKLIST" "$SPLITS"; do
  [[ -f "$f" ]] || { echo "ABORT: missing $f"; exit 1; }
done

# Pull this task's row. Failing here beats training the wrong cell silently.
eval "$(python3 - "$TASKLIST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys, shlex
tasks = json.load(open(sys.argv[1]))["tasks"]
idx = int(sys.argv[2])
t = next((t for t in tasks if t["id"] == idx), None)
if t is None:
    sys.exit(f"no task with id {idx}")
out = {
    "RUN": t["run"], "MODEL": t["model"], "ENCODER": t["encoder"],
    "SEED": str(t["seed"]), "CW": t["class_weights"],
    "AUG": "--augment" if t["augment"] else "--no-augment",
    "NIR": "--use-nir" if t["use_nir"] else "",
    "IGNBG": "--ignore-background" if t.get("ignore_background") else "",
    "BBLR": f"--backbone-lr-mult {t['backbone_lr_mult']}" if t.get("backbone_lr_mult") else "",
    "NIRMODE": f"--nir-mode {t['nir_mode']}" if t.get("nir_mode") else "",
    "FREEZE": "--freeze-backbone" if t.get("freeze_backbone") else "",
}
for k, v in out.items():
    print(f"{k}={shlex.quote(v)}")
PY
)"

if [[ -n "$NIR" || -n "$NIRMODE" ]]; then
  DATA_ROOT="$NIR_ROOT"
else
  DATA_ROOT="$RGB_ROOT"
fi
TILES="$DATA_ROOT/tiles"
LABELS="$DATA_ROOT/semantic_labels"
[[ -d "$TILES" && -d "$LABELS" ]] || { echo "ABORT: missing $TILES or $LABELS"; exit 1; }

OUT_DIR="$OUT_ROOT/$RUN"
if [[ -f "$OUT_DIR/logs/${MODEL}_best_metrics.json" ]]; then
  echo "already complete, skipping: $RUN"
  exit 0
fi
mkdir -p "$OUT_DIR"

echo "=== paper1 re-run task=$SLURM_ARRAY_TASK_ID run=$RUN host=$(hostname) ==="
echo "    model=$MODEL encoder=$ENCODER seed=$SEED weights=$CW aug=$AUG nir=${NIR:-off} ${IGNBG}"
echo "    data=$DATA_ROOT"

python3 scripts/train_gpu.py \
  --model "$MODEL" --encoder "$ENCODER" --pretrained \
  --splits "$SPLITS" --seed "$SEED" \
  --tiles-dir "$TILES" --labels-dir "$LABELS" \
  --output-dir "$OUT_DIR" \
  --class-weights "$CW" $AUG $NIR $IGNBG $BBLR $NIRMODE $FREEZE \
  --epochs "${P1_EPOCHS:-200}" --patience 20 --batch-size 8 \
  --lr 1e-4 --weight-decay 1e-4 --optimizer adamw --loss combo --scheduler cosine

echo "=== done $RUN ==="
