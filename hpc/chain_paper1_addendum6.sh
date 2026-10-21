#!/bin/bash
# Paper 1 addendum I (prereg section I): 15 DINOv3-SAT runs in the RGB conditions b0, b2
# and b3, then their evaluation. Checkpoints are 3.6 GB each (7.2 GB with the last-epoch
# file), more than the gisnorth quota has left, so the output root lives in $HOME and is
# symlinked into the project tree; evaluation reads it through the symlink.
#   setsid nohup bash hpc/chain_paper1_addendum6.sh > hpc_output/chain_paper1_addendum6.log 2>&1 &
set -uo pipefail
P=/project/gisnorth/paper1
R=$P/analysis/results
cd "$P"
STORE=$HOME/paper1_runs/rerun_output_dinov3rgb
mkdir -p "$STORE"
[[ -e $P/rerun_output_dinov3rgb ]] || ln -s "$STORE" $P/rerun_output_dinov3rgb

export SBATCH_JOB_NAME=p1-i
for pass in 1 2 3; do
  P1_PHASE=I P1_TASKLIST=$R/rerun_tasklist_dinov3rgb.json \
    P1_SPLITS=$R/spatial_splits.json P1_OUT_ROOT=$P/rerun_output_dinov3rgb \
    bash hpc/submit_queue_paper1_rerun.sh || exit 1
  while squeue -u "$(whoami)" -h -n p1-i -o "%A" | grep -q .; do sleep 300; done
  n=$(ls $P/rerun_output_dinov3rgb/*/logs/*_best_metrics.json 2>/dev/null | wc -l)
  echo "$(date --iso-8601=seconds) pass $pass: $n/15 addendum I runs complete"
  (( n == 15 )) && break
done
if (( n == 15 )); then
  unset SBATCH_JOB_NAME
  until jid=$(P1_EVAL=I sbatch --parsable -p gpu-h100 hpc/p1_addendum_eval.slurm); do sleep 300; done
  echo "$(date --iso-8601=seconds) eval I submitted: job $jid"
else
  echo "addendum I incomplete ($n/15), not evaluating"
fi
echo "addendum chain 6: done"
