#!/bin/bash
# Paper 1 addendum queue, round 4 (replaces round 3 on 2026-10-09: round 3 counted
# jobs of every partition against the gpu QoS and stalled once C ran on gpu-h100).
# Waits for any p1-rerun job still running so its task is not submitted twice.
# E training (10 runs), the remaining A runs, then the A and E evaluations once no
# p1-rerun job is left. One submitter at a time (3-submitted QoS).
#   setsid nohup bash hpc/chain_paper1_addendum4.sh > hpc_output/chain_paper1_addendum4.log 2>&1 &
set -uo pipefail
P=/project/gisnorth/paper1
R=$P/analysis/results
cd "$P"

submit_when_free() {   # $1 = P1_EVAL value
  while true; do
    n=$(squeue -u "$(whoami)" -h -p gpu -t pending,running -o "%A" | wc -l)
    if (( n < 3 )); then
      jid=$(P1_EVAL="$1" sbatch --parsable hpc/p1_addendum_eval.slurm) && {
        echo "$(date --iso-8601=seconds) eval $1 submitted: job $jid"; return 0; }
      echo "$(date --iso-8601=seconds) sbatch eval $1 rejected, retrying"
    fi
    sleep 120
  done
}

while squeue -u "$(whoami)" -h -n p1-rerun -o "%A" | grep -q .; do sleep 120; done
echo "$(date --iso-8601=seconds) no p1-rerun job left; starting E"

P1_PHASE=E P1_TASKLIST=$R/rerun_tasklist_foundation.json \
  P1_SPLITS=$R/spatial_splits.json P1_OUT_ROOT=$P/rerun_output_foundation \
  bash hpc/submit_queue_paper1_rerun.sh || exit 1

P1_INTENDED_LEAK=1 P1_PHASE=A P1_TASKLIST=$R/rerun_tasklist_randomsplit.json \
  P1_SPLITS=$R/random_splits.json P1_OUT_ROOT=$P/rerun_output_randomsplit \
  bash hpc/submit_queue_paper1_rerun.sh || exit 1

while squeue -u "$(whoami)" -h -n p1-rerun -o "%A" | grep -q .; do sleep 300; done
for spec in E:rerun_output_foundation:10 A:rerun_output_randomsplit:60; do
  IFS=: read -r mode dir need <<< "$spec"
  n=$(ls $P/$dir/*/logs/*_best_metrics.json 2>/dev/null | wc -l)
  echo "$(date --iso-8601=seconds) $mode training: $n/$need runs"
  if (( n == need )); then submit_when_free "$mode"; else echo "  $mode incomplete, not evaluating"; fi
done
echo "addendum chain 4: done"
