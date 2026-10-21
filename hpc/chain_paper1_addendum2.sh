#!/bin/bash
# Paper 1 addendum queue, round 2: C/D evaluation first, then the remaining A training
# runs, then the A evaluation once no p1-rerun job is left. One submitter at a time
# (3-submitted QoS); a rejected sbatch is retried, never skipped.
#   setsid nohup bash hpc/chain_paper1_addendum2.sh > hpc_output/chain_paper1_addendum2.log 2>&1 &
set -uo pipefail
P=/project/gisnorth/paper1
R=$P/analysis/results
cd "$P"

submit_when_free() {   # $1 = P1_EVAL value
  while true; do
    n=$(squeue -u "$(whoami)" -h -t pending,running -o "%A" | wc -l)
    if (( n < 3 )); then
      jid=$(P1_EVAL="$1" sbatch --parsable hpc/p1_addendum_eval.slurm) && {
        echo "$(date --iso-8601=seconds) eval $1 submitted: job $jid"; return 0; }
      echo "$(date --iso-8601=seconds) sbatch eval $1 rejected, retrying"
    fi
    sleep 120
  done
}

submit_when_free DC

P1_INTENDED_LEAK=1 P1_PHASE=A P1_TASKLIST=$R/rerun_tasklist_randomsplit.json \
  P1_SPLITS=$R/random_splits.json P1_OUT_ROOT=$P/rerun_output_randomsplit \
  bash hpc/submit_queue_paper1_rerun.sh || exit 1

while squeue -u "$(whoami)" -h -n p1-rerun -o "%A" | grep -q .; do sleep 300; done
done_n=$(ls $P/rerun_output_randomsplit/*/logs/*_best_metrics.json 2>/dev/null | wc -l)
echo "$(date --iso-8601=seconds) A training finished: $done_n/60 runs"
(( done_n == 60 )) || { echo "ABORT: A incomplete, not evaluating"; exit 1; }
submit_when_free A
echo "addendum chain 2: all submitted"
