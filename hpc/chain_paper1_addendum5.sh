#!/bin/bash
# Paper 1 addenda G and H (prereg 25eb5c1): 45 runs, started only after chain 4 has
# submitted every A run, so only one submitter feeds the gpu QoS at a time. Jobs carry
# the name p1-gh so chain 4's "wait for p1-rerun" before its E/A evaluation does not
# wait on them. Re-runs the submitter until every task has a result (a rejected sbatch
# leaves its task for the next pass).
#   setsid nohup bash hpc/chain_paper1_addendum5.sh > hpc_output/chain_paper1_addendum5.log 2>&1 &
set -uo pipefail
P=/project/gisnorth/paper1
R=$P/analysis/results
cd "$P"

until grep -q "phase A: all runs submitted" hpc_output/chain_paper1_addendum4.log 2>/dev/null; do
  sleep 300
done
echo "$(date --iso-8601=seconds) chain 4 finished submitting A; starting G/H"

export SBATCH_JOB_NAME=p1-gh
for pass in 1 2 3; do
  P1_PHASE=GH P1_TASKLIST=$R/rerun_tasklist_gh.json \
    P1_SPLITS=$R/spatial_splits.json P1_OUT_ROOT=$P/rerun_output_gh \
    bash hpc/submit_queue_paper1_rerun.sh || exit 1
  while squeue -u "$(whoami)" -h -n p1-gh -o "%A" | grep -q .; do sleep 300; done
  n=$(ls $P/rerun_output_gh/*/logs/*_best_metrics.json 2>/dev/null | wc -l)
  echo "$(date --iso-8601=seconds) pass $pass: $n/45 G/H runs complete"
  (( n == 45 )) && break
done
echo "addendum chain 5: done"
