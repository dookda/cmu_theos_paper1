#!/bin/bash
# Paper 1 addendum queue: D (15 runs, background ignored) then A (60 runs, random split).
# One submitter at a time: two throttles reading the same squeue would race for the
# 3-submitted QoS slot and the loser's task would be skipped.
#   nohup bash hpc/chain_paper1_addendum.sh > hpc_output/chain_paper1_addendum.log 2>&1 &
set -uo pipefail
R=/project/gisnorth/paper1
cd "$R"
R_RES=$R/analysis/results

P1_PHASE=D P1_TASKLIST=$R_RES/rerun_tasklist_ignbg.json \
  P1_SPLITS=$R_RES/spatial_splits.json P1_OUT_ROOT=$R/rerun_output_ignbg \
  bash hpc/submit_queue_paper1_rerun.sh || exit 1

P1_INTENDED_LEAK=1 P1_PHASE=A P1_TASKLIST=$R_RES/rerun_tasklist_randomsplit.json \
  P1_SPLITS=$R_RES/random_splits.json P1_OUT_ROOT=$R/rerun_output_randomsplit \
  bash hpc/submit_queue_paper1_rerun.sh || exit 1
echo "addendum chain: all submitted"
