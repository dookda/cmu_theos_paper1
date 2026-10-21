#!/bin/bash
# Self-throttling submitter for the paper1 re-run factorial.
# gpu QOS allows 3 submitted / 2 running, so feed tasks one at a time and skip any
# run that already produced best_metrics.json. Refuses to start unless the splits
# file passes its leakage audit -- retraining on leaky splits is the one outcome
# this whole re-run exists to avoid.
#
#   P1_PHASE=1 nohup bash hpc/submit_queue_paper1_rerun.sh \
#     > hpc_output/submit_queue_paper1_rerun.log 2>&1 &
set -uo pipefail

P1_ROOT="${P1_ROOT:-/project/gisnorth/paper1}"
PHASE="${P1_PHASE:-1}"
TASKLIST="${P1_TASKLIST:-$P1_ROOT/analysis/results/rerun_tasklist_phase${PHASE}.json}"
SPLITS="${P1_SPLITS:-$P1_ROOT/analysis/results/spatial_splits.json}"
OUT_ROOT="${P1_OUT_ROOT:-$P1_ROOT/rerun_output}"
SLURM_SCRIPT="${P1_SLURM:-$P1_ROOT/hpc/paper1_rerun.slurm}"
MAX_SUBMIT=3
# A run needs about 1.5 GB for its two checkpoints. Stop well before that runs out:
# on 2026-08-27 the project filesystem filled, every job then died in one second,
# the throttle saw an empty queue and burned the remaining 52 tasks in six minutes.
MIN_FREE_GB="${P1_MIN_FREE_GB:-20}"
MAX_QUICK_FAILS="${P1_MAX_QUICK_FAILS:-3}"
QUICK_FAIL_SEC="${P1_QUICK_FAIL_SEC:-120}"

[[ -f "$TASKLIST" ]] || { echo "ABORT: missing $TASKLIST — run build_rerun_tasklist.py"; exit 1; }

# Gate: the splits file must report zero evaluation tiles touching train.
# Exception: addendum A (random_splits.json) leaks val/test on purpose to measure the
# inflation. It must say so in its "design" field and P1_INTENDED_LEAK=1 must be set;
# its external set still has to be clean, since it is the common yardstick.
python3 - "$SPLITS" "${P1_INTENDED_LEAK:-0}" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
if sys.argv[2] == "1":
    if not d.get("design", "").startswith("random"):
        sys.exit("ABORT: P1_INTENDED_LEAK=1 but splits file is not the random-split design")
    ext = d["leakage_audit"]["external"]
    if ext["touching_train"]:
        sys.exit(f"ABORT: external touches train: {ext}")
    print(f"intended-leak design; external clean ({ext['tiles']} tiles)")
    sys.exit(0)
a = d["leakage_audit"]["after_buffer"]
bad = {k: v for k, v in a.items() if v["touching_train"]}
if bad:
    sys.exit(f"ABORT: splits still leak: {bad}")
print("splits leakage audit clean: " +
      ", ".join(f"{k}={v['tiles']}" for k, v in a.items()))
PY

mapfile -t pending < <(python3 - "$TASKLIST" "$OUT_ROOT" <<'PY'
import json, sys, pathlib
tasks = json.load(open(sys.argv[1]))["tasks"]
root = pathlib.Path(sys.argv[2])
for t in tasks:
    done = list((root / t["run"] / "logs").glob("*_best_metrics.json"))
    if not done:
        print(f'{t["id"]}:{t["run"]}')
PY
)

free_gb() { df -BG --output=avail "$OUT_ROOT" | tail -1 | tr -dc '0-9'; }

# Jobs submitted but not yet accounted for. A job that fails within QUICK_FAIL_SEC
# never trained anything, so a run of them means the cluster is refusing the work.
watch=()
quick_fails=0
reap() {
  local keep=() id state elapsed
  for id in ${watch[@]+"${watch[@]}"}; do
    read -r state elapsed < <(sacct -j "$id" -X -n -o State,ElapsedRaw | head -1)
    case "$state" in
      ""|PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED) keep+=("$id") ;;
      COMPLETED) quick_fails=0 ;;
      *) if (( ${elapsed:-0} < QUICK_FAIL_SEC )); then
           quick_fails=$((quick_fails + 1))
           echo "$(date --iso-8601=seconds) job $id ended $state after ${elapsed}s"
         else quick_fails=0; fi ;;
    esac
  done
  watch=(${keep[@]+"${keep[@]}"})
}

echo "phase $PHASE — pending runs: ${#pending[@]}, $(free_gb)G free on $OUT_ROOT"
for item in "${pending[@]}"; do
  task="${item%%:*}"
  while true; do
    # QoS limits are per partition: count only gpu-partition jobs, so CPU jobs of
    # other projects and gpu-h100 evaluations do not stall this queue.
    n=$(squeue -u "$(whoami)" -h -p gpu -t pending,running -o "%A" | wc -l)
    (( n < MAX_SUBMIT )) && break
    sleep 120
  done

  reap
  if (( quick_fails >= MAX_QUICK_FAILS )); then
    echo "ABORT: $quick_fails jobs in a row ended within ${QUICK_FAIL_SEC}s — the cluster"
    echo "       is not running this work. Fix the cause before resubmitting;"
    echo "       already-finished runs are skipped when you start this again."
    exit 1
  fi
  avail=$(free_gb)
  if (( avail < MIN_FREE_GB )); then
    echo "ABORT: only ${avail}G free on $OUT_ROOT (need ${MIN_FREE_GB}G) — free space first."
    exit 1
  fi

  echo "$(date --iso-8601=seconds) submitting task=$task ${item##*:} (${avail}G free)"
  jid=$(P1_PHASE="$PHASE" P1_ROOT="$P1_ROOT" sbatch --parsable --array="$task" "$SLURM_SCRIPT")
  echo "  job $jid"
  [[ -n "$jid" ]] && watch+=("${jid%%;*}")
  sleep 5
done
echo "phase $PHASE: all runs submitted"
