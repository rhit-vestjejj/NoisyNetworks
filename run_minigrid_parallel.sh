#!/usr/bin/env bash
# Run all 6 MiniGrid-DoorKey-6x6 variants in parallel.
#
# Usage:
#   chmod +x run_minigrid_parallel.sh
#   ./run_minigrid_parallel.sh
#
# Smoke test with fewer steps:
#   TOTAL_STEPS=20000 ./run_minigrid_parallel.sh
#
# Already-finished runs are skipped automatically (log.npz present).
# Each job writes to runs/<name>/:
#   train.log, rewards.png, sigma.png / epsilon.png, log.npz, model.pt

set -uo pipefail
cd "$(dirname "$0")"

# ---- 1. venv + deps (one-time) -------------------------------------------
if [ ! -d .venv ]; then
    echo "==> creating venv"
    python3 -m venv .venv
fi

PY=.venv/bin/python
$PY -m pip install --upgrade pip --quiet
$PY -m pip install -r requirements.txt --quiet
$PY -m pip install minigrid --quiet

mkdir -p runs
mkdir -p runs/.matplotlib
export MPLCONFIGDIR="$PWD/runs/.matplotlib"
export PYTHONUNBUFFERED=1

# ---- 2. knobs ------------------------------------------------------------
TOTAL_STEPS=${TOTAL_STEPS:-2000000}
ENV="MiniGrid-DoorKey-6x6-v0"

# ---- 3. helper -----------------------------------------------------------
# Launches one training job in the background.
# Returns immediately; caller collects PIDs via $PIDS array.
launch() {
    local name=$1; shift
    local outdir="runs/$name"
    mkdir -p "$outdir"

    if [ -f "$outdir/log.npz" ]; then
        echo "==> skipping $name (already done)"
        return
    fi

    echo "==> launching: $name"
    $PY train_minigrid.py \
        --env-id "$ENV" \
        --total-steps "$TOTAL_STEPS" \
        --out-dir "$outdir" \
        "$@" 2>&1 | tee "$outdir/train.log" &
    PIDS+=($!)
    NAMES+=("$name")
}

# ---- 4. launch all jobs --------------------------------------------------
PIDS=()
NAMES=()

launch doorkey_noisy
launch doorkey_baseline       --no-noisy
launch doorkey_ddqn_noisy     --double
launch doorkey_ddqn_baseline  --double --no-noisy
launch doorkey_per_noisy      --per
launch doorkey_per_baseline   --per --no-noisy

if [ ${#PIDS[@]} -eq 0 ]; then
    echo "==> All runs already done."
    exit 0
fi

echo
echo "==> ${#PIDS[@]} job(s) running. Waiting for all to finish..."
echo "    Follow any run with: tail -f runs/<name>/train.log"
echo

# ---- 5. wait and report --------------------------------------------------
FAILED=()
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${NAMES[$i]}
    if wait "$pid"; then
        echo "==> done:   $name"
    else
        echo "==> FAILED: $name (exit $?)"
        FAILED+=("$name")
    fi
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "==> All jobs finished successfully. Generate plots with:"
    echo "    $PY plot_minigrid.py"
else
    echo "==> ${#FAILED[@]} job(s) failed: ${FAILED[*]}"
    echo "    Check runs/<name>/train.log for details."
    exit 1
fi
