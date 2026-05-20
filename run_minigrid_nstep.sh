#!/usr/bin/env bash
# NoisyDQN vs vanilla DQN with n-step returns on MiniGrid-DoorKey-6x6.
#
# Usage:
#   chmod +x run_minigrid_nstep.sh
#   ./run_minigrid_nstep.sh
#
# Override steps or n:
#   TOTAL_STEPS=200000 N_STEP=5 ./run_minigrid_nstep.sh

set -euo pipefail
cd "$(dirname "$0")"

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

TOTAL_STEPS=${TOTAL_STEPS:-2000000}
N_STEP=${N_STEP:-5}
ENV="MiniGrid-DoorKey-6x6-v0"

run() {
    local name=$1; shift
    local outdir="runs/$name"
    mkdir -p "$outdir"

    if [ -f "$outdir/log.npz" ]; then
        echo "==> skipping $name (already done)"
        return
    fi

    echo
    echo "=========================================="
    echo "==> starting: $name  (n-step=${N_STEP})"
    echo "=========================================="
    $PY train_minigrid.py \
        --env-id "$ENV" \
        --total-steps "$TOTAL_STEPS" \
        --n-step "$N_STEP" \
        --out-dir "$outdir" \
        "$@" 2>&1 | tee "$outdir/train.log"
    echo "==> done: $name"
}

run doorkey_nstep_noisy
run doorkey_nstep_baseline  --no-noisy

echo
echo "==> All jobs finished. Generate plots with:"
echo "    .venv/bin/python plot_minigrid.py"
