#!/usr/bin/env bash
# Run all 4 Asterix DQN / DDQN variants sequentially — one at a time.
#
# Two implementations, each as noisy + baseline:
#   1. DQN  — standard 1-step TD target
#   2. DDQN — Double DQN target (online selects action, target evaluates it)
#
# Usage:
#   chmod +x run_sequential.sh
#   ./run_sequential.sh
#
# Smoke test with fewer steps:
#   TOTAL_STEPS=400000 ./run_sequential.sh
#
# Already-finished runs are skipped automatically (log.npz present).
# Each job writes to runs/<name>/:
#   train.log, rewards.png, sigma.png / epsilon.png, log.npz, model.pt

set -euo pipefail
cd "$(dirname "$0")"

# ---- 1. venv + deps (one-time) -------------------------------------------
if [ ! -d .venv ]; then
    echo "==> creating venv"
    python3 -m venv .venv
fi

PY=.venv/bin/python
$PY -m pip install --upgrade pip --quiet
$PY -m pip install -r requirements.txt --quiet

# ---- 2. Atari ROMs (one-time, idempotent) --------------------------------
echo "==> installing Atari ROMs (idempotent)"
.venv/bin/AutoROM --accept-license --quiet || true

mkdir -p runs
mkdir -p runs/.matplotlib
export MPLCONFIGDIR="$PWD/runs/.matplotlib"
export PYTHONUNBUFFERED=1

# ---- 3. knobs ------------------------------------------------------------
TOTAL_STEPS=${TOTAL_STEPS:-20000000}
BUFFER_CAPACITY=${BUFFER_CAPACITY:-500000}
SAVE_EVERY=${SAVE_EVERY:-200000}
ENV="ALE/Asterix-v5"

# ---- 4. helper: skip if already done, otherwise run and tee log ----------
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
    echo "==> starting: $name"
    echo "=========================================="
    $PY train_atari.py \
        --env-id "$ENV" --algo dqn \
        --buffer-capacity "$BUFFER_CAPACITY" \
        --total-steps "$TOTAL_STEPS" \
        --save-every "$SAVE_EVERY" \
        --out-dir "$outdir" \
        "$@" 2>&1 | tee "$outdir/train.log"
    echo "==> done: $name"
}

# ---- 5. the 4 jobs -------------------------------------------------------

# --- DQN (standard 1-step TD) ---
run asterix_base_noisy
run asterix_base_baseline  --no-noisy

# --- DDQN (double DQN target) ---
run asterix_ddqn_noisy     --double
run asterix_ddqn_baseline  --no-noisy --double

echo
echo "==> All jobs finished. Generate plots with:"
echo "    .venv/bin/python plot_asterix.py"
