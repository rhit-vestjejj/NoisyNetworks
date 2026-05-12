#!/usr/bin/env bash
# Bare-metal version of run_all.sh — no Docker required.
# Creates a venv, installs deps, downloads Atari ROMs, then launches all 6
# paper variants as detached background processes (nohup).
#
# Each job writes:
#   runs/<name>/train.log   — stdout + stderr
#   runs/<name>/pid         — the process PID
#   runs/<name>/{rewards,sigma,epsilon}.png + log.npz + model.pt as training progresses

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

# ---- 3. launch helper ----------------------------------------------------
run() {
    local name=$1; shift
    local outdir="runs/$name"
    mkdir -p "$outdir"

    if [ -f "$outdir/pid" ] && kill -0 "$(cat "$outdir/pid")" 2>/dev/null; then
        echo "==> $name already running (PID $(cat "$outdir/pid")), skipping"
        return
    fi

    echo "==> launching $name"
    nohup $PY "$@" --out-dir "$outdir" \
        > "$outdir/train.log" 2>&1 &
    echo $! > "$outdir/pid"
    echo "    PID $(cat "$outdir/pid")"
}

# ---- 4. the 6 paper variants ---------------------------------------------
# DQN family on Breakout
run breakout_dqn_noisy    train_atari.py --algo dqn  --env-id ALE/Breakout-v5
run breakout_dqn_baseline train_atari.py --algo dqn  --no-noisy --env-id ALE/Breakout-v5

# Dueling family on Asteroids (paper highlights super-human result)
run asteroids_dueling_noisy    train_atari.py --algo dueling           --env-id ALE/Asteroids-v5
run asteroids_dueling_baseline train_atari.py --algo dueling --no-noisy --env-id ALE/Asteroids-v5

# A3C family on Beam Rider (paper highlights super-human result)
run beamrider_a3c_noisy    train_a3c.py            --env-id ALE/BeamRider-v5 --total-steps 320000000
run beamrider_a3c_baseline train_a3c.py --no-noisy --env-id ALE/BeamRider-v5 --total-steps 320000000

echo
echo "==> 6 jobs launched. Monitor with:"
echo "    tail -f runs/breakout_dqn_noisy/train.log"
echo "    ls runs/"
echo "    for f in runs/*/pid; do d=\$(dirname \$f); p=\$(cat \$f); kill -0 \$p 2>/dev/null && echo \"\$d alive (PID \$p)\" || echo \"\$d DEAD\"; done"
echo
echo "Stop everything:"
echo "    for f in runs/*/pid; do kill \$(cat \$f) 2>/dev/null; done"
