#!/usr/bin/env bash
# Build the image once, launch all 6 paper variants in parallel as detached
# Docker containers. Each writes its logs and plots to runs/<name>/ on the host.
#
# Usage (on the server):
#   chmod +x run_all.sh
#   ./run_all.sh
#
# After launch, monitor with:
#   docker ps --filter 'name=noisynet-'
#   docker logs -f noisynet-breakout-dqn-noisy   # tail one job
#   ls runs/                                      # plots + checkpoints

set -euo pipefail
cd "$(dirname "$0")"

echo "==> Building docker image (one-time, ~5 min)"
docker build -t noisynet .

mkdir -p runs

run() {
    local name=$1; shift
    echo "==> launching $name"
    docker run --gpus all -d --rm \
        -v "$PWD/runs:/app/runs" \
        --name "noisynet-$name" \
        noisynet "$@"
}

# ---- DQN family (Breakout) ----------------------------------------------
run breakout-dqn-noisy \
    python train_atari.py --algo dqn \
    --env-id ALE/Breakout-v5 --out-dir runs/breakout_dqn_noisy

run breakout-dqn-baseline \
    python train_atari.py --algo dqn --no-noisy \
    --env-id ALE/Breakout-v5 --out-dir runs/breakout_dqn_baseline

# ---- Dueling family (Asteroids — paper highlights super-human result) ----
run asteroids-dueling-noisy \
    python train_atari.py --algo dueling \
    --env-id ALE/Asteroids-v5 --out-dir runs/asteroids_dueling_noisy

run asteroids-dueling-baseline \
    python train_atari.py --algo dueling --no-noisy \
    --env-id ALE/Asteroids-v5 --out-dir runs/asteroids_dueling_baseline

# ---- A3C family (Beam Rider — paper highlights super-human result) -------
run beamrider-a3c-noisy \
    python train_a3c.py --env-id ALE/BeamRider-v5 \
    --total-steps 320000000 --out-dir runs/beamrider_a3c_noisy

run beamrider-a3c-baseline \
    python train_a3c.py --no-noisy --env-id ALE/BeamRider-v5 \
    --total-steps 320000000 --out-dir runs/beamrider_a3c_baseline

echo
echo "==> 6 jobs launched. Track them with:"
echo "    docker ps --filter 'name=noisynet-'"
echo "    docker logs -f noisynet-breakout-dqn-noisy"
echo "    ls runs/"
