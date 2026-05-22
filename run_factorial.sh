#!/usr/bin/env bash
# Run the 2x2 factorial {NoisyNet, eps-greedy} x {Uniform, PER} replay across
# multiple MiniGrid envs and seeds. This is the experimental backbone for the
# NoisyNet x PER interaction workshop paper.
#
# Output layout:
#   runs/factorial/<env-short>_<cond>_seed<N>/
#     - log.npz       (training curves + per-layer sigma + TD-error stats)
#     - per_diag.npz  (PER sample counts per buffer slot; PER runs only)
#     - args.json     (full hyperparameter snapshot)
#     - model.pt, rewards.png, sigma.png/epsilon.png
#
# Already-finished runs (log.npz present) are skipped. Re-running this script
# is safe and resumes where it left off.
#
# Usage:
#   chmod +x run_factorial.sh
#   ./run_factorial.sh
#
# Knobs:
#   SEEDS="0 1 2 3 4"    space-separated seed list
#   ENVS="empty8 doorkey6 multiroom"   which env shortnames to run
#   PARALLEL=1           number of concurrent jobs (default 1 = sequential)
#   SMOKE=1              tiny run for plumbing-check (overrides step counts)

set -uo pipefail
cd "$(dirname "$0")"

# ---- venv ----------------------------------------------------------------
if [ ! -d .venv ]; then
    echo "==> creating venv"
    python3 -m venv .venv
fi
PY=.venv/bin/python
$PY -m pip install --upgrade pip --quiet
$PY -m pip install -r requirements.txt --quiet
$PY -m pip install minigrid --quiet

mkdir -p runs/factorial runs/.matplotlib
export MPLCONFIGDIR="$PWD/runs/.matplotlib"
export PYTHONUNBUFFERED=1

# ---- experiment config ---------------------------------------------------
SEEDS=${SEEDS:-"0 1 2 3 4"}
ENVS=${ENVS:-"empty8 doorkey6 multiroom"}
PARALLEL=${PARALLEL:-1}
SMOKE=${SMOKE:-0}

# env shortname -> (gym-id, total_steps)
env_id() {
    case "$1" in
        empty8)    echo "MiniGrid-Empty-8x8-v0" ;;
        doorkey6)  echo "MiniGrid-DoorKey-6x6-v0" ;;
        doorkey8)  echo "MiniGrid-DoorKey-8x8-v0" ;;
        multiroom) echo "MiniGrid-MultiRoom-N2-S4-v0" ;;
        keycorr)   echo "MiniGrid-KeyCorridorS3R1-v0" ;;
        *) echo "UNKNOWN_ENV:$1"; return 1 ;;
    esac
}
env_steps() {
    if [ "$SMOKE" = "1" ]; then echo 20000; return; fi
    case "$1" in
        empty8)    echo 500000 ;;
        doorkey6)  echo 2000000 ;;
        doorkey8)  echo 3000000 ;;
        multiroom) echo 3000000 ;;
        keycorr)   echo 3000000 ;;
        *) echo 2000000 ;;
    esac
}

# Condition shortname -> flags for train_minigrid.py
cond_flags() {
    case "$1" in
        noisy_uniform) echo "--noisy" ;;
        noisy_per)     echo "--noisy --per" ;;
        eps_uniform)   echo "--no-noisy" ;;
        eps_per)       echo "--no-noisy --per" ;;
        *) echo "UNKNOWN_COND:$1"; return 1 ;;
    esac
}
CONDS="noisy_uniform noisy_per eps_uniform eps_per"

# ---- job launcher --------------------------------------------------------
launch_one() {
    local env_short=$1 cond=$2 seed=$3
    local gym_id total_steps flags outdir
    gym_id=$(env_id "$env_short")
    total_steps=$(env_steps "$env_short")
    flags=$(cond_flags "$cond")
    outdir="runs/factorial/${env_short}_${cond}_seed${seed}"
    mkdir -p "$outdir"

    if [ -f "$outdir/log.npz" ]; then
        echo "==> skip   $env_short/$cond/seed$seed (done)"
        return 0
    fi

    echo "==> start  $env_short/$cond/seed$seed (steps=$total_steps)"
    # shellcheck disable=SC2086
    $PY train_minigrid.py \
        --env-id "$gym_id" \
        --total-steps "$total_steps" \
        --seed "$seed" \
        --out-dir "$outdir" \
        $flags > "$outdir/train.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "==> done   $env_short/$cond/seed$seed"
    else
        echo "==> FAIL   $env_short/$cond/seed$seed (exit $rc, see $outdir/train.log)"
    fi
    return $rc
}

# ---- enumerate jobs ------------------------------------------------------
JOBS=()
for env_short in $ENVS; do
    for cond in $CONDS; do
        for seed in $SEEDS; do
            JOBS+=("$env_short:$cond:$seed")
        done
    done
done

echo "==> ${#JOBS[@]} jobs queued (envs=[$ENVS] conds=[$CONDS] seeds=[$SEEDS])"
echo "==> parallelism: $PARALLEL"
[ "$SMOKE" = "1" ] && echo "==> SMOKE mode: 20k steps per run"
echo

# ---- run jobs ------------------------------------------------------------
PIDS=()
FAILED=()
for spec in "${JOBS[@]}"; do
    IFS=':' read -r env_short cond seed <<< "$spec"

    # Block until a slot frees up.
    while [ "${#PIDS[@]}" -ge "$PARALLEL" ]; do
        new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            else
                wait "$pid" || FAILED+=("$pid")
            fi
        done
        PIDS=("${new_pids[@]}")
        [ "${#PIDS[@]}" -ge "$PARALLEL" ] && sleep 2
    done

    if [ "$PARALLEL" -eq 1 ]; then
        launch_one "$env_short" "$cond" "$seed" || FAILED+=("$spec")
    else
        ( launch_one "$env_short" "$cond" "$seed" ) &
        PIDS+=($!)
    fi
done

# Drain remaining background jobs.
for pid in "${PIDS[@]:-}"; do
    [ -z "$pid" ] && continue
    wait "$pid" || FAILED+=("pid$pid")
done

echo
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "==> All ${#JOBS[@]} jobs completed successfully."
else
    echo "==> ${#FAILED[@]} job(s) failed: ${FAILED[*]}"
    exit 1
fi
