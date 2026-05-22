#!/usr/bin/env bash
# PER prioritization-exponent sweep, conditioned on exploration regime.
#
# Question: does the optimal PER alpha differ between NoisyNet and eps-greedy?
# The deep-research literature scan flagged this as the cleanest open novelty
# slice — Rainbow / Revisiting Rainbow study alpha robustness inside their full
# stack, but no published source isolates how alpha's effect depends on the
# exploration mechanism that fills the buffer.
#
# Sweep: alpha in {0.4, 0.8} (alpha=0.6 is covered by run_factorial.sh)
# Conditions: {noisy_per, eps_per}
# Seeds: 0..4
# Env: MiniGrid-DoorKey-6x6 only (lean scope; can extend later if interaction
#      is confirmed)
#
# = 2 alphas x 2 conditions x 5 seeds = 20 runs.
#
# Output: runs/alpha_sweep/doorkey6_<cond>_a<alpha>_seed<N>/
#
# Usage:
#   chmod +x run_alpha_sweep.sh
#   ./run_alpha_sweep.sh
#
# Knobs (same shape as run_factorial.sh):
#   SEEDS="0 1 2 3 4"
#   PARALLEL=1
#   SMOKE=1

set -uo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
PY=.venv/bin/python
$PY -m pip install --upgrade pip --quiet
$PY -m pip install -r requirements.txt --quiet
$PY -m pip install minigrid --quiet

mkdir -p runs/alpha_sweep runs/.matplotlib
export MPLCONFIGDIR="$PWD/runs/.matplotlib"
export PYTHONUNBUFFERED=1

SEEDS=${SEEDS:-"0 1 2 3 4"}
PARALLEL=${PARALLEL:-1}
SMOKE=${SMOKE:-0}
ALPHAS="0.4 0.8"
CONDS="noisy_per eps_per"
ENV_ID="MiniGrid-DoorKey-6x6-v0"
if [ "$SMOKE" = "1" ]; then TOTAL_STEPS=20000; else TOTAL_STEPS=2000000; fi

cond_flags() {
    case "$1" in
        noisy_per) echo "--noisy --per" ;;
        eps_per)   echo "--no-noisy --per" ;;
    esac
}

launch_one() {
    local cond=$1 alpha=$2 seed=$3
    local flags outdir
    flags=$(cond_flags "$cond")
    outdir="runs/alpha_sweep/doorkey6_${cond}_a${alpha}_seed${seed}"
    mkdir -p "$outdir"

    if [ -f "$outdir/log.npz" ]; then
        echo "==> skip   $cond/a=$alpha/seed$seed (done)"
        return 0
    fi

    echo "==> start  $cond/a=$alpha/seed$seed"
    # shellcheck disable=SC2086
    $PY train_minigrid.py \
        --env-id "$ENV_ID" \
        --total-steps "$TOTAL_STEPS" \
        --seed "$seed" \
        --per-alpha "$alpha" \
        --out-dir "$outdir" \
        $flags > "$outdir/train.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "==> done   $cond/a=$alpha/seed$seed"
    else
        echo "==> FAIL   $cond/a=$alpha/seed$seed (exit $rc)"
    fi
    return $rc
}

JOBS=()
for cond in $CONDS; do
    for alpha in $ALPHAS; do
        for seed in $SEEDS; do
            JOBS+=("$cond:$alpha:$seed")
        done
    done
done

echo "==> ${#JOBS[@]} alpha-sweep jobs queued (alphas=[$ALPHAS] conds=[$CONDS] seeds=[$SEEDS])"
echo "==> parallelism: $PARALLEL"
[ "$SMOKE" = "1" ] && echo "==> SMOKE mode: 20k steps per run"
echo

PIDS=()
FAILED=()
for spec in "${JOBS[@]}"; do
    IFS=':' read -r cond alpha seed <<< "$spec"

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
        launch_one "$cond" "$alpha" "$seed" || FAILED+=("$spec")
    else
        ( launch_one "$cond" "$alpha" "$seed" ) &
        PIDS+=($!)
    fi
done

for pid in "${PIDS[@]:-}"; do
    [ -z "$pid" ] && continue
    wait "$pid" || FAILED+=("pid$pid")
done

echo
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "==> All ${#JOBS[@]} alpha-sweep jobs completed successfully."
    echo "    For the alpha=0.6 row of the sweep, reuse runs from run_factorial.sh"
    echo "    (runs/factorial/doorkey6_{noisy_per,eps_per}_seed*)."
else
    echo "==> ${#FAILED[@]} job(s) failed: ${FAILED[*]}"
    exit 1
fi
