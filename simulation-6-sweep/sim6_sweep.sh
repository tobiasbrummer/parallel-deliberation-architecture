#!/usr/bin/env bash
# Full Sim 6 sweep: 5 variants × 3 seeds = 15 runs.
# Each run = train + eval. Baseline eval done once per seed.
#
# Run inside `tmux` or `nohup` so it survives SSH drops:
#   tmux new -s sim6
#   bash sim6_sweep.sh 2>&1 | tee /workspace/logs/sweep.log
#
# Resume: re-running is safe — adapters are skipped if already trained,
# results-JSONs are overwritten (so eval can be redone independently).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_TRAIN=${N_TRAIN:-150}
N_EVAL=${N_EVAL:-200}

VARIANTS=("cot" "pda2" "pda3" "pda4" "pda5")
SEEDS=(42 1337 2718)

echo "Sim 6 sweep: ${#VARIANTS[@]} variants × ${#SEEDS[@]} seeds = $((${#VARIANTS[@]} * ${#SEEDS[@]})) runs"
echo "n_train=$N_TRAIN  n_eval=$N_EVAL"
echo

for seed in "${SEEDS[@]}"; do
    first=true
    for variant in "${VARIANTS[@]}"; do
        echo "=== variant=$variant seed=$seed ==="
        if $first; then
            # Eval baseline once per seed, attached to first variant of that seed
            python "$SCRIPT_DIR/sim6.py" \
                --variant "$variant" --seed "$seed" \
                --n-train "$N_TRAIN" --n-eval "$N_EVAL"
            first=false
        else
            python "$SCRIPT_DIR/sim6.py" \
                --variant "$variant" --seed "$seed" \
                --n-train "$N_TRAIN" --n-eval "$N_EVAL" \
                --skip-base
        fi
    done
done

echo
echo "Sweep done. Results in /workspace/results/"
ls -la /workspace/results/
