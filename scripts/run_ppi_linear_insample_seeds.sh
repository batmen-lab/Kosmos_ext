#!/usr/bin/env bash
# Run official Kosmos PPI (linear model + in-sample pseudo labels) over seeds.
#
# Usage:
#   bash scripts/run_ppi_linear_insample_seeds.sh [seed1 seed2 ...]
#
# Example:
#   bash scripts/run_ppi_linear_insample_seeds.sh 42 7 123 2024
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(42 7 123 2024)
fi

export WORLD_MODEL_ENABLED=false
export REDIS_ENABLED=false
export ENABLE_SANDBOXING=false
export OPENAI_TIMEOUT=60
export USE_LITERATURE_CONTEXT=false
export REQUIRE_NOVELTY_CHECK=false
export NUM_HYPOTHESES=1
export PPI_MODEL_DESIGN=linear
export PPI_PSEUDO_MODE=in_sample
export PPI_STAGE1_EPOCHS=4
export PPI_STAGE2_EPOCHS=1
export PPI_EXTERNAL_PER_DONOR=5000
export PPI_MAX_EXTERNAL_SAMPLES=20000
export PPI_MAX_EPOCHS=20
export PPI_PATIENCE=5

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/kosmos-mpl
# Local venv (stable Python 3.11 + PyTorch); override with KOSMOS_PYTHON=...
PYTHON="${KOSMOS_PYTHON:-$ROOT/.venv/bin/python}"

QUESTION="Does adding pseudo-labeled external non-gold RNA data (PPILoss correction) improve cross-donor cell-type classification versus gold-only training, with a linear model and in-sample pseudo-label mode (gold model fit on all gold training data)? Train donor 13272 (80/20 dev), evaluate donor 19593, use external_unlabeled as unlabeled PPI evidence, and report baseline vs PPI accuracy, balanced accuracy, macro F1 and per-class recall changes."

for SEED in "${SEEDS[@]}"; do
  echo "=== starting PPI_SEED=$SEED ==="
  PPI_SEED="$SEED" "$PYTHON" -m kosmos.cli.main \
    run "$QUESTION" \
    --domain biology \
    --max-iterations 1 \
    --budget 5 \
    --data-path "$ROOT/data/csv_donor_splits/gold_13272_19593_labeled.csv" \
    --external-data-path "$ROOT/data/csv_donor_splits/external_unlabeled.csv" \
    --output "$ROOT/artifacts/ppi_linear_insample_seed_${SEED}.json"
  echo "=== PPI_SEED=$SEED finished (exit=$?) ==="
done
