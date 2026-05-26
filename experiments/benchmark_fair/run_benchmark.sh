#!/bin/bash
# Fair benchmark launcher: runs ALL 6 methods across all datasets.
#
# Methods:
#   Phase 1 (transpa env): FlashDeconv, RCTD, MarkerScore
#   Phase 2 (seededntm env): STAMP, SeedTopic ref-based, SeedTopic ref-free
#
# Usage:
#   bash experiments/benchmark_fair/run_benchmark.sh           # all datasets
#   bash experiments/benchmark_fair/run_benchmark.sh visiumHD_CRC_I  # single dataset
#   bash experiments/benchmark_fair/run_benchmark.sh CRC_ONLY  # CRC datasets only
#   bash experiments/benchmark_fair/run_benchmark.sh SINGLE visiumHD_CRC_I
#
# Run from the SeedTopic project root on a GPU node.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK_SCRIPT="$SCRIPT_DIR/run_fair_benchmark.py"
LOG_DIR="$SCRIPT_DIR/logs"
PROCESSED_DIR="$SCRIPT_DIR/data/processed"

cd "$PROJECT_ROOT"

# Check that processed data exists
if [ ! -d "$PROCESSED_DIR" ]; then
    echo "ERROR: Processed data not found at $PROCESSED_DIR"
    echo "Run the following first:"
    echo "  1. bash experiments/benchmark_fair/copy_raw_data.sh"
    echo "  2. micromamba activate seededntm && python experiments/benchmark_fair/preprocess_data.py --all"
    exit 1
fi

eval "$(micromamba shell hook --shell bash)"

DATASETS_ALL=("visium_NPC" "xenium_BC" "visiumHD_CRC_I" "visiumHD_CRC_II")
DATASETS_CRC=("visiumHD_CRC_I" "visiumHD_CRC_II")

MODE="${1:-ALL}"
SINGLE_DS="${2:-}"

case "$MODE" in
    CRC_ONLY)
        DATASETS=("${DATASETS_CRC[@]}")
        ;;
    SINGLE)
        if [ -z "$SINGLE_DS" ]; then
            echo "ERROR: SINGLE mode requires dataset name as 2nd argument"
            exit 1
        fi
        DATASETS=("$SINGLE_DS")
        ;;
    visium_NPC|xenium_BC|visiumHD_CRC_I|visiumHD_CRC_II)
        DATASETS=("$MODE")
        ;;
    ALL|*)
        DATASETS=("${DATASETS_ALL[@]}")
        ;;
esac

# CLEAN START: remove ALL old results and logs
echo "============================================================"
echo "FAIR BENCHMARK — FULL 6-METHOD RUN (CLEAN START)"
echo "Datasets: ${DATASETS[*]}"
echo "Methods:  FlashDeconv, RCTD, MarkerScore, STAMP, SeedTopic (ref-based), SeedTopic (ref-free)"
echo "Log dir:  $LOG_DIR"
echo "============================================================"
echo ""
echo "Cleaning previous results..."
rm -f "$SCRIPT_DIR/fair_benchmark_results.json"
rm -rf "$SCRIPT_DIR/raw_outputs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log
echo "  Done."

# Phase 1: FlashDeconv + RCTD + MarkerScore (transpa env, up to 2 parallel)
echo ""
echo ">>> Phase 1: FlashDeconv + RCTD + MarkerScore (transpa env)"
echo "============================================================"
set +u
micromamba activate transpa
set -u

PIDS=()
for ds in "${DATASETS[@]}"; do
    echo "  Starting FlashDeconv + RCTD + MarkerScore on $ds ..."
    python "$BENCHMARK_SCRIPT" \
        --dataset "$ds" \
        --methods flashdeconv rctd marker_scoring \
        --device cuda \
        > "$LOG_DIR/${ds}_phase1.log" 2>&1 &
    PIDS+=($!)

    # Limit to 2 parallel (GPU memory for FlashDeconv/RCTD)
    if [ ${#PIDS[@]} -ge 2 ]; then
        wait "${PIDS[0]}"
        echo "  Completed: $ds (exit=$?)"
        PIDS=("${PIDS[@]:1}")
    fi
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  Process $pid completed (exit=$?)"
done
echo "  Phase 1 complete."

# Phase 2: STAMP + SeedTopic ref-based + SeedTopic ref-free (seededntm env, up to 3 parallel)
echo ""
echo ">>> Phase 2: STAMP + SeedTopic (seededntm env)"
echo "============================================================"
set +u
micromamba activate seededntm
set -u

PIDS=()
for ds in "${DATASETS[@]}"; do
    echo "  Starting STAMP + SeedTopic (ref-based + ref-free) on $ds ..."
    python "$BENCHMARK_SCRIPT" \
        --dataset "$ds" \
        --methods stamp seedtopic seedtopic_reffree \
        --device cuda \
        > "$LOG_DIR/${ds}_phase2.log" 2>&1 &
    PIDS+=($!)

    # Limit to 3 parallel (SeedTopic/STAMP lighter on GPU)
    if [ ${#PIDS[@]} -ge 3 ]; then
        wait "${PIDS[0]}"
        echo "  Completed (exit=$?)"
        PIDS=("${PIDS[@]:1}")
    fi
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  Process $pid completed (exit=$?)"
done
echo "  Phase 2 complete."

echo ""
echo "============================================================"
echo "ALL DONE. Results: $SCRIPT_DIR/fair_benchmark_results.json"
echo "Logs:    $LOG_DIR/"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review logs for any errors: ls $LOG_DIR/*.log"
echo "  2. Check results: python -c \"import json; r=json.load(open('$SCRIPT_DIR/fair_benchmark_results.json')); print(f'{len(r)} results recorded')\""
