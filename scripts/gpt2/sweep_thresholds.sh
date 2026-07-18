#!/bin/bash
# Threshold sweep for circuit extraction with candidate_kl metric
#
# Usage:
#   bash scripts/gpt2/sweep_thresholds.sh quote_close
#   bash scripts/gpt2/sweep_thresholds.sh bracket_type

set -e

TASK="${1:-quote_close}"
MODEL_PATH="${2:-artifacts/band-norm-sparsemax/checkpoint-240000}"
N_EXAMPLES="${3:-128}"
METRIC="${4:-candidate_kl}"
OUTPUT_BASE="${5:-artifacts/circuits_sweep}"

echo "=========================================="
echo "THRESHOLD SWEEP FOR: $TASK"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Metric: $METRIC"
echo "N examples: $N_EXAMPLES"
echo ""

# Run sweep
for THRESH in 0.005 0.01 0.02 0.05 0.1 0.2; do
  echo "----------------------------------------"
  echo "Running with threshold: $THRESH"
  echo "----------------------------------------"

  OUTPUT_DIR="${OUTPUT_BASE}/${TASK}_t${THRESH}"

  python scripts/gpt2/extract.py \
    --model_path "$MODEL_PATH" \
    --extract_circuit "$TASK" \
    --n_examples "$N_EXAMPLES" \
    --threshold "$THRESH" \
    --metric "$METRIC" \
    --trim_rounds 0 \
    --output_dir "$OUTPUT_DIR"

  echo ""
done

echo "=========================================="
echo "SWEEP COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "To compare results:"
echo "  python scripts/gpt2/compare_sweeps.py --sweep_dir $OUTPUT_BASE --task $TASK"
echo ""
