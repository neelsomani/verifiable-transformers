#!/bin/bash
# Threshold sweep for circuit extraction with candidate_kl metric
#
# Usage:
#   bash scripts/gpt2/sweep_thresholds.sh quote_close
#   bash scripts/gpt2/sweep_thresholds.sh bracket_type
#
# Optional positional arguments:
#   task model_path n_examples metric output_root domain_manifest

set -e

TASK="${1:-quote_close}"
MODEL_PATH="${2:-artifacts/gpt2-norm-free}"
N_EXAMPLES="${3:-768}"
METRIC="${4:-candidate_kl}"
OUTPUT_BASE="${5:-artifacts/gpt2-circuits-v4/base}"
DOMAIN_MANIFEST="${6:-artifacts/gpt2-behavior-domains-v4/development.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=========================================="
echo "THRESHOLD SWEEP FOR: $TASK"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Metric: $METRIC"
echo "N examples: $N_EXAMPLES"
echo "Domain: $DOMAIN_MANIFEST"
echo ""

# Run sweep
for THRESH in 0.005 0.01 0.02 0.05 0.1 0.2; do
  echo "----------------------------------------"
  echo "Running with threshold: $THRESH"
  echo "----------------------------------------"

  OUTPUT_DIR="${OUTPUT_BASE}/${TASK}_t${THRESH}"

  "$PYTHON_BIN" scripts/gpt2/extract.py \
    --model_path "$MODEL_PATH" \
    --extract_circuit "$TASK" \
    --n_examples "$N_EXAMPLES" \
    --domain_manifest "$DOMAIN_MANIFEST" \
    --threshold "$THRESH" \
    --metric "$METRIC" \
    --min_agreement 1.0 \
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
