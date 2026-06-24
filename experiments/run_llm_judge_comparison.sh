#!/bin/bash
# Run 9 experiments: 3 models × 3 buyer types with LLM-as-judge evaluation
# Same conditions as the original regex-based comparison for direct comparison

set -e

PROJECT_ID="itpc-gcp-ai-eng-claude"
REGION="us-east5"
SEED=42
MORTGAGE_PCT=0.95
MAX_TURNS=20

MODELS=(
    "claude-sonnet-4-5@20250929"
    "claude-opus-4-5@20251101"
    "claude-3-5-haiku@20241022"
)

BUYERS=(
    "naive"
    "standard"
    "trap"
)

TOTAL=$((${#MODELS[@]} * ${#BUYERS[@]}))
COUNT=0

echo "=========================================="
echo "  LLM Judge Model Comparison"
echo "  $TOTAL experiments to run"
echo "=========================================="
echo ""

for MODEL in "${MODELS[@]}"; do
    for BUYER in "${BUYERS[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Model=$MODEL Buyer=$BUYER"
        echo "  Starting at $(date '+%H:%M:%S')"

        venv/bin/python experiments/run_experiment.py \
            --project-id "$PROJECT_ID" \
            --region "$REGION" \
            --model "$MODEL" \
            --buyer-sophistication "$BUYER" \
            --seed "$SEED" \
            --mortgage-pct "$MORTGAGE_PCT" \
            --max-turns "$MAX_TURNS" \
            --use-llm-judge \
            --property-type major

        echo "  Finished at $(date '+%H:%M:%S')"
        echo ""
    done
done

echo "=========================================="
echo "  All $TOTAL experiments complete!"
echo "=========================================="
