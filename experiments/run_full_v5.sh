#!/bin/bash
# Auto-launch full V5 parallel experiments after test completes
# This script monitors the test and launches the full suite when ready

set -e

TEST_PID=$(pgrep -f "run_v5_experiments.py.*num-seeds 2" || echo "")
TEST_DIR="experiments/harm_results/v5_20260625_145315"

echo "========================================================================"
echo "  V5 FULL EXPERIMENT AUTO-LAUNCHER"
echo "========================================================================"
echo "Monitoring test experiment..."
echo "Test PID: ${TEST_PID:-Not found}"
echo "Test dir: $TEST_DIR"
echo ""

if [ -z "$TEST_PID" ]; then
    echo "WARNING: Test experiment not running. Proceeding to full launch..."
else
    echo "Waiting for test to complete (PID $TEST_PID)..."
    while kill -0 $TEST_PID 2>/dev/null; do
        COMPLETED=$(find $TEST_DIR -name "result.json" 2>/dev/null | wc -l)
        echo "  [$(date +%H:%M:%S)] Test progress: $COMPLETED games completed"
        sleep 30
    done
    echo ""
    echo "✓ Test experiment completed!"
fi

# Check test results
echo ""
echo "========================================================================"
echo "  VALIDATING TEST RESULTS"
echo "========================================================================"

REPORT=$(find $TEST_DIR -name "report.html" 2>/dev/null | head -1)
COMPLETED_GAMES=$(find $TEST_DIR -name "result.json" 2>/dev/null | wc -l)

echo "Completed games: $COMPLETED_GAMES"
echo "Report: ${REPORT:-Not found}"

if [ -z "$REPORT" ]; then
    echo ""
    echo "WARNING: Test report not found. Checking for errors..."
    find $TEST_DIR -name "*.log" -o -name "*.err" 2>/dev/null || echo "No error logs found"
    echo ""
    echo "Proceeding to full launch anyway..."
fi

echo ""
echo "========================================================================"
echo "  LAUNCHING FULL V5 PARALLEL EXPERIMENTS"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Models: Sonnet 4.5, Opus 4.5, Haiku 3.5"
echo "  Experiments: ALL (6 types)"
echo "  Seeds: 5 per condition"
echo "  Max workers: 6"
echo "  Judge: Gemini 2.5 Pro"
echo ""
echo "Expected: ~855 games, ~2-3 hours"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Launch full parallel experiments
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --region us-east5 \
    --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \
    --experiments all \
    --judge-model gemini-2.5-pro \
    --num-seeds 5 \
    --max-turns 20 \
    --max-workers 6 \
    --output-dir experiments/harm_results

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  FULL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - All experiments completed"
    echo ""
    echo "Results:"
    LATEST=$(ls -td experiments/harm_results/v5_parallel_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "  Directory: $LATEST"
        echo "  Master report: $LATEST/master_report.html"
        echo ""
        echo "Next steps:"
        echo "  1. Open master_report.html in browser"
        echo "  2. Review individual experiment reports"
        echo "  3. Run inter-judge agreement analysis"
        echo "  4. Generate paper tables/figures"
    fi
else
    echo "✗ FAILED - Check logs for errors"
    echo ""
    LATEST=$(ls -td experiments/harm_results/v5_parallel_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "Partial results in: $LATEST"
    fi
fi

exit $EXIT_CODE
