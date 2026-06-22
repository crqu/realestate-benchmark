#!/bin/bash
# Quick launcher for real estate transaction experiment
# 
# Usage: ./run_experiment.sh YOUR_PROJECT_ID

if [ -z "$1" ]; then
    echo "Usage: $0 YOUR_PROJECT_ID"
    echo ""
    echo "Example:"
    echo "  $0 my-gcp-project-123"
    exit 1
fi

# Launch in tmux
./experiments/run_in_tmux.sh "$1" us-east5 claude-sonnet-4-5@20250929 standard
