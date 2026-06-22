#!/usr/bin/env bash
#
# Run Real Estate Transaction Experiment in Tmux
#
# This script creates a new tmux session and runs an end-to-end experiment
# to observe AI agent behaviors in a real estate transaction scenario.
#
# Usage:
#   ./experiments/run_in_tmux.sh YOUR_PROJECT_ID
#
# The experiment will:
# 1. Load a property from the Ames Housing dataset
# 2. Create seller and buyer agents using Vertex AI (Claude)
# 3. Run a complete transaction negotiation
# 4. Compute BIAI metrics (P1-P4 + SAI)
# 5. Generate a comprehensive report
#
# Requirements:
# - Google Cloud project with Vertex AI enabled
# - Authentication configured (gcloud auth application-default login)
# - Ames dataset downloaded to data/ames/

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SESSION_NAME="realestate-experiment"
PROJECT_ID="${1:-}"
REGION="${2:-us-east5}"
MODEL="${3:-claude-sonnet-4-5@20250929}"
BUYER_SOPHISTICATION="${4:-standard}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Print header
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Real Estate Transaction Experiment - Tmux Runner${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo

# Validate project ID
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: Google Cloud project ID required${NC}"
    echo
    echo "Usage: $0 PROJECT_ID [REGION] [MODEL] [BUYER_SOPHISTICATION]"
    echo
    echo "Examples:"
    echo "  $0 my-project-123"
    echo "  $0 my-project-123 us-east5 claude-sonnet-4-5@20250929 savvy"
    echo
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Project ID:           $PROJECT_ID"
echo "  Region:               $REGION"
echo "  Model:                $MODEL"
echo "  Buyer Sophistication: $BUYER_SOPHISTICATION"
echo "  Repository:           $REPO_ROOT"
echo

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Tmux session '$SESSION_NAME' already exists${NC}"
    read -p "Do you want to kill it and start fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Killing existing session..."
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Attaching to existing session..."
        tmux attach-session -t "$SESSION_NAME"
        exit 0
    fi
fi

# Check for Ames dataset
if [ ! -d "$REPO_ROOT/data/ames" ] || [ ! -f "$REPO_ROOT/data/ames/train.csv" ]; then
    echo -e "${YELLOW}⚠️  Ames dataset not found${NC}"
    echo
    echo "You need to download the Ames Housing dataset first."
    echo "See: $REPO_ROOT/data/ames/README.md"
    echo
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for virtual environment
if [ ! -d "$REPO_ROOT/venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found at $REPO_ROOT/venv${NC}"
    echo "Creating virtual environment..."
    cd "$REPO_ROOT"
    python3 -m venv venv
    source venv/bin/activate
    pip install -e ".[all]"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment found${NC}"
fi

# Create experiment directory
mkdir -p "$REPO_ROOT/experiments/results"

# Create tmux session with environment setup
echo
echo -e "${GREEN}🚀 Creating tmux session: $SESSION_NAME${NC}"
echo

# Create the session
tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"

# Set up the environment in the tmux session
tmux send-keys -t "$SESSION_NAME" "# Real Estate Transaction Experiment" C-m
tmux send-keys -t "$SESSION_NAME" "# Project: $PROJECT_ID | Model: $MODEL | Buyer: $BUYER_SOPHISTICATION" C-m
tmux send-keys -t "$SESSION_NAME" "" C-m

# Activate virtual environment
tmux send-keys -t "$SESSION_NAME" "source venv/bin/activate" C-m
tmux send-keys -t "$SESSION_NAME" "clear" C-m

# Display welcome message
tmux send-keys -t "$SESSION_NAME" "cat << 'EOF'" C-m
tmux send-keys -t "$SESSION_NAME" "══════════════════════════════════════════════════════════════════" C-m
tmux send-keys -t "$SESSION_NAME" "  REAL ESTATE TRANSACTION EXPERIMENT" C-m
tmux send-keys -t "$SESSION_NAME" "══════════════════════════════════════════════════════════════════" C-m
tmux send-keys -t "$SESSION_NAME" "" C-m
tmux send-keys -t "$SESSION_NAME" "This experiment will:" C-m
tmux send-keys -t "$SESSION_NAME" "  1. Load a property from the Ames Housing dataset" C-m
tmux send-keys -t "$SESSION_NAME" "  2. Create seller & buyer agents (Vertex AI + Claude)" C-m
tmux send-keys -t "$SESSION_NAME" "  3. Run a complete real estate negotiation" C-m
tmux send-keys -t "$SESSION_NAME" "  4. Compute behavioral integrity metrics (P1-P4 + SAI)" C-m
tmux send-keys -t "$SESSION_NAME" "  5. Generate a comprehensive analysis report" C-m
tmux send-keys -t "$SESSION_NAME" "" C-m
tmux send-keys -t "$SESSION_NAME" "Configuration:" C-m
tmux send-keys -t "$SESSION_NAME" "  • Project:  $PROJECT_ID" C-m
tmux send-keys -t "$SESSION_NAME" "  • Region:   $REGION" C-m
tmux send-keys -t "$SESSION_NAME" "  • Model:    $MODEL" C-m
tmux send-keys -t "$SESSION_NAME" "  • Buyer:    $BUYER_SOPHISTICATION sophistication" C-m
tmux send-keys -t "$SESSION_NAME" "" C-m
tmux send-keys -t "$SESSION_NAME" "The experiment will run automatically." C-m
tmux send-keys -t "$SESSION_NAME" "Results will be saved to: experiments/results/" C-m
tmux send-keys -t "$SESSION_NAME" "" C-m
tmux send-keys -t "$SESSION_NAME" "Press ENTER to start, or Ctrl+C to cancel..." C-m
tmux send-keys -t "$SESSION_NAME" "══════════════════════════════════════════════════════════════════" C-m
tmux send-keys -t "$SESSION_NAME" "EOF" C-m

# Pause for user to read
tmux send-keys -t "$SESSION_NAME" "read -p 'Press ENTER to continue...'" C-m

# Run the experiment
EXPERIMENT_CMD="python experiments/run_experiment.py \
  --project-id '$PROJECT_ID' \
  --region '$REGION' \
  --model '$MODEL' \
  --buyer-sophistication '$BUYER_SOPHISTICATION' \
  --property-type random \
  --max-turns 50 \
  --seed 42"

tmux send-keys -t "$SESSION_NAME" "$EXPERIMENT_CMD" C-m

# Instructions
echo -e "${GREEN}✅ Tmux session created successfully!${NC}"
echo
echo -e "${BLUE}To attach to the session:${NC}"
echo "  tmux attach-session -t $SESSION_NAME"
echo
echo -e "${BLUE}To detach from the session (while inside):${NC}"
echo "  Press: Ctrl+B, then D"
echo
echo -e "${BLUE}To kill the session:${NC}"
echo "  tmux kill-session -t $SESSION_NAME"
echo
echo -e "${BLUE}Useful tmux commands (while attached):${NC}"
echo "  Ctrl+B [       Enter scroll mode (use arrows, PgUp/PgDn)"
echo "  Ctrl+B :       Enter command mode"
echo "  q              Exit scroll mode"
echo
echo -e "${YELLOW}Attaching to session in 3 seconds...${NC}"
sleep 3

# Attach to the session
tmux attach-session -t "$SESSION_NAME"
