#!/usr/bin/env python3
"""Run 9 experiments: 3 models x 3 buyer types with LLM-as-judge evaluation."""

import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT_ID = "itpc-gcp-ai-eng-claude"
REGION = "us-east5"
SEED = "42"
MORTGAGE_PCT = "0.95"
MAX_TURNS = "20"

MODELS = [
    "claude-sonnet-4-5@20250929",
    "claude-opus-4-5@20251101",
    "claude-3-5-haiku@20241022",
]

BUYERS = ["naive", "standard", "trap"]

PYTHON = str(Path(__file__).parent.parent / "venv" / "bin" / "python")
SCRIPT = str(Path(__file__).parent / "run_experiment.py")

results_log = []
total = len(MODELS) * len(BUYERS)
count = 0

for model in MODELS:
    for buyer in BUYERS:
        count += 1
        model_short = model.split("@")[0].replace("claude-", "")
        print(f"\n[{count}/{total}] {model_short} x {buyer}")
        print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        t0 = time.time()

        cmd = [
            PYTHON, SCRIPT,
            "--project-id", PROJECT_ID,
            "--region", REGION,
            "--model", model,
            "--buyer-sophistication", buyer,
            "--seed", SEED,
            "--mortgage-pct", MORTGAGE_PCT,
            "--max-turns", MAX_TURNS,
            "--use-llm-judge",
            "--property-type", "major",
        ]

        proc = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - t0

        results_log.append({
            "model": model, "buyer": buyer,
            "status": "ok" if proc.returncode == 0 else f"exit_{proc.returncode}",
            "elapsed_s": round(elapsed),
        })
        print(f"  Done in {elapsed:.0f}s (exit={proc.returncode})")

print(f"\n{'='*60}")
print(f"  All {total} experiments complete")
print(f"{'='*60}")
for r in results_log:
    m = r["model"].split("@")[0].replace("claude-", "")
    print(f"  {m:>20} x {r['buyer']:<10} {r['status']:<8} ({r['elapsed_s']}s)")
