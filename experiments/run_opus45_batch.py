#!/usr/bin/env python3
"""Run all harm taxonomy experiments for Opus 4.5.

Runs a reduced matrix to keep total runtime manageable (~2-3 hours).
Each game takes ~7 minutes with Opus 4.5 on Vertex AI.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "experiments"))

from run_harm_experiments import (
    run_advice_experiment,
    run_coercion_experiment,
    run_confidentiality_experiment,
    run_discrimination_experiment,
    run_misrepresentation_experiment,
    run_tool_use_experiment,
)

from realestate_benchmark.data.ames import load_ames_data
from realestate_benchmark.models.vertex import VertexModel

OUTPUT_DIR = Path("experiments/harm_results/opus45_batch")


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = VertexModel(
        project_id="itpc-gcp-ai-eng-claude",
        region="us-east5",
        model="claude-opus-4-5",
    )
    df = load_ames_data()
    print(f"Loaded {len(df)} properties")
    print(f"Started at {datetime.now().isoformat()}")
    print(f"{'=' * 70}\n")

    all_results = {}

    # 1. Discrimination: 2 attributes × 1 property type × 1 seed × 2 (T/C) = 4 games
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 1: DISCRIMINATION")
    print(f"{'=' * 70}\n")
    try:
        results = run_discrimination_experiment(
            model, df,
            protected_attributes=["race", "disability"],
            property_types=["major"],
            num_seeds=1,
            base_seed=42,
            max_turns=14,
            output_dir=OUTPUT_DIR / "discrimination",
            use_llm_judge=True,
        )
        all_results["discrimination"] = results
        with open(OUTPUT_DIR / "discrimination_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Completed {len(results)} discrimination games")
    except Exception as e:
        print(f"  DISCRIMINATION FAILED: {e}")
        all_results["discrimination"] = []

    # 2. Misrepresentation: 2 pressure × 2 buyer × 1 property = 4 games
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 2: MISREPRESENTATION")
    print(f"{'=' * 70}\n")
    try:
        results = run_misrepresentation_experiment(
            model, df,
            pressure_levels={"low": 0.50, "high": 0.95},
            buyer_levels=["naive", "savvy"],
            property_types=["major"],
            num_seeds=1,
            base_seed=42,
            max_turns=14,
            output_dir=OUTPUT_DIR / "misrepresentation",
            use_llm_judge=True,
        )
        all_results["misrepresentation"] = results
        with open(OUTPUT_DIR / "misrepresentation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Completed {len(results)} misrepresentation games")
    except Exception as e:
        print(f"  MISREPRESENTATION FAILED: {e}")
        all_results["misrepresentation"] = []

    # 3. Confidentiality: 2 info types × 2 strategies × 1 buyer × 1 prop = 4 games
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 3: CONFIDENTIALITY")
    print(f"{'=' * 70}\n")
    try:
        results = run_confidentiality_experiment(
            model, df,
            info_types=["mortgage_balance", "competing_offers"],
            extraction_strategies=["direct", "social_engineering"],
            buyer_levels=["savvy"],
            property_types=["major"],
            num_seeds=1,
            base_seed=42,
            max_turns=14,
            output_dir=OUTPUT_DIR / "confidentiality",
            use_llm_judge=True,
        )
        all_results["confidentiality"] = results
        with open(OUTPUT_DIR / "confidentiality_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Completed {len(results)} confidentiality games")
    except Exception as e:
        print(f"  CONFIDENTIALITY FAILED: {e}")
        all_results["confidentiality"] = []

    # 4. Advice: 3 solicitation types × 1 buyer × 1 prop = 3 games
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 4: ADVICE")
    print(f"{'=' * 70}\n")
    try:
        results = run_advice_experiment(
            model, df,
            solicitation_types=["direct_legal", "direct_financial", "indirect"],
            buyer_levels=["naive"],
            property_types=["major"],
            num_seeds=1,
            base_seed=42,
            max_turns=14,
            output_dir=OUTPUT_DIR / "advice",
            use_llm_judge=True,
        )
        all_results["advice"] = results
        with open(OUTPUT_DIR / "advice_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Completed {len(results)} advice games")
    except Exception as e:
        print(f"  ADVICE FAILED: {e}")
        all_results["advice"] = []

    # 5. Coercion: 2 pressure × 2 vulnerability = 4 games
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 5: COERCION")
    print(f"{'=' * 70}\n")
    try:
        results = run_coercion_experiment(
            model, df,
            vulnerability_types=["standard", "elderly"],
            property_types=["major"],
            num_seeds=1,
            base_seed=42,
            max_turns=14,
            output_dir=OUTPUT_DIR / "coercion",
            use_llm_judge=True,
        )
        all_results["coercion"] = results
        with open(OUTPUT_DIR / "coercion_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Completed {len(results)} coercion games")
    except Exception as e:
        print(f"  COERCION FAILED: {e}")
        all_results["coercion"] = []

    # 6. Tool-use: 3 autonomy × 1 buyer = 3 games
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 6: TOOL-USE")
    print(f"{'=' * 70}\n")
    try:
        results = run_tool_use_experiment(
            model, df,
            autonomy_levels=["standard", "autonomous", "cautious"],
            buyer_levels=["naive"],
            property_types=["major"],
            num_seeds=1,
            base_seed=42,
            max_turns=14,
            output_dir=OUTPUT_DIR / "tool_use",
            use_llm_judge=True,
        )
        all_results["tool_use"] = results
        with open(OUTPUT_DIR / "tool_use_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Completed {len(results)} tool-use games")
    except Exception as e:
        print(f"  TOOL-USE FAILED: {e}")
        all_results["tool_use"] = []

    # Summary
    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  ALL EXPERIMENTS COMPLETE — {elapsed/60:.1f} minutes total")
    print(f"{'=' * 70}\n")

    summary = {}
    for exp_name, results in all_results.items():
        n = len(results)
        completed = sum(1 for r in results if r.get("completed"))
        summary[exp_name] = {"n_games": n, "n_completed": completed}
        print(f"  {exp_name}: {n} games, {completed} sales")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({"summary": summary, "all_results": all_results}, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
