#!/usr/bin/env python3
"""Unified batch runner for harm taxonomy experiments.

Usage:
    venv/bin/python experiments/run_harm_batch.py \
        --project-id YOUR_PROJECT_ID \
        --experiments discrimination confidentiality coercion \
        --num-seeds 3 \
        --output-dir experiments/harm_results
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_harm_experiments import (
    run_advice_experiment,
    run_coercion_experiment,
    run_confidentiality_experiment,
    run_discrimination_experiment,
    run_misrepresentation_experiment,
)

from realestate_benchmark.data.ames import load_ames_data
from realestate_benchmark.evaluation.harm_analysis import (
    compute_confidentiality_resistance_score,
    compute_discrimination_index,
    compute_harm_emergence_rate,
    compute_vulnerability_exploitation_index,
    generate_harm_report,
)
from realestate_benchmark.models.vertex import VertexModel

ALL_EXPERIMENTS = [
    "discrimination",
    "misrepresentation",
    "confidentiality",
    "advice",
    "coercion",
]


def main():
    parser = argparse.ArgumentParser(description="Harm taxonomy experiment batch runner")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-east5")
    parser.add_argument("--model", default="claude-sonnet-4-5@20250929")
    parser.add_argument(
        "--experiments", nargs="+", default=ALL_EXPERIMENTS, choices=ALL_EXPERIMENTS,
    )
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--output-dir", default="experiments/harm_results")
    parser.add_argument("--use-llm-judge", action="store_true", default=True)
    parser.add_argument("--no-llm-judge", dest="use_llm_judge", action="store_false")
    parser.add_argument("--judge-model", default=None,
                        help="Judge model (e.g. gemini-2.5-pro). Defaults to same as --model")
    parser.add_argument("--judge-provider", default=None,
                        choices=["vertex", "gemini"],
                        help="Judge provider (inferred from model name if omitted)")
    parser.add_argument("--max-game-workers", type=int, default=4,
                        help="Max parallel games within each experiment (default: 4)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.output_dir) / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print("  HARM TAXONOMY EXPERIMENT BATCH")
    print(f"{'=' * 70}")
    print(f"Output: {batch_dir}")
    print(f"Model: {args.model}")
    print(f"Judge: {args.judge_model or args.model}")
    print(f"Experiments: {args.experiments}")
    print(f"Seeds: {args.num_seeds} (base: {args.base_seed})")
    print(f"Max turns: {args.max_turns}")
    print(f"LLM Judge: {args.use_llm_judge}")
    print()

    with open(batch_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = VertexModel(project_id=args.project_id, region=args.region, model=args.model)

    judge_model_instance = None
    if args.judge_model:
        judge_name = args.judge_model
        provider = args.judge_provider
        if provider is None:
            provider = "gemini" if "gemini" in judge_name else "vertex"

        if provider == "gemini":
            from realestate_benchmark.models.gemini import GeminiModel
            judge_model_instance = GeminiModel(
                project_id=args.project_id,
                region=args.region,
                model=judge_name,
                json_mode=True,
            )
        else:
            judge_model_instance = VertexModel(
                project_id=args.project_id,
                region=args.region,
                model=judge_name,
            )
        print(f"Judge model: {judge_name} ({provider})")

    df = load_ames_data()
    print(f"Loaded {len(df)} properties from Ames dataset\n")

    all_experiment_results: dict[str, list[dict]] = {}

    common_kwargs = {
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "max_turns": args.max_turns,
        "use_llm_judge": args.use_llm_judge,
        "judge_model": judge_model_instance,
        "max_game_workers": args.max_game_workers,
    }

    # --- Run each experiment ---

    if "discrimination" in args.experiments:
        print(f"\n{'=' * 70}")
        print("  EXPERIMENT: DISCRIMINATION (Fair Housing + Bargaining)")
        print(f"{'=' * 70}\n")
        results = run_discrimination_experiment(
            model, df,
            output_dir=batch_dir / "discrimination",
            **common_kwargs,
        )
        all_experiment_results["discrimination"] = results
        print(f"\n  Completed {len(results)} discrimination games")

    if "misrepresentation" in args.experiments:
        print(f"\n{'=' * 70}")
        print("  EXPERIMENT: MISREPRESENTATION")
        print(f"{'=' * 70}\n")
        results = run_misrepresentation_experiment(
            model, df,
            output_dir=batch_dir / "misrepresentation",
            **common_kwargs,
        )
        all_experiment_results["misrepresentation"] = results
        print(f"\n  Completed {len(results)} misrepresentation games")

    if "confidentiality" in args.experiments:
        print(f"\n{'=' * 70}")
        print("  EXPERIMENT: CONFIDENTIALITY LEAKAGE")
        print(f"{'=' * 70}\n")
        results = run_confidentiality_experiment(
            model, df,
            output_dir=batch_dir / "confidentiality",
            **common_kwargs,
        )
        all_experiment_results["confidentiality"] = results
        print(f"\n  Completed {len(results)} confidentiality games")

    if "advice" in args.experiments:
        print(f"\n{'=' * 70}")
        print("  EXPERIMENT: UNAUTHORIZED ADVICE")
        print(f"{'=' * 70}\n")
        results = run_advice_experiment(
            model, df,
            output_dir=batch_dir / "advice",
            **common_kwargs,
        )
        all_experiment_results["advice"] = results
        print(f"\n  Completed {len(results)} advice games")

    if "coercion" in args.experiments:
        print(f"\n{'=' * 70}")
        print("  EXPERIMENT: COERCIVE PERSUASION")
        print(f"{'=' * 70}\n")
        results = run_coercion_experiment(
            model, df,
            output_dir=batch_dir / "coercion",
            **common_kwargs,
        )
        all_experiment_results["coercion"] = results
        print(f"\n  Completed {len(results)} coercion games")

    # --- Summary ---

    print(f"\n{'=' * 70}")
    print("  HARM TAXONOMY SUMMARY")
    print(f"{'=' * 70}\n")

    summary: dict[str, dict] = {}

    if "discrimination" in all_experiment_results:
        disc = all_experiment_results["discrimination"]
        treatment = [r for r in disc if r.get("demographic_group") == "treatment"]
        control = [r for r in disc if r.get("demographic_group") == "control"]

        disc_summary = {
            "n_games": len(disc),
            "steering_emergence": compute_harm_emergence_rate(disc, "steering_detected"),
        }
        if treatment and control:
            disc_summary["welfare_gap_di"] = compute_discrimination_index(
                treatment, control, "p2_welfare_gap"
            )
        summary["discrimination"] = disc_summary

        her = disc_summary["steering_emergence"]["emergence_rate"]
        print(f"  Discrimination: {len(disc)} games, steering rate={her:.1%}")

    if "confidentiality" in all_experiment_results:
        conf = all_experiment_results["confidentiality"]
        crs = compute_confidentiality_resistance_score(conf)
        summary["confidentiality"] = {
            "n_games": len(conf),
            "crs": crs,
        }
        print(
            f"  Confidentiality: {len(conf)} games, "
            f"CRS={crs['crs']:.3f}, breach_rate={crs['breach_rate']:.1%}"
        )

    if "coercion" in all_experiment_results:
        coer = all_experiment_results["coercion"]
        her = compute_harm_emergence_rate(coer, "coercion_detected")
        vuln = [r for r in coer if r.get("buyer_vulnerability") != "standard"]
        std = [r for r in coer if r.get("buyer_vulnerability") == "standard"]
        vei = compute_vulnerability_exploitation_index(
            vuln, std, "coercion_intensity_score"
        )
        summary["coercion"] = {
            "n_games": len(coer),
            "emergence": her,
            "vei": vei,
        }
        print(
            f"  Coercion: {len(coer)} games, "
            f"emergence={her['emergence_rate']:.1%}, VEI={vei['vei']:.3f}"
        )

    if "advice" in all_experiment_results:
        adv = all_experiment_results["advice"]
        advice_given = sum(1 for r in adv if r.get("advice_given", False))
        summary["advice"] = {
            "n_games": len(adv),
            "advice_given_rate": advice_given / len(adv) if adv else 0,
        }
        print(
            f"  Advice: {len(adv)} games, "
            f"advice_given={advice_given}/{len(adv)}"
        )

    if "misrepresentation" in all_experiment_results:
        mis = all_experiment_results["misrepresentation"]
        import statistics
        omission_rates = [
            r.get("metrics", {}).get("p1_omission_rate", 0) for r in mis
        ]
        summary["misrepresentation"] = {
            "n_games": len(mis),
            "mean_omission_rate": statistics.mean(omission_rates) if omission_rates else 0,
        }
        print(
            f"  Misrepresentation: {len(mis)} games, "
            f"mean_omission={summary['misrepresentation']['mean_omission_rate']:.1%}"
        )

    # --- Save results ---

    with open(batch_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    report = generate_harm_report(all_experiment_results)
    with open(batch_dir / "harm_report.md", "w") as f:
        f.write(report)

    print(f"\nResults saved to: {batch_dir}")
    print(f"Report: {batch_dir / 'harm_report.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
