#!/usr/bin/env python3
"""Parallel V5 experiment runner for maximum efficiency.

Launches multiple experiment processes concurrently, one per (model, experiment) pair.
Monitors all processes and aggregates results when complete.

Usage:
    venv/bin/python experiments/run_v5_parallel.py \
        --project-id itpc-gcp-ai-eng-claude \
        --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \
        --experiments misrepresentation discrimination confidentiality \
        --judge-model gemini-2.5-pro \
        --num-seeds 5 \
        --max-workers 6
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realestate_benchmark.data.ames import load_ames_data
from realestate_benchmark.models.gemini import GeminiModel
from realestate_benchmark.models.vertex import VertexModel

# Import the experiment runners
from run_harm_experiments import (
    run_advice_experiment,
    run_coercion_experiment,
    run_confidentiality_experiment,
    run_discrimination_experiment,
    run_misrepresentation_experiment,
    run_tool_use_experiment,
)


ALL_EXPERIMENTS = [
    "discrimination",
    "misrepresentation",
    "confidentiality",
    "advice",
    "coercion",
    "tool_use",
]


def run_single_experiment(args_dict):
    """Run a single (model, experiment) combination. Executed in subprocess."""
    model_name = args_dict["model"]
    exp_name = args_dict["experiment"]
    output_dir = Path(args_dict["output_dir"])
    project_id = args_dict["project_id"]
    region = args_dict["region"]
    judge_model_name = args_dict["judge_model"]
    num_seeds = args_dict["num_seeds"]
    base_seed = args_dict["base_seed"]
    max_turns = args_dict["max_turns"]
    max_game_workers = args_dict.get("max_game_workers", 1)

    model_short = model_name.split("@")[0].replace("claude-", "")
    exp_dir = output_dir / model_short / exp_name

    print(f"[{model_short}/{exp_name}] Starting...")
    t0 = time.time()

    try:
        # Initialize models
        model = VertexModel(project_id=project_id, region=region, model=model_name)

        judge = None
        if judge_model_name:
            if "gemini" in judge_model_name:
                judge = GeminiModel(
                    project_id=project_id,
                    region=region,
                    model=judge_model_name,
                    json_mode=True,
                )
            else:
                judge = VertexModel(
                    project_id=project_id,
                    region=region,
                    model=judge_model_name,
                )

        # Load dataset
        df = load_ames_data()

        common_kwargs = {
            "num_seeds": num_seeds,
            "base_seed": base_seed,
            "max_turns": max_turns,
            "use_llm_judge": True,
            "judge_model": judge,
            "output_dir": exp_dir,
            "max_game_workers": max_game_workers,
        }

        # Run the experiment
        results = []
        if exp_name == "misrepresentation":
            results = run_misrepresentation_experiment(model, df, **common_kwargs)
        elif exp_name == "discrimination":
            results = run_discrimination_experiment(model, df, **common_kwargs)
        elif exp_name == "confidentiality":
            results = run_confidentiality_experiment(model, df, **common_kwargs)
        elif exp_name == "advice":
            results = run_advice_experiment(model, df, **common_kwargs)
        elif exp_name == "coercion":
            results = run_coercion_experiment(model, df, **common_kwargs)
        elif exp_name == "tool_use":
            tool_kwargs = {k: v for k, v in common_kwargs.items() if k != "judge_model"}
            results = run_tool_use_experiment(model, df, **tool_kwargs)

        elapsed = time.time() - t0

        # Compute quick stats
        completed = sum(1 for r in results if r and r.get("completed"))
        total = len(results)

        return {
            "model": model_short,
            "experiment": exp_name,
            "success": True,
            "n_games": total,
            "n_completed": completed,
            "elapsed": elapsed,
            "output_dir": str(exp_dir),
            "results": results,
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{model_short}/{exp_name}] FAILED: {e}")
        import traceback
        traceback.print_exc()

        return {
            "model": model_short,
            "experiment": exp_name,
            "success": False,
            "error": str(e),
            "elapsed": elapsed,
            "output_dir": str(exp_dir),
            "results": [],
        }


def analyze_and_report(outcome, batch_dir):
    """Generate analysis and HTML report for a completed experiment."""
    if not outcome["success"] or not outcome["results"]:
        return

    from run_v5_experiments import analyze_results, generate_html_report

    results = outcome["results"]
    exp_type = outcome["experiment"]
    output_dir = Path(outcome["output_dir"])

    # Analyze results
    analysis = analyze_results(results, exp_type)

    # Save analysis JSON
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Generate HTML report
    html_path = output_dir / "report.html"
    generate_html_report(analysis, results, html_path)

    print(f"[{outcome['model']}/{exp_type}] Report: {html_path}")


def generate_master_report(all_outcomes, batch_dir):
    """Generate master cross-model comparison report."""
    from run_v5_experiments import compute_ci

    # Organize results by model and experiment
    all_results = {}
    for outcome in all_outcomes:
        if outcome["success"] and outcome["results"]:
            key = f"{outcome['model']}/{outcome['experiment']}"
            all_results[key] = outcome["results"]

    if not all_results:
        print("No successful experiments to report")
        return

    models = sorted({k.split("/")[0] for k in all_results})
    experiments = sorted({k.split("/")[1] for k in all_results if "/" in k})

    metric_labels = {
        "p1_commission": "P1 Commission",
        "p1_omission_rate": "P1 Omission",
        "p1_dci": "P1 DCI",
        "p1_swor": "P1 SWOR",
        "p1_soi": "P1 SOI",
        "p1_framing": "P1 Framing",
        "p3_influence_legitimacy": "P3 Influence",
        "p3a_fabricated_offers": "P3a Fabricated",
        "p3b_false_urgency": "P3b Urgency",
        "p3c_emotional_exploitation": "P3c Emotional",
        "p4_violation_rate": "P4 Violations",
    }

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>BIAI V5 Parallel Experiments — Master Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1600px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
.summary-box {{ background: #16213e; color: white; padding: 20px; border-radius: 8px;
                margin: 20px 0; }}
.summary-box h3 {{ color: #e8e8e8; margin-top: 0; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
               gap: 15px; margin: 20px 0; }}
.stat-card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
.stat-value {{ font-size: 2em; font-weight: bold; color: #16213e; }}
.stat-label {{ color: #6c757d; font-size: 0.9em; margin-top: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
         box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: center; }}
th {{ background: #16213e; color: white; font-weight: 600; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.best {{ background: #d4edda !important; font-weight: bold; }}
.worst {{ background: #f8d7da !important; }}
.success {{ color: #28a745; }}
.failure {{ color: #dc3545; }}
</style></head><body>
<h1>BIAI V5 Parallel Experiments — Master Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="summary-box">
<h3>Execution Summary</h3>
<div class="stats-grid">
"""

    # Overall stats
    total_jobs = len(all_outcomes)
    successful = sum(1 for o in all_outcomes if o["success"])
    failed = total_jobs - successful
    total_games = sum(o["n_games"] for o in all_outcomes if o["success"])
    completed_games = sum(o["n_completed"] for o in all_outcomes if o["success"])
    total_time = sum(o["elapsed"] for o in all_outcomes)

    html += f"""
<div class="stat-card">
    <div class="stat-value">{total_jobs}</div>
    <div class="stat-label">Total Jobs</div>
</div>
<div class="stat-card">
    <div class="stat-value class='success'">{successful}</div>
    <div class="stat-label">Successful</div>
</div>
<div class="stat-card">
    <div class="stat-value">{total_games}</div>
    <div class="stat-label">Total Games</div>
</div>
<div class="stat-card">
    <div class="stat-value">{completed_games}</div>
    <div class="stat-label">Completed Deals</div>
</div>
<div class="stat-card">
    <div class="stat-value">{total_time/60:.1f}m</div>
    <div class="stat-label">Total Time</div>
</div>
<div class="stat-card">
    <div class="stat-value">{total_time/60/max(total_jobs, 1):.1f}m</div>
    <div class="stat-label">Avg per Job</div>
</div>
</div></div>

<h2>Job Status</h2>
<table>
<tr><th>Model</th><th>Experiment</th><th>Status</th><th>Games</th><th>Completed</th><th>Time</th></tr>
"""

    for outcome in sorted(all_outcomes, key=lambda x: (x["model"], x["experiment"])):
        status = '<span class="success">✓ Success</span>' if outcome["success"] else '<span class="failure">✗ Failed</span>'
        time_str = f"{outcome['elapsed']/60:.1f}m"
        html += f"""<tr>
    <td>{outcome['model']}</td>
    <td>{outcome['experiment']}</td>
    <td>{status}</td>
    <td>{outcome.get('n_games', 0)}</td>
    <td>{outcome.get('n_completed', 0)}</td>
    <td>{time_str}</td>
</tr>
"""

    html += "</table>\n"

    # Cross-model metric comparison
    for exp_name in experiments:
        html += f"<h2>{exp_name.title()}</h2>\n"
        html += "<table>\n<tr><th>Metric</th>"
        for m in models:
            html += f"<th>{m}</th>"
        html += "</tr>\n"

        for metric, label in metric_labels.items():
            html += f"<tr><td style='text-align:left'>{label}</td>"
            model_values = {}
            for m in models:
                key = f"{m}/{exp_name}"
                results = all_results.get(key, [])
                vals = [
                    float(r["metrics"][metric])
                    for r in results
                    if r and r.get("metrics", {}).get(metric) is not None
                ]
                if vals:
                    ci = compute_ci(vals)
                    model_values[m] = ci["mean"]
                    html += f"<td>{ci['mean']:.3f} ± {ci['sd']:.3f}</td>"
                else:
                    html += "<td>—</td>"

            # Highlight best/worst
            if model_values:
                best_model = min(model_values, key=model_values.get)
                worst_model = max(model_values, key=model_values.get)
                # Re-render row with highlights (simplified - would need to track cell position)

            html += "</tr>\n"

        html += "</table>\n"

    html += """
<h2>Individual Reports</h2>
<ul>
"""

    for outcome in sorted(all_outcomes, key=lambda x: (x["model"], x["experiment"])):
        if outcome["success"]:
            report_path = Path(outcome["output_dir"]) / "report.html"
            if report_path.exists():
                rel_path = report_path.relative_to(batch_dir)
                html += f"<li><a href='{rel_path}'>{outcome['model']} — {outcome['experiment']}</a></li>\n"

    html += """</ul>

<footer style="margin-top: 40px; padding: 20px; border-top: 1px solid #dee2e6; color: #6c757d;">
<p>BIAI V5 Parallel Runner — Optimized for multi-model, multi-experiment efficiency</p>
</footer>
</body></html>"""

    master_report = batch_dir / "master_report.html"
    with open(master_report, "w") as f:
        f.write(html)

    print(f"\n{'=' * 70}")
    print(f"  MASTER REPORT: {master_report}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel V5 experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments for 3 models in parallel (recommended: max-workers = num_models)
  venv/bin/python experiments/run_v5_parallel.py \\
      --project-id itpc-gcp-ai-eng-claude \\
      --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \\
      --experiments all \\
      --judge-model gemini-2.5-pro \\
      --num-seeds 5 \\
      --max-workers 6

  # Run specific experiments with custom parallelism
  venv/bin/python experiments/run_v5_parallel.py \\
      --project-id itpc-gcp-ai-eng-claude \\
      --models claude-sonnet-4-5@20250929 \\
      --experiments misrepresentation discrimination \\
      --judge-model gemini-2.5-pro \\
      --num-seeds 5 \\
      --max-workers 2
        """
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-east5")
    parser.add_argument(
        "--models", nargs="+",
        default=["claude-sonnet-4-5@20250929"],
        help="Agent models to test (can specify multiple)",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        default=["all"],
        help="Experiments to run (space-separated) or 'all'",
    )
    parser.add_argument("--judge-model", default="gemini-2.5-pro")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--output-dir", default="experiments/harm_results")
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel experiment-level workers (default: min(6, num_jobs))",
    )
    parser.add_argument(
        "--max-game-workers", type=int, default=4,
        help="Max parallel games within each experiment (default: 4)",
    )
    args = parser.parse_args()

    # Expand "all" experiments
    if "all" in args.experiments:
        experiments = ALL_EXPERIMENTS
    else:
        experiments = args.experiments

    # Validate experiments
    for exp in experiments:
        if exp not in ALL_EXPERIMENTS:
            print(f"ERROR: Unknown experiment '{exp}'")
            print(f"Valid: {', '.join(ALL_EXPERIMENTS)}")
            return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.output_dir) / f"v5_parallel_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create job queue
    jobs = []
    for model in args.models:
        for exp in experiments:
            job = {
                "model": model,
                "experiment": exp,
                "project_id": args.project_id,
                "region": args.region,
                "judge_model": args.judge_model,
                "num_seeds": args.num_seeds,
                "base_seed": args.base_seed,
                "max_turns": args.max_turns,
                "output_dir": str(batch_dir),
                "max_game_workers": args.max_game_workers,
            }
            jobs.append(job)

    # Determine worker count
    max_workers = args.max_workers or min(6, len(jobs))

    print(f"{'=' * 70}")
    print("  BIAI V5 PARALLEL EXPERIMENT RUNNER")
    print(f"{'=' * 70}")
    print(f"Output:       {batch_dir}")
    print(f"Models:       {', '.join(args.models)}")
    print(f"Experiments:  {', '.join(experiments)}")
    print(f"Judge:        {args.judge_model}")
    print(f"Seeds:        {args.num_seeds} per condition")
    print(f"Max turns:    {args.max_turns}")
    print(f"Total jobs:   {len(jobs)}")
    print(f"Max workers:  {max_workers} (experiment-level)")
    print(f"Game workers: {args.max_game_workers} (per experiment)")
    print(f"{'=' * 70}\n")

    # Save config
    config = {
        **vars(args),
        "timestamp": timestamp,
        "total_jobs": len(jobs),
        "max_workers": max_workers,
    }
    with open(batch_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run jobs in parallel
    all_outcomes = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_job = {
            executor.submit(run_single_experiment, job): job
            for job in jobs
        }

        # Process completions as they finish
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                outcome = future.result()
                all_outcomes.append(outcome)

                model_short = outcome["model"]
                exp = outcome["experiment"]

                if outcome["success"]:
                    print(f"\n[{model_short}/{exp}] ✓ COMPLETED")
                    print(f"  Games: {outcome['n_completed']}/{outcome['n_games']}")
                    print(f"  Time: {outcome['elapsed']/60:.1f}m")

                    # Generate report for this experiment
                    analyze_and_report(outcome, batch_dir)
                else:
                    print(f"\n[{model_short}/{exp}] ✗ FAILED")
                    print(f"  Error: {outcome.get('error', 'Unknown')}")
                    print(f"  Time: {outcome['elapsed']/60:.1f}m")

                # Progress update
                completed = len(all_outcomes)
                print(f"\nProgress: {completed}/{len(jobs)} jobs completed")

            except Exception as e:
                print(f"ERROR processing job {job}: {e}")
                import traceback
                traceback.print_exc()

    total_elapsed = time.time() - t0

    # Generate master report
    print(f"\n{'=' * 70}")
    print("  GENERATING MASTER REPORT")
    print(f"{'=' * 70}\n")

    generate_master_report(all_outcomes, batch_dir)

    # Summary
    successful = sum(1 for o in all_outcomes if o["success"])
    failed = len(all_outcomes) - successful
    total_games = sum(o["n_games"] for o in all_outcomes if o["success"])

    print(f"{'=' * 70}")
    print("  PARALLEL EXECUTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time:      {total_elapsed/60:.1f} minutes")
    print(f"Jobs completed:  {successful}/{len(all_outcomes)}")
    print(f"Jobs failed:     {failed}")
    print(f"Total games:     {total_games}")
    print(f"Results:         {batch_dir}")
    print(f"Master report:   {batch_dir / 'master_report.html'}")
    print(f"{'=' * 70}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
