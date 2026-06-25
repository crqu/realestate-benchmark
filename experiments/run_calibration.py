#!/usr/bin/env python3
"""Calibrate Gemini judge against Claude judge on existing experiments.

Loads existing LLM-judged experiment transcripts, re-evaluates them with a
Gemini judge, and generates a comparison report.

Usage:
    venv/bin/python experiments/run_calibration.py \
        --project-id itpc-gcp-ai-eng-claude \
        --judge-model gemini-2.5-pro
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realestate_benchmark.data.ames import get_property, load_ames_data
from realestate_benchmark.data.properties import extract_defects, partition_property
from realestate_benchmark.environment.database import Database
from realestate_benchmark.evaluation.llm_judge import compute_llm_metrics
from realestate_benchmark.models.gemini import GeminiModel

CONTINUOUS_METRICS = [
    "p1_commission",
    "p1_omission_rate",
    "p1_framing",
    "p3_influence_legitimacy",
    "p4_violation_rate",
]

NORM_KEYS = [
    "N1_no_fraud",
    "N2_respond_to_inquiry",
    "N3_formal_disclosure",
    "N4_no_fabricated_urgency",
    "N5_fair_dealing",
]


def discover_experiments(results_dir: Path, pattern: str = "exp_*"):
    """Find all LLM-judged experiments matching the pattern."""
    experiments = []
    for exp_dir in sorted(results_dir.glob(pattern)):
        metrics_path = exp_dir / "metrics.json"
        property_path = exp_dir / "property.json"
        db_path = exp_dir / "game.db"

        if not all(p.exists() for p in [metrics_path, property_path, db_path]):
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        if metrics.get("evaluation_mode") != "LLM-as-judge":
            continue

        with open(property_path) as f:
            prop_info = json.load(f)

        experiments.append({
            "dir": exp_dir,
            "name": exp_dir.name,
            "metrics": metrics,
            "property_id": prop_info["property_id"],
            "true_value": prop_info["true_value"],
            "asking_price": prop_info["asking_price"],
        })

    return experiments


def load_experiment_data(exp, df):
    """Load transcript, state, and property data for an experiment."""
    db = Database(str(exp["dir"] / "game.db"))

    cursor = db._connection.cursor()
    cursor.execute("SELECT game_id, config FROM games LIMIT 1")
    row = cursor.fetchone()
    cursor.close()

    game_id = row["game_id"]
    config = json.loads(row["config"]) if row["config"] else {}
    max_turns = config.get("max_turns", 50)

    transcript = db.load_transcript(game_id)
    state = db.load_state(game_id)

    property_data = get_property(exp["property_id"], df)
    _, hidden = partition_property(property_data)
    defects = extract_defects(hidden)

    db._connection.close()

    return transcript, state, defects, property_data, max_turns


def compute_agreement(claude_metrics, gemini_metrics, tolerance=0.1):
    """Compare two metric dicts, return agreement statistics."""
    results = {}

    for key in CONTINUOUS_METRICS:
        c_val = claude_metrics.get(key, 0.0)
        g_val = gemini_metrics.get(key, 0.0)
        diff = g_val - c_val
        results[key] = {
            "claude": c_val,
            "gemini": g_val,
            "diff": diff,
            "abs_diff": abs(diff),
            "agree": abs(diff) <= tolerance,
        }

    c_norms = claude_metrics.get("p4_norms", {})
    g_norms = gemini_metrics.get("p4_norms", {})
    norm_results = {}
    for norm_key in NORM_KEYS:
        c_val = c_norms.get(norm_key, True)
        g_val = g_norms.get(norm_key, True)
        norm_results[norm_key] = {
            "claude": c_val,
            "gemini": g_val,
            "agree": c_val == g_val,
        }
    results["norms"] = norm_results

    continuous_agree = sum(1 for k in CONTINUOUS_METRICS if results[k]["agree"])
    norm_agree = sum(1 for v in norm_results.values() if v["agree"])
    total = len(CONTINUOUS_METRICS) + len(NORM_KEYS)
    results["overall_agreement"] = (continuous_agree + norm_agree) / total

    return results


def compute_cohens_kappa(comparisons):
    """Compute Cohen's kappa for binary norm agreement across experiments."""
    all_norm_pairs = []
    for comp in comparisons:
        for norm_key in NORM_KEYS:
            nr = comp["agreement"]["norms"][norm_key]
            all_norm_pairs.append((nr["claude"], nr["gemini"]))

    if not all_norm_pairs:
        return 0.0

    n = len(all_norm_pairs)
    agree = sum(1 for c, g in all_norm_pairs if c == g)
    p_o = agree / n

    c_true = sum(1 for c, _ in all_norm_pairs if c)
    g_true = sum(1 for _, g in all_norm_pairs if g)
    p_e = (c_true * g_true + (n - c_true) * (n - g_true)) / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def generate_html_report(comparisons, output_path, judge_model_name):
    """Generate HTML calibration report."""
    n = len(comparisons)

    # Aggregate statistics
    metric_stats = {}
    for key in CONTINUOUS_METRICS:
        diffs = [c["agreement"][key]["diff"] for c in comparisons]
        abs_diffs = [c["agreement"][key]["abs_diff"] for c in comparisons]
        agrees = [c["agreement"][key]["agree"] for c in comparisons]
        metric_stats[key] = {
            "mean_diff": sum(diffs) / n,
            "mean_abs_diff": sum(abs_diffs) / n,
            "agreement_rate": sum(agrees) / n,
            "max_abs_diff": max(abs_diffs),
        }

    norm_stats = {}
    for norm_key in NORM_KEYS:
        agrees = [c["agreement"]["norms"][norm_key]["agree"] for c in comparisons]
        norm_stats[norm_key] = sum(agrees) / n

    kappa = compute_cohens_kappa(comparisons)
    overall_agreement = sum(c["agreement"]["overall_agreement"] for c in comparisons) / n

    # Build comparison rows
    exp_rows = ""
    for comp in comparisons:
        exp_name = comp["experiment"]
        ag = comp["agreement"]
        exp_rows += f"""
        <tr>
            <td class="exp-name">{exp_name}</td>"""
        for key in CONTINUOUS_METRICS:
            c_val = ag[key]["claude"]
            g_val = ag[key]["gemini"]
            diff = ag[key]["diff"]
            cls = "agree" if ag[key]["agree"] else "disagree"
            exp_rows += f"""
            <td class="{cls}">{c_val:.1%}</td>
            <td class="{cls}">{g_val:.1%}</td>
            <td class="{cls}">{diff:+.1%}</td>"""
        exp_rows += "\n        </tr>"

    # Build norm comparison rows
    norm_rows = ""
    for comp in comparisons:
        exp_name = comp["experiment"]
        ag = comp["agreement"]["norms"]
        norm_rows += f"""
        <tr>
            <td class="exp-name">{exp_name}</td>"""
        for nk in NORM_KEYS:
            c_val = ag[nk]["claude"]
            g_val = ag[nk]["gemini"]
            cls = "agree" if ag[nk]["agree"] else "disagree"
            c_icon = "&#10003;" if c_val else "&#10007;"
            g_icon = "&#10003;" if g_val else "&#10007;"
            norm_rows += f"""
            <td class="{cls}">{c_icon} / {g_icon}</td>"""
        norm_rows += "\n        </tr>"

    # Bias summary rows
    bias_rows = ""
    for key in CONTINUOUS_METRICS:
        s = metric_stats[key]
        direction = "Gemini stricter" if s["mean_diff"] > 0.005 else (
            "Gemini more lenient" if s["mean_diff"] < -0.005 else "No systematic bias"
        )
        bias_rows += f"""
        <tr>
            <td>{key}</td>
            <td>{s['mean_diff']:+.3f}</td>
            <td>{s['mean_abs_diff']:.3f}</td>
            <td>{s['max_abs_diff']:.3f}</td>
            <td>{s['agreement_rate']:.0%}</td>
            <td>{direction}</td>
        </tr>"""

    # Norm agreement summary
    norm_summary_rows = ""
    for nk in NORM_KEYS:
        rate = norm_stats[nk]
        norm_summary_rows += f"""
        <tr>
            <td>{nk}</td>
            <td>{rate:.0%}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Judge Calibration: Claude vs {judge_model_name}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
    h2 {{ color: #16213e; margin-top: 30px; }}
    .summary-box {{ background: #fff; border-radius: 8px; padding: 20px; margin: 15px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 15px; }}
    .stat-card {{ background: #f8f9fa; border-radius: 6px; padding: 15px; text-align: center; }}
    .stat-card .value {{ font-size: 2em; font-weight: bold; color: #16213e; }}
    .stat-card .label {{ color: #666; font-size: 0.9em; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: #fff;
             border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    th {{ background: #16213e; color: white; padding: 10px 8px; text-align: center;
          font-size: 0.85em; }}
    td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eee; font-size: 0.85em; }}
    td.exp-name {{ text-align: left; font-family: monospace; font-size: 0.8em; }}
    .agree {{ background: #d4edda; }}
    .disagree {{ background: #f8d7da; }}
    .finding {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px;
                margin: 10px 0; border-radius: 0 8px 8px 0; }}
    .meta {{ color: #666; font-size: 0.9em; margin-top: 30px; }}
</style>
</head>
<body>

<h1>LLM Judge Calibration Report</h1>
<p>Comparing Claude (original judge) vs <strong>{judge_model_name}</strong> on {n} experiments</p>

<div class="summary-box">
<h2>Overall Agreement</h2>
<div class="summary-grid">
    <div class="stat-card">
        <div class="value">{overall_agreement:.0%}</div>
        <div class="label">Overall Agreement</div>
    </div>
    <div class="stat-card">
        <div class="value">{kappa:.2f}</div>
        <div class="label">Cohen's Kappa (norms)</div>
    </div>
    <div class="stat-card">
        <div class="value">{n}</div>
        <div class="label">Experiments Compared</div>
    </div>
</div>
</div>

<h2>Systematic Bias Analysis</h2>
<p>Positive mean diff = Gemini finds MORE violations than Claude. Negative = fewer.</p>
<table>
<tr>
    <th>Metric</th>
    <th>Mean Diff (G-C)</th>
    <th>Mean |Diff|</th>
    <th>Max |Diff|</th>
    <th>Agreement Rate</th>
    <th>Direction</th>
</tr>
{bias_rows}
</table>

<h2>Norm Agreement Summary</h2>
<table>
<tr><th>Norm</th><th>Agreement Rate</th></tr>
{norm_summary_rows}
</table>

<h2>Per-Experiment Continuous Metrics</h2>
<p>Each cell shows Claude / Gemini / Diff. Green = agree within 10pp, Red = disagree.</p>
<table>
<tr>
    <th>Experiment</th>
    <th colspan="3">P1 Commission</th>
    <th colspan="3">P1 Omission</th>
    <th colspan="3">P1 Framing</th>
    <th colspan="3">P3 Influence</th>
    <th colspan="3">P4 Violation</th>
</tr>
<tr>
    <th></th>
    {"".join('<th>C</th><th>G</th><th>&Delta;</th>' for _ in CONTINUOUS_METRICS)}
</tr>
{exp_rows}
</table>

<h2>Per-Experiment Norm Compliance</h2>
<p>Each cell shows Claude / Gemini (&#10003; = compliant, &#10007; = violation). Green = agree.</p>
<table>
<tr>
    <th>Experiment</th>
    {"".join(f'<th>{nk}</th>' for nk in NORM_KEYS)}
</tr>
{norm_rows}
</table>

<div class="meta">
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Judge model: {judge_model_name}</p>
    <p>Agreement tolerance: 10 percentage points for continuous metrics</p>
</div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Gemini judge against Claude judge on existing experiments"
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-east5")
    parser.add_argument("--judge-model", default="gemini-2.5-pro")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output", default=None)
    parser.add_argument("--experiment-pattern", default="exp_*",
                        help="Glob pattern to filter experiments")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Limit number of experiments to process")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"docs/calibration_{args.judge_model.replace('-', '_')}.html"

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    print("Discovering LLM-judged experiments...")
    experiments = discover_experiments(results_dir, args.experiment_pattern)
    if args.max_experiments:
        experiments = experiments[:args.max_experiments]
    print(f"Found {len(experiments)} LLM-judged experiments")

    if not experiments:
        print("No experiments found. Check --results-dir and --experiment-pattern.")
        return 1

    # Load Ames dataset once
    print("Loading Ames dataset...")
    df = load_ames_data()

    # Initialize Gemini judge
    print(f"Initializing {args.judge_model} judge...")
    gemini_judge = GeminiModel(
        project_id=args.project_id,
        region=args.region,
        model=args.judge_model,
        json_mode=True,
    )

    # Re-evaluate each experiment
    comparisons = []
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Re-evaluating {exp['name']}...")
        t0 = time.time()

        try:
            transcript, state, defects, property_data, max_turns = (
                load_experiment_data(exp, df)
            )

            gemini_metrics = compute_llm_metrics(
                gemini_judge, transcript, state, defects, property_data,
                exp["true_value"], max_turns=max_turns,
            )

            agreement = compute_agreement(exp["metrics"], gemini_metrics)

            comparisons.append({
                "experiment": exp["name"],
                "claude_metrics": exp["metrics"],
                "gemini_metrics": gemini_metrics,
                "agreement": agreement,
            })

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.0f}s — agreement: {agreement['overall_agreement']:.0%}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        time.sleep(2)

    if not comparisons:
        print("No experiments were successfully re-evaluated.")
        return 1

    # Generate report
    print(f"\nGenerating calibration report ({len(comparisons)} experiments)...")
    report_path = generate_html_report(comparisons, output_path, args.judge_model)
    print(f"Report saved to: {report_path}")

    # Save raw comparison data
    raw_path = output_path.with_suffix(".json")
    raw_data = []
    for comp in comparisons:
        entry = {
            "experiment": comp["experiment"],
            "agreement": comp["agreement"],
            "gemini_metrics": {
                k: v for k, v in comp["gemini_metrics"].items()
                if k != "llm_evidence"
            },
        }
        raw_data.append(entry)
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2, default=str)
    print(f"Raw data saved to: {raw_path}")

    # Print summary
    overall = sum(c["agreement"]["overall_agreement"] for c in comparisons) / len(comparisons)
    kappa = compute_cohens_kappa(comparisons)
    print(f"\n{'='*60}")
    print(f"  Overall agreement: {overall:.0%}")
    print(f"  Cohen's kappa (norms): {kappa:.2f}")
    print(f"  Experiments: {len(comparisons)}")
    print(f"{'='*60}")

    for key in CONTINUOUS_METRICS:
        diffs = [c["agreement"][key]["diff"] for c in comparisons]
        mean_diff = sum(diffs) / len(diffs)
        direction = "stricter" if mean_diff > 0.005 else (
            "more lenient" if mean_diff < -0.005 else "similar"
        )
        print(f"  {key}: mean diff {mean_diff:+.3f} (Gemini {direction})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
