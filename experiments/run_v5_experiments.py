#!/usr/bin/env python3
"""V5 experiment runner: calibrated judge, multiple seeds, statistical analysis.

Implements NEXT_STEPS.md requirements:
- Gemini 2.5 Pro as independent judge (eliminates same-model-as-judge confound)
- 5 seeds per condition (statistical power)
- Full model x buyer matrix
- New metrics: DCI, SWOR, SOI, P3a/P3b/P3c decomposition
- PSI (Pressure Sensitivity Index) computation
- Statistical tests and confidence intervals
- HTML report generation

Usage:
    venv/bin/python experiments/run_v5_experiments.py \
        --project-id itpc-gcp-ai-eng-claude \
        --experiment misrepresentation \
        --models claude-sonnet-4-5@20250929 claude-3-5-haiku@20241022 \
        --num-seeds 5
"""

import argparse
import json
import math
import statistics
import sys
import time
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
from realestate_benchmark.models.gemini import GeminiModel
from realestate_benchmark.models.vertex import VertexModel


def compute_ci(values, confidence=0.95):
    """Compute mean, SD, and confidence interval for a list of values."""
    if not values:
        return {"mean": 0.0, "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return {"mean": mean, "sd": 0.0, "ci_low": mean, "ci_high": mean, "n": n}
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    t_crit = {
        3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 15: 2.145, 20: 2.093, 30: 2.045,
    }
    t = t_crit.get(n, 1.96)
    margin = t * se
    return {
        "mean": mean,
        "sd": sd,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "n": n,
        "se": se,
    }


def mann_whitney_u(x, y):
    """Simple Mann-Whitney U test (two-sided). Returns U statistic and approximate p-value."""
    if not x or not y:
        return {"U": 0, "p": 1.0, "significant": False}

    nx, ny = len(x), len(y)
    combined = [(v, "x") for v in x] + [(v, "y") for v in y]
    combined.sort(key=lambda t: t[0])

    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[id(combined[k])] = avg_rank
        i = j

    r_x = sum(ranks[id(t)] for t in combined if t[1] == "x")
    u_x = r_x - nx * (nx + 1) / 2
    u_y = nx * ny - u_x
    u = min(u_x, u_y)

    mu = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12) if (nx + ny + 1) > 0 else 1
    if sigma == 0:
        return {"U": u, "p": 1.0, "significant": False}

    z = abs((u - mu) / sigma)
    p = 2 * (1 - _norm_cdf(z))
    return {"U": u, "p": p, "z": z, "significant": p < 0.05}


def _norm_cdf(z):
    """Approximate standard normal CDF using Abramowitz & Stegun."""
    if z < 0:
        return 1 - _norm_cdf(-z)
    t = 1 / (1 + 0.2316419 * z)
    d = 0.3989422804014327
    p = d * math.exp(-z * z / 2) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
        t * (-1.821255978 + t * 1.330274429))))
    )
    return 1 - p


def compute_psi(results, metric_key):
    """Compute Pressure Sensitivity Index for a metric across pressure levels."""
    by_pressure = {}
    for r in results:
        pressure = r.get("seller_pressure", r.get("mortgage_pct", "unknown"))
        val = r.get("metrics", {}).get(metric_key, r.get(metric_key))
        if val is not None:
            by_pressure.setdefault(str(pressure), []).append(float(val))

    if len(by_pressure) < 2:
        return None

    pressure_keys = sorted(by_pressure.keys())
    low_key = pressure_keys[0]
    high_key = pressure_keys[-1]

    low_vals = by_pressure[low_key]
    high_vals = by_pressure[high_key]

    low_ci = compute_ci(low_vals)
    high_ci = compute_ci(high_vals)

    psi_value = high_ci["mean"] - low_ci["mean"]
    test = mann_whitney_u(low_vals, high_vals)

    return {
        "psi": psi_value,
        "low_pressure": {**low_ci, "level": low_key},
        "high_pressure": {**high_ci, "level": high_key},
        "test": test,
    }


def analyze_results(results, experiment_type):
    """Analyze experiment results with statistical tests."""
    analysis = {
        "experiment_type": experiment_type,
        "total_games": len(results),
        "completed_games": sum(1 for r in results if r.get("completed")),
        "failed_games": sum(1 for r in results if r is None),
    }

    core_metrics = [
        "p1_commission", "p1_omission_rate", "p1_framing",
        "p1_dci", "p1_swor", "p1_soi",
        "p3_influence_legitimacy", "p3a_fabricated_offers",
        "p3b_false_urgency", "p3c_emotional_exploitation",
        "p4_violation_rate",
    ]

    metric_stats = {}
    for metric in core_metrics:
        values = []
        for r in results:
            v = r.get("metrics", {}).get(metric)
            if v is not None:
                values.append(float(v))
        if values:
            metric_stats[metric] = compute_ci(values)

    analysis["metric_stats"] = metric_stats

    if experiment_type == "misrepresentation":
        by_buyer = {}
        for r in results:
            buyer = r.get("buyer_sophistication", "unknown")
            by_buyer.setdefault(buyer, []).append(r)

        buyer_comparison = {}
        for buyer, buyer_results in by_buyer.items():
            buyer_stats = {}
            for metric in core_metrics:
                values = [
                    r.get("metrics", {}).get(metric)
                    for r in buyer_results
                    if r.get("metrics", {}).get(metric) is not None
                ]
                if values:
                    buyer_stats[metric] = compute_ci([float(v) for v in values])
            buyer_comparison[buyer] = {
                "n": len(buyer_results),
                "stats": buyer_stats,
            }
        analysis["by_buyer"] = buyer_comparison

        psi_metrics = ["p1_commission", "p1_omission_rate", "p1_dci", "p1_framing"]
        psi_results = {}
        for metric in psi_metrics:
            psi = compute_psi(results, metric)
            if psi:
                psi_results[metric] = psi
        analysis["psi"] = psi_results

        buyers = sorted(by_buyer.keys())
        if len(buyers) >= 2:
            cross_buyer_tests = {}
            for i, b1 in enumerate(buyers):
                for b2 in buyers[i + 1:]:
                    pair_key = f"{b1}_vs_{b2}"
                    pair_tests = {}
                    for metric in core_metrics:
                        v1 = [
                            float(r["metrics"][metric])
                            for r in by_buyer[b1]
                            if r.get("metrics", {}).get(metric) is not None
                        ]
                        v2 = [
                            float(r["metrics"][metric])
                            for r in by_buyer[b2]
                            if r.get("metrics", {}).get(metric) is not None
                        ]
                        if v1 and v2:
                            pair_tests[metric] = mann_whitney_u(v1, v2)
                    cross_buyer_tests[pair_key] = pair_tests
            analysis["cross_buyer_tests"] = cross_buyer_tests

    return analysis


def generate_html_report(analysis, results, output_path):
    """Generate an HTML report from experiment analysis."""
    exp_type = analysis["experiment_type"]
    metric_stats = analysis.get("metric_stats", {})
    by_buyer = analysis.get("by_buyer", {})
    psi = analysis.get("psi", {})
    cross_tests = analysis.get("cross_buyer_tests", {})

    metric_labels = {
        "p1_commission": "P1 Commission (False Claims)",
        "p1_omission_rate": "P1 Omission Rate",
        "p1_framing": "P1 Framing Bias",
        "p1_dci": "Disclosure Completeness (DCI)",
        "p1_swor": "Severity-Weighted Omission (SWOR)",
        "p1_soi": "Strategic Omission Index (SOI)",
        "p3_influence_legitimacy": "P3 Influence (Overall)",
        "p3a_fabricated_offers": "P3a Fabricated Offers",
        "p3b_false_urgency": "P3b False Urgency",
        "p3c_emotional_exploitation": "P3c Emotional Exploitation",
        "p4_violation_rate": "P4 Norm Violation Rate",
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BIAI V5 Experiment Report — {exp_type.title()}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
h3 {{ color: #0f3460; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
         box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: center; }}
th {{ background: #16213e; color: white; font-weight: 600; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.sig {{ color: #e74c3c; font-weight: bold; }}
.not-sig {{ color: #95a5a6; }}
.metric-card {{ background: white; border-radius: 8px; padding: 16px; margin: 10px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
              gap: 12px; }}
.psi-positive {{ color: #e74c3c; }}
.psi-negative {{ color: #27ae60; }}
.summary-box {{ background: #16213e; color: white; padding: 20px; border-radius: 8px;
                margin: 20px 0; }}
.summary-box h3 {{ color: #e8e8e8; margin-top: 0; }}
</style>
</head>
<body>
<h1>BIAI V5: {exp_type.title()} Experiment Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="summary-box">
<h3>Experiment Summary</h3>
<p>Total games: {analysis['total_games']} | Completed: {analysis['completed_games']} |
   Failed: {analysis['failed_games']}</p>
</div>

<h2>Overall Metric Statistics</h2>
<table>
<tr><th>Metric</th><th>Mean</th><th>SD</th><th>95% CI</th><th>N</th></tr>
"""

    for metric, stats in sorted(metric_stats.items()):
        label = metric_labels.get(metric, metric)
        html += (
            f"<tr><td style='text-align:left'>{label}</td>"
            f"<td>{stats['mean']:.3f}</td>"
            f"<td>{stats['sd']:.3f}</td>"
            f"<td>[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]</td>"
            f"<td>{stats['n']}</td></tr>\n"
        )

    html += "</table>\n"

    if by_buyer:
        html += "<h2>By Buyer Sophistication</h2>\n"
        buyers = sorted(by_buyer.keys())
        key_metrics = ["p1_commission", "p1_omission_rate", "p1_dci", "p1_swor",
                       "p3_influence_legitimacy", "p4_violation_rate"]

        html += "<table>\n<tr><th>Metric</th>"
        for b in buyers:
            html += f"<th>{b.title()} (n={by_buyer[b]['n']})</th>"
        html += "</tr>\n"

        for metric in key_metrics:
            label = metric_labels.get(metric, metric)
            html += f"<tr><td style='text-align:left'>{label}</td>"
            for b in buyers:
                stats = by_buyer[b]["stats"].get(metric)
                if stats:
                    html += f"<td>{stats['mean']:.3f} &plusmn; {stats['sd']:.3f}</td>"
                else:
                    html += "<td>—</td>"
            html += "</tr>\n"
        html += "</table>\n"

    if psi:
        html += "<h2>Pressure Sensitivity Index (PSI)</h2>\n"
        html += "<p>PSI = metric(high_pressure) - metric(low_pressure). "
        html += "Positive = integrity degrades under pressure.</p>\n"
        html += "<table>\n<tr><th>Metric</th><th>PSI</th><th>Low Pressure</th>"
        html += "<th>High Pressure</th><th>p-value</th><th>Significant</th></tr>\n"

        for metric, psi_data in sorted(psi.items()):
            label = metric_labels.get(metric, metric)
            psi_val = psi_data["psi"]
            psi_class = "psi-positive" if psi_val > 0 else "psi-negative"
            sig_class = "sig" if psi_data["test"]["significant"] else "not-sig"
            sig_text = "Yes" if psi_data["test"]["significant"] else "No"

            html += (
                f"<tr><td style='text-align:left'>{label}</td>"
                f"<td class='{psi_class}'>{psi_val:+.3f}</td>"
                f"<td>{psi_data['low_pressure']['mean']:.3f} &plusmn; "
                f"{psi_data['low_pressure']['sd']:.3f} ({psi_data['low_pressure']['level']})</td>"
                f"<td>{psi_data['high_pressure']['mean']:.3f} &plusmn; "
                f"{psi_data['high_pressure']['sd']:.3f} ({psi_data['high_pressure']['level']})</td>"
                f"<td>{psi_data['test']['p']:.4f}</td>"
                f"<td class='{sig_class}'>{sig_text}</td></tr>\n"
            )
        html += "</table>\n"

    if cross_tests:
        html += "<h2>Cross-Buyer Statistical Tests (Mann-Whitney U)</h2>\n"
        for pair, tests in sorted(cross_tests.items()):
            sig_tests = {m: t for m, t in tests.items() if t.get("significant")}
            if not sig_tests and not tests:
                continue

            html += f"<h3>{pair.replace('_', ' ').title()}</h3>\n"
            html += "<table>\n<tr><th>Metric</th><th>U</th><th>z</th>"
            html += "<th>p-value</th><th>Significant</th></tr>\n"

            for metric, test in sorted(tests.items()):
                label = metric_labels.get(metric, metric)
                sig_class = "sig" if test.get("significant") else "not-sig"
                sig_text = "p < 0.05" if test.get("significant") else "n.s."
                html += (
                    f"<tr><td style='text-align:left'>{label}</td>"
                    f"<td>{test.get('U', 0):.1f}</td>"
                    f"<td>{test.get('z', 0):.3f}</td>"
                    f"<td>{test['p']:.4f}</td>"
                    f"<td class='{sig_class}'>{sig_text}</td></tr>\n"
                )
            html += "</table>\n"

    html += """
<h2>Individual Game Results</h2>
<details>
<summary>Click to expand ({n_games} games)</summary>
<table>
<tr><th>#</th><th>Property</th><th>Buyer</th><th>Pressure</th>
<th>Completed</th><th>Price</th><th>Commission</th><th>Omission</th>
<th>DCI</th><th>SWOR</th></tr>
""".format(n_games=len(results))

    for i, r in enumerate(results):
        if r is None:
            continue
        m = r.get("metrics", {})
        completed = "Yes" if r.get("completed") else "No"
        price = f"${r.get('final_price', 0):,}" if r.get("completed") else "—"
        buyer = r.get("buyer_sophistication", "?")
        pressure = r.get("seller_pressure", r.get("mortgage_pct", "?"))

        html += (
            f"<tr><td>{i+1}</td>"
            f"<td>{r.get('property_id', '?')}</td>"
            f"<td>{buyer}</td>"
            f"<td>{pressure}</td>"
            f"<td>{completed}</td>"
            f"<td>{price}</td>"
            f"<td>{m.get('p1_commission', 0):.2f}</td>"
            f"<td>{m.get('p1_omission_rate', 0):.2f}</td>"
            f"<td>{m.get('p1_dci', 0):.2f}</td>"
            f"<td>{m.get('p1_swor', 0):.2f}</td>"
            f"</tr>\n"
        )

    html += """</table>
</details>

<footer style="margin-top: 40px; padding: 20px; border-top: 1px solid #dee2e6;
               color: #6c757d; font-size: 0.9em;">
<p>BIAI (Behavioral Integrity in Agent Interactions) — CoLM 2026 Workshop</p>
<p>New metrics: DCI (Disclosure Completeness Index), SWOR (Severity-Weighted Omission Rate),
   SOI (Strategic Omission Index), P3a/P3b/P3c decomposition, PSI (Pressure Sensitivity Index)</p>
</footer>
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="V5 experiment runner with statistical analysis")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-east5")
    parser.add_argument(
        "--models", nargs="+",
        default=["claude-sonnet-4-5@20250929"],
        help="Agent models to test",
    )
    parser.add_argument(
        "--experiment",
        choices=["misrepresentation", "discrimination", "confidentiality",
                 "advice", "coercion", "all"],
        default="misrepresentation",
    )
    parser.add_argument("--judge-model", default="gemini-2.5-pro")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--output-dir", default="experiments/harm_results")
    parser.add_argument(
        "--max-game-workers", type=int, default=4,
        help="Max parallel games within each experiment (default: 4)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.output_dir) / f"v5_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print("  BIAI V5 EXPERIMENTS — Calibrated Judge + Statistical Power")
    print(f"{'=' * 70}")
    print(f"Output:     {batch_dir}")
    print(f"Models:     {', '.join(args.models)}")
    print(f"Judge:      {args.judge_model}")
    print(f"Experiment: {args.experiment}")
    print(f"Seeds:      {args.num_seeds} (base: {args.base_seed})")
    print(f"Max turns:  {args.max_turns}")
    print()

    with open(batch_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    judge = GeminiModel(
        project_id=args.project_id,
        region=args.region,
        model=args.judge_model,
        json_mode=True,
    )
    print(f"Judge model initialized: {args.judge_model}")

    df = load_ames_data()
    print(f"Loaded {len(df)} properties from Ames dataset\n")

    experiments = (
        ["misrepresentation", "discrimination", "confidentiality", "advice", "coercion"]
        if args.experiment == "all" else [args.experiment]
    )

    all_results = {}

    for agent_model_name in args.models:
        model = VertexModel(
            project_id=args.project_id,
            region=args.region,
            model=agent_model_name,
        )
        model_short = agent_model_name.split("@")[0].replace("claude-", "")

        for exp_name in experiments:
            print(f"\n{'=' * 70}")
            print(f"  {agent_model_name} — {exp_name.upper()}")
            print(f"{'=' * 70}\n")

            exp_dir = batch_dir / model_short / exp_name
            t0 = time.time()

            common_kwargs = {
                "num_seeds": args.num_seeds,
                "base_seed": args.base_seed,
                "max_turns": args.max_turns,
                "use_llm_judge": True,
                "judge_model": judge,
                "output_dir": exp_dir,
                "max_game_workers": args.max_game_workers,
            }

            results = []
            try:
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
            except Exception as e:
                print(f"  EXPERIMENT FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

            elapsed = time.time() - t0
            key = f"{model_short}/{exp_name}"
            all_results[key] = results

            print(f"\n  {len(results)} games completed in {elapsed:.0f}s")

            analysis = analyze_results(results, exp_name)

            with open(exp_dir / "analysis.json", "w") as f:
                json.dump(analysis, f, indent=2, default=str)

            html_path = exp_dir / "report.html"
            generate_html_report(analysis, results, html_path)
            print(f"  Report: {html_path}")

    generate_cross_model_report(all_results, batch_dir)

    print(f"\n{'=' * 70}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results: {batch_dir}")


def generate_cross_model_report(all_results, batch_dir):
    """Generate a cross-model comparison HTML report."""
    if not all_results:
        return

    models = sorted({k.split("/")[0] for k in all_results})
    experiments = sorted({k.split("/")[1] for k in all_results if "/" in k})

    metric_labels = {
        "p1_commission": "Commission",
        "p1_omission_rate": "Omission",
        "p1_dci": "DCI",
        "p1_swor": "SWOR",
        "p1_soi": "SOI",
        "p3_influence_legitimacy": "P3 Influence",
        "p4_violation_rate": "P4 Violations",
    }

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>BIAI V5 Cross-Model Comparison</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px;
       margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
h2 {{ color: #16213e; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
         box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: center; }}
th {{ background: #16213e; color: white; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.best {{ background: #d4edda !important; font-weight: bold; }}
.worst {{ background: #f8d7da !important; }}
</style></head><body>
<h1>BIAI V5: Cross-Model Comparison</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>Models: {', '.join(models)}</p>
"""

    for exp_name in experiments:
        html += f"<h2>{exp_name.title()}</h2>\n"
        html += "<table>\n<tr><th>Metric</th>"
        for m in models:
            html += f"<th>{m}</th>"
        html += "</tr>\n"

        for metric, label in metric_labels.items():
            html += f"<tr><td style='text-align:left'>{label}</td>"
            values_by_model = {}
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
                    values_by_model[m] = ci
                    html += f"<td>{ci['mean']:.3f} &plusmn; {ci['sd']:.3f}</td>"
                else:
                    html += "<td>—</td>"
            html += "</tr>\n"

        html += "</table>\n"

    html += "</body></html>"

    report_path = batch_dir / "cross_model_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nCross-model report: {report_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
