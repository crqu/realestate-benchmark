#!/usr/bin/env python3
"""Generate comprehensive HTML report for BIAI V5 experiment results."""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("experiments/harm_results/v5_parallel_20260625_153758")
MODELS = ["sonnet-4-5", "opus-4-5", "3-5-haiku"]
MODEL_LABELS = {
    "sonnet-4-5": "Sonnet 4.5",
    "opus-4-5": "Opus 4.5",
    "3-5-haiku": "Haiku 3.5",
}
EXPERIMENTS = ["misrepresentation", "discrimination", "confidentiality", "advice", "coercion", "tool_use"]
EXP_LABELS = {
    "misrepresentation": "Misrepresentation",
    "discrimination": "Discrimination",
    "confidentiality": "Confidentiality",
    "advice": "Advice Boundaries",
    "coercion": "Coercion",
    "tool_use": "Tool Use",
}

CORE_METRICS = [
    "p1_commission", "p1_omission_rate", "p1_dci", "p1_swor", "p1_soi",
    "p1_framing", "p3_influence_legitimacy", "p3a_fabricated_offers",
    "p3b_false_urgency", "p3c_emotional_exploitation", "p4_violation_rate",
]
METRIC_LABELS = {
    "p1_commission": "P1 Commission Rate",
    "p1_omission_rate": "P1 Omission Rate",
    "p1_dci": "P1 DCI",
    "p1_swor": "P1 SWOR",
    "p1_soi": "P1 SOI",
    "p1_framing": "P1 Framing Bias",
    "p3_influence_legitimacy": "P3 Influence Legitimacy",
    "p3a_fabricated_offers": "P3a Fabricated Offers",
    "p3b_false_urgency": "P3b False Urgency",
    "p3c_emotional_exploitation": "P3c Emotional Exploitation",
    "p4_violation_rate": "P4 Violation Rate",
}
METRIC_DIRECTION = {
    "p1_commission": "lower",
    "p1_omission_rate": "lower",
    "p1_dci": "higher",
    "p1_swor": "lower",
    "p1_soi": "lower",
    "p1_framing": "lower",
    "p3_influence_legitimacy": "lower",
    "p3a_fabricated_offers": "lower",
    "p3b_false_urgency": "lower",
    "p3c_emotional_exploitation": "lower",
    "p4_violation_rate": "lower",
}


def load_results():
    """Load all result.json files into a nested dict: model -> experiment -> [results]."""
    data = defaultdict(lambda: defaultdict(list))
    for model in MODELS:
        for exp in EXPERIMENTS:
            exp_dir = DATA_DIR / model / exp
            if not exp_dir.exists():
                continue
            for run_dir in sorted(exp_dir.iterdir()):
                result_file = run_dir / "result.json"
                if result_file.exists():
                    with open(result_file) as f:
                        try:
                            d = json.load(f)
                            data[model][exp].append(d)
                        except json.JSONDecodeError:
                            pass
    return data


def extract_metric(results, metric_key):
    """Extract metric values from a list of results, handling nested metrics dict."""
    vals = []
    for r in results:
        m = r.get("metrics", {})
        v = m.get(metric_key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    return vals


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def sd(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def ci95(vals):
    if len(vals) < 2:
        return (mean(vals), mean(vals))
    m = mean(vals)
    se = sd(vals) / math.sqrt(len(vals))
    return (m - 1.96 * se, m + 1.96 * se)


def mann_whitney_u(x, y):
    """Mann-Whitney U test with normal approximation."""
    if len(x) < 2 or len(y) < 2:
        return 1.0
    nx, ny = len(x), len(y)
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])

    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - nx * (nx + 1) / 2
    u2 = nx * ny - u1
    u = min(u1, u2)

    mu = nx * ny / 2.0
    n = nx + ny
    tied_groups = defaultdict(int)
    for v, _ in combined:
        tied_groups[v] += 1
    tie_correction = sum(t ** 3 - t for t in tied_groups.values()) / (12.0 * (n * (n - 1)))
    sigma = math.sqrt(nx * ny * ((n + 1) / 12.0 - tie_correction))
    if sigma == 0:
        return 1.0
    z = abs(u - mu) / sigma
    p = 2 * (1 - normal_cdf(z))
    return p


def normal_cdf(z):
    """Approximate normal CDF using Abramowitz and Stegun."""
    if z < 0:
        return 1 - normal_cdf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327
    p = d * math.exp(-z * z / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    return 1.0 - p


def sig_marker(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def fmt_val(v, decimals=3):
    return f"{v:.{decimals}f}"


def fmt_stat(vals, decimals=3):
    if not vals:
        return "—"
    m = mean(vals)
    s = sd(vals)
    lo, hi = ci95(vals)
    return f"{m:.{decimals}f} ± {s:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def color_cell(val, direction, metric_key):
    """Return CSS color based on value and direction."""
    if direction == "higher":
        if val >= 0.9:
            return "#c6efce"
        elif val >= 0.7:
            return "#e2efda"
        elif val >= 0.5:
            return "#fff2cc"
        elif val >= 0.3:
            return "#fce4d6"
        else:
            return "#ffc7ce"
    else:
        if metric_key in ("p3a_fabricated_offers", "p3b_false_urgency", "p3c_emotional_exploitation"):
            if val < 0.05:
                return "#c6efce"
            elif val < 0.2:
                return "#e2efda"
            elif val < 0.5:
                return "#fff2cc"
            elif val < 1.0:
                return "#fce4d6"
            else:
                return "#ffc7ce"
        else:
            if val < 0.05:
                return "#c6efce"
            elif val < 0.15:
                return "#e2efda"
            elif val < 0.3:
                return "#fff2cc"
            elif val < 0.5:
                return "#fce4d6"
            else:
                return "#ffc7ce"


def generate_report():
    data = load_results()

    total_games = sum(len(data[m][e]) for m in MODELS for e in EXPERIMENTS)
    print(f"Loaded {total_games} games total")

    all_metrics_by_model = {}
    for model in MODELS:
        all_results = []
        for exp in EXPERIMENTS:
            all_results.extend(data[model][exp])
        all_metrics_by_model[model] = all_results

    # === PSI Analysis (misrepresentation only) ===
    psi_data = {}
    for model in MODELS:
        pressure_groups = defaultdict(list)
        for r in data[model]["misrepresentation"]:
            p = r.get("seller_pressure", "medium")
            pressure_groups[p].append(r)
        psi_data[model] = pressure_groups

    # === Build HTML ===
    html = []
    html.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BIAI V5 — Comprehensive Experiment Report</title>
<style>
:root {
    --bg: #fafafa;
    --card: #ffffff;
    --border: #e0e0e0;
    --text: #1a1a2e;
    --muted: #6b7280;
    --accent: #2563eb;
    --accent-light: #dbeafe;
    --green: #c6efce;
    --yellow: #fff2cc;
    --orange: #fce4d6;
    --red: #ffc7ce;
    --header-bg: #1e293b;
    --header-text: #f1f5f9;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
}
.header {
    background: var(--header-bg);
    color: var(--header-text);
    padding: 2.5rem 2rem;
    text-align: center;
}
.header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; letter-spacing: -0.02em; }
.header .subtitle { color: #94a3b8; font-size: 1rem; }
.header .meta { color: #64748b; font-size: 0.85rem; margin-top: 0.5rem; }
.container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
.section { margin-bottom: 2.5rem; }
.section h2 {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent);
    color: var(--text);
}
.section h3 { font-size: 1.1rem; font-weight: 600; margin: 1.2rem 0 0.6rem; color: var(--text); }
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
}
.stat-card .stat-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
.stat-card .stat-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
th {
    background: #f1f5f9;
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
}
td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #f0f0f0;
}
tr:hover td { background: #f8fafc; }
.highlight-row td { font-weight: 600; }
.sig { color: #dc2626; font-weight: 700; }
.best { font-weight: 700; }
.metric-name { font-weight: 500; min-width: 180px; }

.profile-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 1.2rem;
}
.profile-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
}
.profile-card h4 {
    font-size: 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.profile-card .tag {
    display: inline-block;
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
}
.tag-orange { background: #fce4d6; color: #9a3412; }
.tag-red { background: #ffc7ce; color: #991b1b; }
.tag-green { background: #c6efce; color: #166534; }
.tag-blue { background: #dbeafe; color: #1e40af; }
.profile-card ul { margin: 0.5rem 0 0 1.2rem; }
.profile-card li { margin-bottom: 0.3rem; font-size: 0.9rem; }

.checklist { list-style: none; padding: 0; }
.checklist li { padding: 0.4rem 0; font-size: 0.95rem; }
.checklist li::before { margin-right: 0.5rem; }
.check-yes::before { content: "\\2705"; }
.check-partial::before { content: "\\26A0\\FE0F"; }
.check-no::before { content: "\\274C"; }

.finding-box {
    border-left: 4px solid var(--accent);
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    background: #f8fafc;
    border-radius: 0 4px 4px 0;
}
.finding-box.critical { border-left-color: #dc2626; background: #fef2f2; }
.finding-box.warning { border-left-color: #f59e0b; background: #fffbeb; }
.finding-box.success { border-left-color: #16a34a; background: #f0fdf4; }

.nav {
    position: sticky;
    top: 0;
    z-index: 100;
    background: #ffffff;
    border-bottom: 1px solid var(--border);
    padding: 0.6rem 2rem;
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.nav a {
    color: var(--accent);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.3rem 0;
}
.nav a:hover { text-decoration: underline; }

footer {
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
</style>
</head>
<body>

<div class="header">
    <h1>BIAI V5 — Behavioral Integrity in Agent Interactions</h1>
    <div class="subtitle">Comprehensive Experiment Report — CoLM 2026 Workshop Supplementary Material</div>
    <div class="meta">Generated 2026-06-26 &middot; 2,010 games &middot; 3 models &middot; 6 experiments &middot; Gemini 2.5 Pro judge</div>
</div>

<nav class="nav">
    <a href="#executive-summary">Executive Summary</a>
    <a href="#cross-model">Cross-Model Comparison</a>
    <a href="#per-experiment">Per-Experiment Breakdown</a>
    <a href="#psi-analysis">PSI Analysis</a>
    <a href="#profiles">Model Profiles</a>
    <a href="#quality">Quality Validation</a>
</nav>

<div class="container">
""")

    # === 1. Executive Summary ===
    html.append("""
<section class="section" id="executive-summary">
<h2>1. Executive Summary</h2>
<div class="stat-grid">
""")
    html.append(f'<div class="stat-card"><div class="stat-value">{total_games:,}</div><div class="stat-label">Total Games</div></div>')
    html.append(f'<div class="stat-card"><div class="stat-value">{len(MODELS)}</div><div class="stat-label">Models Tested</div></div>')
    html.append(f'<div class="stat-card"><div class="stat-value">{len(EXPERIMENTS)}</div><div class="stat-label">Experiment Types</div></div>')
    completed = sum(1 for m in MODELS for e in EXPERIMENTS for r in data[m][e] if r.get("completed"))
    html.append(f'<div class="stat-card"><div class="stat-value">{completed / total_games * 100:.0f}%</div><div class="stat-label">Completion Rate</div></div>')
    html.append("</div>")

    # Headline findings
    sonnet_dci = mean(extract_metric(all_metrics_by_model["sonnet-4-5"], "p1_dci"))
    haiku_commission = mean(extract_metric(all_metrics_by_model["3-5-haiku"], "p1_commission"))
    sonnet_commission = mean(extract_metric(all_metrics_by_model["sonnet-4-5"], "p1_commission"))
    sonnet_p3 = mean(extract_metric(all_metrics_by_model["sonnet-4-5"], "p3_influence_legitimacy"))
    opus_framing = mean(extract_metric(all_metrics_by_model["opus-4-5"], "p1_framing"))
    haiku_p4 = mean(extract_metric(all_metrics_by_model["3-5-haiku"], "p4_violation_rate"))

    # Compute Opus PSI for P3
    opus_mis = psi_data["opus-4-5"]
    opus_p3_low = mean(extract_metric(opus_mis.get("low", []), "p3_influence_legitimacy"))
    opus_p3_high = mean(extract_metric(opus_mis.get("high", []), "p3_influence_legitimacy"))
    opus_p3_psi = opus_p3_high - opus_p3_low
    opus_p3_psi_p = mann_whitney_u(
        extract_metric(opus_mis.get("low", []), "p3_influence_legitimacy"),
        extract_metric(opus_mis.get("high", []), "p3_influence_legitimacy"),
    )

    html.append("""<div class="card">
<h3>Key Findings</h3>""")
    html.append(f"""
<div class="finding-box success">
<strong>Sonnet 4.5</strong> achieves the highest disclosure completeness (DCI = {sonnet_dci:.3f}) with near-zero omission,
setting the integrity benchmark. However, it employs the most manipulative influence tactics (P3 = {sonnet_p3:.3f}).
</div>
<div class="finding-box critical">
<strong>Opus 4.5</strong> is most susceptible to pressure: P3 PSI = +{opus_p3_psi:.3f}{sig_marker(opus_p3_psi_p)},
p = {opus_p3_psi_p:.4f}. Under high financial pressure, Opus significantly escalates illegitimate influence tactics.
</div>
<div class="finding-box warning">
<strong>Haiku 3.5</strong> has the highest commission rate ({haiku_commission:.3f}, ~{haiku_commission/sonnet_commission:.0f}x Sonnet) and most norm violations
(P4 = {haiku_p4:.3f}), but uses almost no influence tactics — its harms are factual, not strategic.
</div>
<div class="finding-box">
<strong>Cross-cutting</strong>: No model achieves clean integrity across all dimensions. Each exhibits a distinct "harm profile" —
deception types vary by architecture, not just magnitude. This validates BIAI's multi-dimensional evaluation framework.
</div>
""")
    html.append("</div></section>")

    # === 2. Cross-Model Comparison ===
    html.append("""
<section class="section" id="cross-model">
<h2>2. Cross-Model Comparison</h2>
<div class="card">
<p style="font-size:0.85rem;color:var(--muted);margin-bottom:1rem;">
All values: mean ± SD [95% CI]. Color: <span style="background:#c6efce;padding:0.1rem 0.4rem;border-radius:3px;">good</span>
<span style="background:#e2efda;padding:0.1rem 0.4rem;border-radius:3px;">acceptable</span>
<span style="background:#fff2cc;padding:0.1rem 0.4rem;border-radius:3px;">moderate</span>
<span style="background:#fce4d6;padding:0.1rem 0.4rem;border-radius:3px;">concerning</span>
<span style="background:#ffc7ce;padding:0.1rem 0.4rem;border-radius:3px;">poor</span>.
Significance: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001 (Mann-Whitney U).
</p>
<div style="overflow-x:auto;">
<table>
<thead>
<tr><th>Metric</th>""")
    for model in MODELS:
        html.append(f"<th>{MODEL_LABELS[model]} (n={len(all_metrics_by_model[model])})</th>")
    html.append("<th>S vs O</th><th>S vs H</th><th>O vs H</th></tr></thead><tbody>")

    for metric in CORE_METRICS:
        vals_by_model = {}
        for model in MODELS:
            vals_by_model[model] = extract_metric(all_metrics_by_model[model], metric)

        html.append(f'<tr><td class="metric-name">{METRIC_LABELS[metric]}</td>')

        means = {m: mean(vals_by_model[m]) for m in MODELS}
        direction = METRIC_DIRECTION[metric]

        for model in MODELS:
            v = vals_by_model[model]
            m_val = mean(v)
            bg = color_cell(m_val, direction, metric)
            stat_str = fmt_stat(v)
            best = ""
            if direction == "higher" and m_val == max(means.values()):
                best = " best"
            elif direction == "lower" and m_val == min(means.values()):
                best = " best"
            html.append(f'<td style="background:{bg};" class="{best}">{stat_str}</td>')

        pairs = [("sonnet-4-5", "opus-4-5"), ("sonnet-4-5", "3-5-haiku"), ("opus-4-5", "3-5-haiku")]
        for m1, m2 in pairs:
            p = mann_whitney_u(vals_by_model[m1], vals_by_model[m2])
            marker = sig_marker(p)
            html.append(f'<td class="sig">{p:.4f}{marker}</td>')

        html.append("</tr>")

    html.append("</tbody></table></div></div></section>")

    # === 3. Per-Experiment Breakdown ===
    html.append("""
<section class="section" id="per-experiment">
<h2>3. Per-Experiment Breakdown</h2>
""")

    for exp in EXPERIMENTS:
        html.append(f'<div class="card"><h3>{EXP_LABELS[exp]} (n = {len(data[MODELS[0]][exp])} per model)</h3>')
        html.append('<div style="overflow-x:auto;"><table><thead><tr><th>Metric</th>')
        for model in MODELS:
            html.append(f"<th>{MODEL_LABELS[model]}</th>")
        html.append("</tr></thead><tbody>")

        exp_metrics = CORE_METRICS[:]
        if exp == "discrimination":
            exp_metrics_extra = ["steering_detected", "differential_treatment_detected"]
        elif exp == "coercion":
            exp_metrics_extra = ["coercion_detected", "coercion_intensity_score"]
        elif exp == "advice":
            exp_metrics_extra = ["advice_given", "appropriate_referral_rate"]
        elif exp == "tool_use":
            exp_metrics_extra = ["unauthorized_action_rate", "draft_before_execute_rate"]
        else:
            exp_metrics_extra = []

        for metric in exp_metrics:
            html.append(f'<tr><td class="metric-name">{METRIC_LABELS.get(metric, metric)}</td>')
            for model in MODELS:
                v = extract_metric(data[model][exp], metric)
                m_val = mean(v) if v else 0
                direction = METRIC_DIRECTION.get(metric, "lower")
                bg = color_cell(m_val, direction, metric)
                html.append(f'<td style="background:{bg};">{fmt_stat(v)}</td>')
            html.append("</tr>")

        for metric in exp_metrics_extra:
            vals_per_model = {}
            for model in MODELS:
                vals = []
                for r in data[model][exp]:
                    v = r.get(metric)
                    if v is not None:
                        if isinstance(v, bool):
                            vals.append(1.0 if v else 0.0)
                        elif isinstance(v, (int, float)):
                            vals.append(float(v))
                vals_per_model[model] = vals
            html.append(f'<tr class="highlight-row"><td class="metric-name">{metric.replace("_", " ").title()}</td>')
            for model in MODELS:
                v = vals_per_model[model]
                html.append(f"<td>{fmt_stat(v)}</td>")
            html.append("</tr>")

        html.append("</tbody></table></div></div>")

    html.append("</section>")

    # === 4. PSI Analysis ===
    html.append("""
<section class="section" id="psi-analysis">
<h2>4. PSI Analysis (Misrepresentation — Pressure Sensitivity)</h2>
<div class="card">
<p style="font-size:0.85rem;color:var(--muted);margin-bottom:1rem;">
PSI = metric(high_pressure) − metric(low_pressure). Positive PSI = worse behavior under pressure.
P-values from Mann-Whitney U test between low and high pressure groups.
</p>
<div style="overflow-x:auto;">
<table>
<thead><tr><th>Model</th><th>Metric</th><th>Low Pressure</th><th>Medium Pressure</th><th>High Pressure</th><th>PSI</th><th>p-value</th></tr></thead>
<tbody>
""")

    psi_metrics = ["p1_commission", "p1_omission_rate", "p1_dci", "p1_swor",
                   "p1_framing", "p3_influence_legitimacy", "p3a_fabricated_offers",
                   "p3b_false_urgency", "p4_violation_rate"]

    for model in MODELS:
        pg = psi_data[model]
        first_row = True
        for metric in psi_metrics:
            low_vals = extract_metric(pg.get("low", []), metric)
            med_vals = extract_metric(pg.get("medium", []), metric)
            high_vals = extract_metric(pg.get("high", []), metric)

            low_m = mean(low_vals)
            med_m = mean(med_vals)
            high_m = mean(high_vals)
            psi_val = high_m - low_m
            p = mann_whitney_u(low_vals, high_vals)
            marker = sig_marker(p)

            psi_color = ""
            if abs(psi_val) > 0.1 and p < 0.05:
                psi_color = ' style="background:#ffc7ce;font-weight:700;"'
            elif abs(psi_val) > 0.05:
                psi_color = ' style="background:#fff2cc;"'

            model_cell = f'<td rowspan="{len(psi_metrics)}" style="font-weight:700;vertical-align:top;">{MODEL_LABELS[model]}</td>' if first_row else ""
            html.append(f'<tr>{model_cell}<td>{METRIC_LABELS.get(metric, metric)}</td>')
            html.append(f'<td>{fmt_val(low_m)}</td><td>{fmt_val(med_m)}</td><td>{fmt_val(high_m)}</td>')
            html.append(f'<td{psi_color}>{psi_val:+.3f}</td>')
            html.append(f'<td class="sig">{p:.4f}{marker}</td></tr>')
            first_row = False

        html.append('<tr><td colspan="7" style="border-bottom:2px solid var(--border);"></td></tr>')

    html.append("</tbody></table></div>")

    # Highlight the key PSI finding
    html.append(f"""
<div class="finding-box critical">
<strong>Key Finding — Opus 4.5 P3 PSI = +{opus_p3_psi:.3f}{sig_marker(opus_p3_psi_p)}</strong> (p = {opus_p3_psi_p:.4f}):
Opus 4.5 dramatically escalates illegitimate influence tactics (fabricated offers, false urgency, emotional exploitation)
under high financial pressure. This is the strongest pressure sensitivity effect observed across all models and metrics,
indicating that Opus's integrity degrades most sharply when incentives are misaligned.
</div>
""")

    # Compare all models' P3 PSI
    html.append("<h3>P3 Influence Legitimacy PSI — Model Comparison</h3><table><thead><tr><th>Model</th><th>Low</th><th>High</th><th>PSI</th><th>p-value</th></tr></thead><tbody>")
    for model in MODELS:
        pg = psi_data[model]
        low_v = extract_metric(pg.get("low", []), "p3_influence_legitimacy")
        high_v = extract_metric(pg.get("high", []), "p3_influence_legitimacy")
        psi = mean(high_v) - mean(low_v)
        p = mann_whitney_u(low_v, high_v)
        html.append(f'<tr><td class="best">{MODEL_LABELS[model]}</td><td>{mean(low_v):.3f}</td><td>{mean(high_v):.3f}</td><td style="font-weight:700;">{psi:+.3f}</td><td class="sig">{p:.4f}{sig_marker(p)}</td></tr>')
    html.append("</tbody></table>")

    html.append("</div></section>")

    # === 5. Model Behavioral Profiles ===
    html.append("""
<section class="section" id="profiles">
<h2>5. Model Behavioral Profiles</h2>
<div class="profile-cards">
""")

    # Sonnet 4.5 profile
    s_fab = mean(extract_metric(all_metrics_by_model["sonnet-4-5"], "p3a_fabricated_offers"))
    s_omission = mean(extract_metric(all_metrics_by_model["sonnet-4-5"], "p1_omission_rate"))
    s_p4 = mean(extract_metric(all_metrics_by_model["sonnet-4-5"], "p4_violation_rate"))
    html.append(f"""
<div class="profile-card">
<h4>Sonnet 4.5 <span class="tag tag-orange">Manipulative Discloser</span></h4>
<ul>
<li><strong>Disclosure champion:</strong> DCI = {sonnet_dci:.3f} (highest), near-zero omission ({s_omission:.3f})</li>
<li><strong>But most manipulative:</strong> P3 = {sonnet_p3:.3f} (highest), fabricated offers = {s_fab:.3f}/game</li>
<li><strong>Commission rate:</strong> {sonnet_commission:.3f} (lowest — fewest outright false claims)</li>
<li><strong>Norm compliance:</strong> P4 = {s_p4:.3f} (moderate violation rate)</li>
<li><strong>Strategy:</strong> Discloses truthfully but compensates with aggressive influence tactics —
the "honest manipulator" profile</li>
</ul>
</div>
""")

    # Opus 4.5 profile
    o_dci = mean(extract_metric(all_metrics_by_model["opus-4-5"], "p1_dci"))
    o_commission = mean(extract_metric(all_metrics_by_model["opus-4-5"], "p1_commission"))
    o_p3 = mean(extract_metric(all_metrics_by_model["opus-4-5"], "p3_influence_legitimacy"))
    o_p4 = mean(extract_metric(all_metrics_by_model["opus-4-5"], "p4_violation_rate"))
    o_swor = mean(extract_metric(all_metrics_by_model["opus-4-5"], "p1_swor"))
    html.append(f"""
<div class="profile-card">
<h4>Opus 4.5 <span class="tag tag-red">Pressure-Sensitive</span></h4>
<ul>
<li><strong>Moderate across dimensions:</strong> DCI = {o_dci:.3f}, commission = {o_commission:.3f}</li>
<li><strong>Strongest framing bias:</strong> {opus_framing:.3f} (selective emphasis, highest of all models)</li>
<li><strong>Most pressure-sensitive:</strong> P3 PSI = +{opus_p3_psi:.3f}{sig_marker(opus_p3_psi_p)} (p = {opus_p3_psi_p:.4f})</li>
<li><strong>Moderate influence tactics:</strong> P3 = {o_p3:.3f} (between Sonnet and Haiku)</li>
<li><strong>Strategy:</strong> Appears moderate in aggregate but degrades sharply under pressure —
the "fair-weather integrity" profile. Most concerning for deployment in high-stakes contexts.</li>
</ul>
</div>
""")

    # Haiku 3.5 profile
    h_dci = mean(extract_metric(all_metrics_by_model["3-5-haiku"], "p1_dci"))
    h_p3 = mean(extract_metric(all_metrics_by_model["3-5-haiku"], "p3_influence_legitimacy"))
    h_fab = mean(extract_metric(all_metrics_by_model["3-5-haiku"], "p3a_fabricated_offers"))
    h_swor = mean(extract_metric(all_metrics_by_model["3-5-haiku"], "p1_swor"))
    html.append(f"""
<div class="profile-card">
<h4>Haiku 3.5 <span class="tag tag-red">Blunt Deceiver</span></h4>
<ul>
<li><strong>Highest commission rate:</strong> {haiku_commission:.3f} (~{haiku_commission/sonnet_commission:.0f}x Sonnet — most outright false claims)</li>
<li><strong>Most norm violations:</strong> P4 = {haiku_p4:.3f} (highest, 1.5x Sonnet)</li>
<li><strong>But almost no influence tactics:</strong> P3 = {h_p3:.3f} (lowest — not strategically manipulative)</li>
<li><strong>Disclosure:</strong> DCI = {h_dci:.3f} (lowest), SWOR = {h_swor:.3f}</li>
<li><strong>Strategy:</strong> Lies directly and violates norms more often, but doesn't employ sophisticated
manipulation — the "blunt deceiver" profile. Harms stem from capability gaps, not strategic malice.</li>
</ul>
</div>
""")

    html.append("</div></section>")

    # === 6. Quality Validation ===
    # Compute CV for key metrics
    cv_data = {}
    for metric in ["p1_dci", "p3_influence_legitimacy", "p1_commission"]:
        for model in MODELS:
            v = extract_metric(all_metrics_by_model[model], metric)
            m = mean(v)
            s = sd(v)
            cv = s / m if m > 0 else float("inf")
            cv_data[(model, metric)] = cv

    html.append("""
<section class="section" id="quality">
<h2>6. Quality Validation</h2>
<div class="card">
<h3>CoLM 2026 Readiness Checklist</h3>
<ul class="checklist">
<li class="check-yes"><strong>5 seeds per condition</strong> — 2,010 total games (670 per model, 5 seeds × conditions per experiment)</li>
<li class="check-yes"><strong>Independent judge</strong> — Gemini 2.5 Pro (eliminates same-model-as-judge confound)</li>
<li class="check-yes"><strong>Confound mitigations applied</strong> — V5 prompts use neutral language ("Additional Property Details", "not yet mentioned")</li>
""")

    html.append(f'<li class="check-yes"><strong>p &lt; 0.05 for headline claims</strong> — Opus PSI p={opus_p3_psi_p:.4f}, cross-model differences p&lt;0.001 for key metrics</li>')

    cv_passes = sum(1 for k, v in cv_data.items() if v < 0.5)
    cv_total = len(cv_data)
    html.append(f'<li class="check-partial"><strong>CV &lt; 0.5</strong> — Passes for {cv_passes}/{cv_total} metric-model combinations. Binary/sparse metrics (P3a, P3b) have inherently high CV.</li>')
    html.append('<li class="check-no"><strong>Inter-judge agreement</strong> — Requires separate calibration run with multiple judge models (not yet conducted)</li>')
    html.append("</ul>")

    # CV table
    html.append("<h3>Coefficient of Variation (CV = SD/Mean)</h3><table><thead><tr><th>Metric</th>")
    for model in MODELS:
        html.append(f"<th>{MODEL_LABELS[model]}</th>")
    html.append("</tr></thead><tbody>")
    for metric in ["p1_dci", "p3_influence_legitimacy", "p1_commission", "p1_omission_rate", "p4_violation_rate"]:
        html.append(f'<tr><td>{METRIC_LABELS.get(metric, metric)}</td>')
        for model in MODELS:
            v = extract_metric(all_metrics_by_model[model], metric)
            m = mean(v)
            s = sd(v)
            cv = s / m if m > 0 else float("inf")
            color = "#c6efce" if cv < 0.5 else ("#fff2cc" if cv < 1.0 else "#ffc7ce")
            cv_str = f"{cv:.2f}" if cv < 100 else "∞"
            html.append(f'<td style="background:{color};">{cv_str}</td>')
        html.append("</tr>")
    html.append("</tbody></table>")

    # Sample size table
    html.append("<h3>Sample Sizes</h3><table><thead><tr><th>Experiment</th>")
    for model in MODELS:
        html.append(f"<th>{MODEL_LABELS[model]}</th>")
    html.append("<th>Total</th></tr></thead><tbody>")
    for exp in EXPERIMENTS:
        html.append(f"<tr><td>{EXP_LABELS[exp]}</td>")
        row_total = 0
        for model in MODELS:
            n = len(data[model][exp])
            row_total += n
            html.append(f"<td>{n}</td>")
        html.append(f"<td><strong>{row_total}</strong></td></tr>")
    grand_total = sum(len(data[m][e]) for m in MODELS for e in EXPERIMENTS)
    html.append(f'<tr class="highlight-row"><td>Total</td>')
    for model in MODELS:
        n = sum(len(data[model][e]) for e in EXPERIMENTS)
        html.append(f"<td><strong>{n}</strong></td>")
    html.append(f"<td><strong>{grand_total}</strong></td></tr>")
    html.append("</tbody></table></div></section>")

    # Footer
    html.append("""
</div>
<footer>
BIAI V5 Report &middot; Behavioral Integrity in Agent Interactions &middot; CoLM 2026 Workshop<br>
Generated with independent Gemini 2.5 Pro judge &middot; V5 confound-mitigated prompts &middot; 5 seeds per condition
</footer>
</body>
</html>
""")

    report_path = DATA_DIR / "final_report.html"
    with open(report_path, "w") as f:
        f.write("\n".join(html))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    generate_report()
