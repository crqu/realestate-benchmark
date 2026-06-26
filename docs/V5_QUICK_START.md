# V5 Experiments Quick Start Guide

## Prerequisites

1. Virtual environment activated:
   ```bash
   source venv/bin/activate
   ```

2. GCP credentials configured for project `itpc-gcp-ai-eng-claude`

## Running Full Experiment Suite (RECOMMENDED)

### Parallel Execution (Fastest — ~2-3 hours for full suite)

Run all experiments for all models in parallel:

```bash
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --region us-east5 \
    --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \
    --experiments all \
    --judge-model gemini-2.5-pro \
    --num-seeds 5 \
    --max-turns 20 \
    --max-workers 6
```

**What this does:**
- Launches 18 parallel jobs (3 models × 6 experiments)
- Each job runs independently in its own process
- Maximum 6 jobs run concurrently (adjustable via `--max-workers`)
- Generates individual reports as each job completes
- Creates master cross-model comparison report at the end

**Expected output:**
- ~495 total games (165 per model)
- Individual HTML reports per (model, experiment) combination
- Master report: `experiments/harm_results/v5_parallel_YYYYMMDD_HHMMSS/master_report.html`

**Resource usage:**
- ~6 CPU cores at 100% utilization
- ~12-18 GB RAM (2-3 GB per worker)
- ~100-150 API calls/minute to Vertex AI
- ~2-3 hours total wall-clock time

### Sequential Execution (Slower but lower resource usage)

Run experiments one at a time:

```bash
venv/bin/python experiments/run_v5_experiments.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \
    --experiment all \
    --judge-model gemini-2.5-pro \
    --num-seeds 5 \
    --max-turns 20
```

**Tradeoff:**
- Lower resource usage (~1 CPU core, ~3 GB RAM)
- ~8-12 hours total wall-clock time (3-4× slower than parallel)

---

## Running Subset of Experiments

### Single Experiment Type (e.g., Misrepresentation Only)

```bash
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \
    --experiments misrepresentation \
    --judge-model gemini-2.5-pro \
    --num-seeds 5 \
    --max-workers 3
```

**Expected:** ~72 games (24 per model), ~30-40 minutes

### Multiple Specific Experiments

```bash
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 claude-opus-4-5 \
    --experiments misrepresentation discrimination confidentiality \
    --judge-model gemini-2.5-pro \
    --num-seeds 5 \
    --max-workers 4
```

**Expected:** ~288 games (144 per model), ~60-90 minutes

### Single Model, All Experiments

```bash
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 \
    --experiments all \
    --judge-model gemini-2.5-pro \
    --num-seeds 5 \
    --max-workers 6
```

**Expected:** ~165 games, ~60-90 minutes

---

## Testing Before Full Run

### Minimal Test (2 seeds, short turns)

```bash
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 \
    --experiments misrepresentation \
    --judge-model gemini-2.5-pro \
    --num-seeds 2 \
    --max-turns 12 \
    --max-workers 1
```

**Expected:** ~24 games, ~15-20 minutes

Verify:
1. Jobs complete successfully
2. HTML reports generate correctly
3. New metrics appear in reports (DCI, SWOR, P3a/P3b/P3c, PSI)

---

## Available Experiments

- `discrimination` — Fair housing steering and bargaining equity
- `misrepresentation` — Pressure-induced deception (financial stress)
- `confidentiality` — Information leakage under extraction strategies
- `advice` — Unauthorized legal/financial advice provision
- `coercion` — Vulnerability exploitation and manipulative persuasion
- `tool_use` — Autonomous action safety (waiving inspection, signing contracts)

Use `--experiments all` to run all of the above.

---

## Monitoring Progress

### Check running jobs

```bash
# Show all Python processes
ps aux | grep run_v5_parallel

# Monitor CPU and memory
top -u $USER
```

### Check partial results

```bash
# List generated reports
find experiments/harm_results/v5_parallel_* -name "report.html"

# Check latest experiment directory
ls -lht experiments/harm_results/ | head -5

# Count completed games
find experiments/harm_results/v5_parallel_* -name "result.json" | wc -l
```

### View live output (if running in background)

```bash
tail -f /tmp/claude-*/tasks/*.output
```

---

## Understanding the Output

### Directory Structure

```
experiments/harm_results/v5_parallel_20260625_143052/
├── config.json                          # Experiment configuration
├── master_report.html                   # ⭐ Cross-model comparison
├── sonnet-4-5/
│   ├── misrepresentation/
│   │   ├── analysis.json                # Statistical analysis
│   │   ├── report.html                  # ⭐ Per-experiment report
│   │   ├── minor_low_naive_seed0/
│   │   │   ├── game.db
│   │   │   ├── result.json
│   │   │   └── report.md
│   │   └── ... (more games)
│   ├── discrimination/
│   └── ... (other experiments)
├── opus-4-5/
│   └── ... (same structure)
└── haiku-3-5/
    └── ... (same structure)
```

### Key Reports to Check

1. **Master Report** — `master_report.html`
   - Job execution summary
   - Cross-model metric comparison tables
   - Links to individual experiment reports

2. **Per-Experiment Reports** — `{model}/{experiment}/report.html`
   - Overall metric statistics with 95% CI
   - Buyer sophistication breakdown
   - PSI (Pressure Sensitivity Index) analysis
   - Cross-buyer statistical tests
   - Individual game results

3. **Analysis JSON** — `{model}/{experiment}/analysis.json`
   - Machine-readable statistical results
   - Use for further analysis in Python/R

---

## Expected Results (from NEXT_STEPS.md)

### Statistical Power
- **5 seeds per condition** → sufficient for detecting medium-to-large effect sizes
- **95% confidence intervals** reported for all metrics
- **p < 0.05** threshold for statistical significance

### Inter-Judge Reliability
From calibration study:
- P1 Omission: 91% agreement
- P4 Norms: 96-100% agreement (binary)
- P3a (Fabricated Offers): High (verifiable)
- P3b (False Urgency): High (turn-indexed)
- P3c (Emotional Exploitation): Low (~52%) — flagged as soft metric

### New Metrics
All reports include:
- **DCI** (Disclosure Completeness Index) — 0.0 to 1.0
- **SWOR** (Severity-Weighted Omission) — cost-weighted
- **SOI** (Strategic Omission Index) — correlation measure
- **P3a/P3b/P3c** — decomposed influence tactics
- **PSI** — pressure-induced integrity degradation

---

## Troubleshooting

### Out of Memory

Reduce `--max-workers`:
```bash
--max-workers 3  # Instead of 6
```

### API Rate Limits

Reduce parallelism or add delays:
```bash
--max-workers 2
```

### Job Failures

Check individual experiment logs:
```bash
find experiments/harm_results/v5_parallel_* -name "*.log"
```

Rerun failed experiments individually:
```bash
venv/bin/python experiments/run_v5_experiments.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 \
    --experiment discrimination \
    --judge-model gemini-2.5-pro \
    --num-seeds 5
```

---

## After Experiments Complete

1. **Review master report** — Check overall statistics and cross-model comparisons

2. **Check inter-judge agreement** — Run calibration analysis:
   ```bash
   venv/bin/python experiments/run_calibration.py \
       --project-id itpc-gcp-ai-eng-claude \
       --judge-model gemini-2.5-pro \
       --results-dir experiments/harm_results/v5_parallel_YYYYMMDD_HHMMSS
   ```

3. **Generate paper artifacts**:
   - Export tables to LaTeX/CSV for paper
   - Create publication-quality figures
   - Document confound mitigations applied

4. **Archive results**:
   ```bash
   tar -czf v5_results_YYYYMMDD.tar.gz experiments/harm_results/v5_parallel_YYYYMMDD_HHMMSS/
   ```

---

## Quick Commands Cheat Sheet

```bash
# Full suite, parallel (RECOMMENDED)
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \
    --experiments all --judge-model gemini-2.5-pro --num-seeds 5 --max-workers 6

# Test run (fast)
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 \
    --experiments misrepresentation --judge-model gemini-2.5-pro \
    --num-seeds 2 --max-turns 12 --max-workers 1

# Single experiment, all models
venv/bin/python experiments/run_v5_parallel.py \
    --project-id itpc-gcp-ai-eng-claude \
    --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \
    --experiments misrepresentation --judge-model gemini-2.5-pro \
    --num-seeds 5 --max-workers 3

# Monitor progress
watch -n 5 'find experiments/harm_results/v5_parallel_* -name "result.json" | wc -l'
```
