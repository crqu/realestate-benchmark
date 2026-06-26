# V5 Experiment Status — June 25, 2026

## Current Status: 🔄 EXPERIMENTS RUNNING

### Test Experiment (In Progress)
- **Started:** 14:53 (June 25, 2026)
- **Command:** `run_v5_experiments.py --num-seeds 2 --max-turns 12`
- **Progress:** 1/~24 games completed
- **Status:** Running
- **Output:** `experiments/harm_results/v5_20260625_145315/`

### Full Experiment Suite (Queued)
- **Status:** Auto-launcher monitoring test, will start when test completes
- **Script:** `experiments/run_full_v5.sh`
- **Log:** `experiments/full_v5_launcher.log`
- **Command:**
  ```bash
  venv/bin/python experiments/run_v5_parallel.py \
      --project-id itpc-gcp-ai-eng-claude \
      --models claude-sonnet-4-5@20250929 claude-opus-4-5 claude-3-5-haiku@20241022 \
      --experiments all \
      --judge-model gemini-2.5-pro \
      --num-seeds 5 \
      --max-workers 6
  ```
- **Expected:** ~855 games, ~2-3 hours
- **Output:** `experiments/harm_results/v5_parallel_YYYYMMDD_HHMMSS/`

---

## Implementation Completed ✅

All NEXT_STEPS.md requirements have been implemented:

1. ✅ **System Prompt Confound Fixes**
   - Removed priming language from seller prompts

2. ✅ **Enhanced Metrics**
   - P3 decomposition (P3a/P3b/P3c)
   - DCI, SWOR, SOI metrics
   - PSI (Pressure Sensitivity Index)

3. ✅ **Independent Judge Infrastructure**
   - Gemini 2.5 Pro as cross-judge
   - Harm-specific judges support independent models

4. ✅ **Statistical Experiment Infrastructure**
   - Sequential runner: `run_v5_experiments.py`
   - **Parallel runner: `run_v5_parallel.py`** (3-4× faster)

---

## Monitoring Progress

### Check test experiment
```bash
# Watch progress
tail -f experiments/full_v5_launcher.log

# Count completed games
find experiments/harm_results/v5_20260625_145315/ -name "result.json" | wc -l

# Check test processes
ps aux | grep run_v5
```

### Check full experiments (when running)
```bash
# Monitor launcher
tail -f experiments/full_v5_launcher.log

# Count completed games (across all experiments)
find experiments/harm_results/v5_parallel_* -name "result.json" | wc -l

# Check parallel workers
ps aux | grep run_v5_parallel
```

### View results
```bash
# Latest test report
find experiments/harm_results/v5_20260625_145315/ -name "report.html"

# Latest full experiment master report
ls -t experiments/harm_results/v5_parallel_*/master_report.html | head -1

# Open in browser
firefox $(ls -t experiments/harm_results/v5_parallel_*/master_report.html | head -1)
```

---

## Expected Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Test experiment | 20-30 min | 🔄 In progress |
| Auto-launcher monitoring | Continuous | 🔄 Running |
| Full experiments (parallel) | 2-3 hours | ⏳ Queued |
| **Total** | **~3-4 hours** | **🔄 Active** |

---

## Deliverables

### When Test Completes
- ✓ Test validation report (`report.html`)
- ✓ Verification of new metrics (DCI, SWOR, P3a/P3b/P3c)
- ✓ Confidence in pipeline before full run

### When Full Experiments Complete
- 📊 Master cross-model comparison report
- 📊 18 individual experiment reports (3 models × 6 experiments)
- 📊 ~855 game results with full metrics
- 📊 Statistical analysis (CI, p-values, PSI)
- 📊 Inter-judge agreement data

---

## Next Steps (After Completion)

1. **Review master report**
   - Check overall statistics
   - Verify new metrics appear correctly
   - Identify significant findings

2. **Run calibration analysis**
   ```bash
   venv/bin/python experiments/run_calibration.py \
       --project-id itpc-gcp-ai-eng-claude \
       --judge-model gemini-2.5-pro
   ```

3. **Quality check** (per NEXT_STEPS.md Section 4)
   - [ ] p < 0.05 for headline claims
   - [ ] Inter-judge agreement > 70%
   - [ ] Coefficient of variation < 0.5
   - [ ] No confound-explainable findings

4. **Generate paper artifacts**
   - Export tables to LaTeX/CSV
   - Create figures
   - Document methodology

5. **Iterate if needed**
   - Increase seeds if variance too high
   - Refine prompts if confounds detected
   - Add experiments for follow-up questions

---

## Emergency Controls

### Stop all experiments
```bash
pkill -f run_v5
```

### Stop just the full suite
```bash
pkill -f run_v5_parallel
```

### Stop just the test
```bash
pkill -f "run_v5_experiments.*num-seeds 2"
```

### Resume from partial results
The parallel runner is fault-tolerant. If stopped mid-run:
1. Check partial results in latest `v5_parallel_*` directory
2. Re-run only failed experiments manually
3. Aggregate results across runs

---

## Files & Directories

### Code
- `experiments/run_v5_experiments.py` — Sequential runner
- `experiments/run_v5_parallel.py` — **Parallel runner (RECOMMENDED)**
- `experiments/run_full_v5.sh` — Auto-launcher script
- `src/realestate_benchmark/evaluation/llm_judge.py` — Enhanced metrics

### Documentation
- `docs/V5_IMPLEMENTATION.md` — Technical details
- `docs/V5_QUICK_START.md` — Usage guide
- `docs/V5_COMPLETION_SUMMARY.md` — Final summary
- `EXPERIMENT_STATUS.md` — **This file**

### Logs
- `experiments/full_v5_launcher.log` — Auto-launcher output
- `experiments/harm_results/v5_*/` — Experiment outputs

---

## Contact Information

- **Implementation:** V5 codebase (June 25, 2026)
- **Questions:** See documentation in `docs/`
- **Issues:** Check logs, consult troubleshooting sections
- **Updates:** Monitor `EXPERIMENT_STATUS.md` (this file)

---

**Last Updated:** June 25, 2026 15:06  
**Status:** Test running, full suite queued via auto-launcher  
**ETA for completion:** ~3-4 hours from test start
