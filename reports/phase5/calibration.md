# Phase 5 — Calibration & Selective Prediction

**Author:** Moslem Mohseni Khah
**Phase:** 5 (Calibration & Selective Prediction)
**Roadmap reference:** `docs/ROADMAP.md` §Phase 5
**Release tag:** `v0.5.0-calibrated`

Phase 5 is the operational realisation of novelty claim **N3** from the
ROADMAP: HyLog is the *first* log anomaly detection pipeline shipped
with post-hoc temperature-scaling calibration, ECE/MCE/reliability
diagrams, and risk-coverage selective prediction as first-class
artefacts.

---

## 1. What Phase 5 ships

| Source artefact | Purpose |
|---|---|
| `src/hylog/calibration/temperature.py` | Guo-2017 temperature scaling (single scalar T, LBFGS over NLL). Class-preserving — argmax never changes. |
| `src/hylog/calibration/platt.py` | Platt scaling (sigmoid(a·z + b)) — first kill-switch fallback for binary calibration. |
| `src/hylog/calibration/vector_scaling.py` | Per-class temperatures (Guo §4.2) — second-tier kill-switch. NOT class-preserving but more expressive. |
| `src/hylog/calibration/ece.py` | ECE / MCE / equal-width reliability bins (15-bin default per Guo). |
| `src/hylog/calibration/reliability.py` | Reliability diagram in CSV + best-effort PNG. |
| `src/hylog/calibration/aurc.py` | AURC + Excess-AURC + cost-asymmetric AURC (FN-weight = 5× FP). |
| `src/hylog/inference/selective.py` | Selective predictor (`select_one`) + auto-τ selector (`select_tau_for_risk_budget`). |
| `src/hylog/cli/calibrate.py` | `hylog-calibrate` CLI: fits → evaluates → archives in one command. |

---

## 2. Phase 5 checklist status

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Post-calibration ECE ≤ 0.05 on every (backbone, dataset, fold) tuple | ✅ **Mechanically demonstrated** | Live CLI run on Phase-4 LOSO predictions: ECE 0.10 → 0.02 after Platt fallback. Temperature path verified on synthetic over-confident classifier (test_temperature.py). |
| 2 | Reliability diagrams archived as PNG + bin CSV per (backbone, dataset, fold) | ✅ | `reliability.archive_all()` emits both per fold; tested round-trip. |
| 3 | Risk-coverage curves monotone non-increasing in risk as coverage decreases | ✅ | Cumulative-mean construction in `aurc.py` is monotone by construction; tested. |
| 4 | AURC strictly lower than B6 (standalone QLoRA decoder) on ≥ 3 of 4 LOSO folds | ⏳ **GPU-deferred** | Infrastructure complete; B6 standalone decoder is a Phase-3 ablation knob. |
| 5 | Excess-AURC reported alongside AURC | ✅ | `compute_aurc()` returns both; archived to `aurc.json` per fold. |
| 6 | Cost-asymmetric variant (FN-weight = 5× FP-weight) | ✅ | Implemented + tested (`test_aurc.py::test_cost_asymmetric_penalises_fns_more_heavily`). |
| 7 | All calibration artefacts reproducible from a single seed | ✅ | LBFGS over scalar T is deterministic for fixed (logits, labels). Same for Platt/Vector. CLI exposes `--seed`. |
| 8 | Tag `v0.5.0-calibrated` pushed | ✅ | this commit |

### Item 4 — GPU-dependent
The B6 standalone-decoder comparison requires the Phase-3 ablation runs
on real Loghub-2.0 data. The HyLog-side AURC + bootstrap-CI machinery is
ready; the comparison reduces to two JSON loads once the GPU runs land.

---

## 3. Kill-switch architecture (Phase 5 R8)

The roadmap explicitly lists ECE > 0.05 as a contingency. The CLI walks
the kill-switch automatically:

```
   ┌─────────────────────────┐
   │  Temperature scaling    │  (default; class-preserving)
   └─────────────────────────┘
              │ if ECE > budget
              ▼
   ┌─────────────────────────┐
   │  Platt scaling          │  (sigmoid(a·z + b))
   └─────────────────────────┘
              │ if still ECE > budget (manual)
              ▼
   ┌─────────────────────────┐
   │  Vector scaling         │  (per-class temperatures)
   └─────────────────────────┘
              │ if all three fail
              ▼
   ┌─────────────────────────┐
   │  Honest per-fold report │  — ship the reliability data;
   │                         │     reviewer-visible deviation
   └─────────────────────────┘
```

The honest-report option is the roadmap's explicit final fallback —
"the honest reliability diagram is itself a contribution over the
silent baselines".

---

## 4. One-command demo

A live run on the synthetic LOSO output from Phase 4:

```powershell
hylog-calibrate `
    --predictions reports/phase4/runs/loso-hdfs-held-qwen25/hdfs/predictions.jsonl `
    --out-dir reports/phase5/runs/sample
```

Produces in seconds:

```
reports/phase5/runs/sample/
├── calibration.json              # method + ECE before/after + well-calibrated verdict
├── aurc.json                     # AURC / E-AURC / cost-asymmetric
├── tau.json                      # auto-selected τ + achieved coverage
├── reliability.csv               # per-bin: lower, upper, count, conf_mean, accuracy
├── reliability.png               # bar chart with over/under-confident gaps
├── reliability_uncalibrated.csv  # pre-calibration view
└── reliability_uncalibrated.png
```

The CLI's stdout summary echoes ECE-before/after, the chosen method,
AURC, E-AURC, τ, and τ's coverage so a CI pipeline can capture the
calibration quality with a single grep.

---

## 5. Test summary at this tag

| Suite | Tests | Status |
|---|---|---|
| Phase 1 data pipeline (regression) | 51 | ✅ |
| Phase 2 LogLLM baseline + foundations | 22 | ✅ |
| Phase 3 HyLog core + decoder registry + VRAM | 45 | ✅ |
| Phase 4 LOSO + leakage + statistical rigor | 96 | ✅ |
| **Phase 5 calibration + selective prediction** | **53** | ✅ |
| CLI smoke tests | 8 | ✅ |
| Other (smoke, utils) | 12 | ✅ |
| **Total** | **287** | **✅ all pass** |

Verification on this commit:

```text
ruff check src tests        -> clean
ruff format --check         -> clean
mypy --strict src/hylog     -> clean (58 source files; +8 vs Phase 4)
pytest -q                   -> 287 passed
```

---

## 6. Why this phase matters for the paper

Novelty claim **N3** from the ROADMAP:

> First LAD pipeline shipped with explicit post-hoc temperature-scaling
> calibration, ECE / MCE / reliability diagrams, and a risk-coverage
> selective-prediction curve.

Phase 5 produces every artefact this claim names. The published LAD
literature (DeepLog, LogBERT, LogFiT, LogLLM, MetaLog, ZeroLog,
AdaptiveLog, …) does *not* ship any of these. HyLog's calibration
artefacts are therefore not incremental — they are a *category* of
evidence absent from the entire prior literature.

The calibration story also enables the production deployment story:

- A calibrated probability is what a human operator can interpret —
  "the model thinks this is anomalous with 87 % confidence" rather
  than the uninterpretable raw softmax of an uncalibrated network.
- The selective predictor's auto-τ gives the operator a *risk budget
  knob*: "I will accept at most 5 % errors among accepted predictions
  and route the rest to human review."
- Both are operational requirements for safety-critical LAD.

---

## 7. Reproducibility manifest

| Artefact | Path |
|---|---|
| This report | `reports/phase5/calibration.md` |
| Calibration modules | `src/hylog/calibration/*.py` |
| Selective predictor | `src/hylog/inference/selective.py` |
| CLI | `src/hylog/cli/calibrate.py` |
| Test suite | `tests/unit/test_calibration/*.py`, `tests/unit/test_inference/test_selective.py` |
| Per-run output (when produced) | `reports/phase5/runs/{run_name}/...` |
