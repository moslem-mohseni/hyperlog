# LOSO Protocol Specification

**Author:** Moslem Mohseni Khah
**Document type:** Methodological protocol (reviewer-facing)
**Phase:** 4 — Cross-system evaluation
**Roadmap reference:** `docs/ROADMAP.md` §Phase 4

This document is the formal statement of HyLog's leave-one-system-out
(LOSO) cross-system evaluation protocol. It is the contract between
HyLog's empirical claims and an independent reviewer.

---

## 1. Definitions

| Term | Meaning |
|---|---|
| **System** | One of the registered datasets: HDFS, BGL, Thunderbird, OpenStack. |
| **Source split** | The train + val splits emitted by `splits/<system>.json` for systems that are NOT held out in a given fold. |
| **Target split** | The test split for the system that IS held out in a given fold. |
| **Fold** | One concrete (sources, target) pairing. With four systems we run four folds. |
| **Core LOSO** | The three folds over {HDFS, BGL, Thunderbird} — size-comparable systems. |
| **Sensitivity LOSO** | The fourth fold with OpenStack held out. Reported separately because of OpenStack's much smaller size (roadmap §3.2). |
| **Zero target labels** | A protocol property: the target system's labels are NEVER consumed during training, validation, or hyperparameter selection. They are used ONLY for the evaluation metric panel. |
| **Information leakage** | Any condition where a preprocessed line or a group_id from the target test split appears in the training input. |

---

## 2. Protocol steps (per fold)

For each held-out system `T` ∈ {HDFS, BGL, Thunderbird, OpenStack}:

1. **Compose the training set** = the union of (train + val) splits of all
   systems other than `T`, drawn from `splits/<system>.json` manifests.
2. **Compose the target inference set** = the test split of `T`.
3. **Strip target labels** — if the kill-switch's self-supervised
   augmentation is enabled, the target lines are admitted into a *separate*
   unsupervised pass with all labels overwritten by `0`
   (`hylog.evaluation.cross_system.strip_labels`).
4. **Run the leakage audit** (`hylog.evaluation.leakage_audit.audit_leakage`)
   over training input vs target inference input. The audit returns a
   `LeakageReport`. The fold's run **must** be aborted on any non-zero
   intersection — `leakage_strict=True` is the default.
5. **Train** — invoke the supplied `TrainerProtocol` callable on the
   composed training set. Validation losses are computed on the same
   training set's held-out validation slice (NOT from `T`).
6. **Predict** — emit `(y_pred, y_score)` over the target inference set.
7. **Evaluate** — compute the full metric panel
   (`hylog.evaluation.metrics.compute_metric_panel`) using the target
   labels.
8. **Archive** — write `metrics.json`, `confusion.{csv,png,txt}`,
   `leakage.json`, and `predictions.jsonl` under
   `reports/phase4/runs/{run_name}/{T}/`.

After all folds:

9. **Aggregate** — compute macro mean ± std of every metric across folds
   into `reports/phase4/runs/{run_name}/summary.json`.

---

## 3. Reproducibility commitments

- **Deterministic splits** — every fold's training and target sets are
  reconstructed from the byte-identical Phase-1 manifests
  (`splits/<system>.json`). The SHA-256 of every manifest is committed
  to the repo and the orchestrator refuses to run if a manifest does
  not hash to its recorded value.
- **Seed sweep** — every fold is run with the seeds
  `[42, 1337, 2024, 31415, 27182]` so the reported macro statistics
  carry an n=5 standard error.
- **Per-prediction provenance** — the per-fold `predictions.jsonl`
  records the original `group_id`, the true label, the predicted
  label, and the calibrated anomaly probability. The granularity is one
  line per sequence so a reviewer can join HyLog's predictions back to
  the original Loghub-2.0 sequence.
- **Leakage artefact** — every fold persists its `leakage.json` with
  cardinalities, the verdict, and a deterministic sample of leaked
  lines (capped at 16). The audit is the load-bearing guarantee for
  novelty claim N2.

---

## 4. Information-leakage audit — the load-bearing safeguard

A cross-system claim is only as credible as its leakage audit. The audit
implemented at `src/hylog/evaluation/leakage_audit.py` checks two
properties on every fold:

**P1 — Line disjointness.** For every preprocessed line in the
training input, its SHA-256 fingerprint must not appear in the SHA-256
fingerprint set of the target inference input. The fingerprint is over
the preprocessed (regex-masked) line so identical lines that differ
only in volatile arguments are correctly considered the same.

**P2 — Group disjointness.** No `group_id` may appear in both
training and target inference. Phase 1 already enforces this within a
single dataset; the LOSO audit re-checks it across datasets as
defence in depth.

The audit is *active* — its result is archived and a non-clean verdict
aborts the run by default. It is also *passive* — the per-fold
`leakage.json` is a permanent artefact that any reviewer can inspect.

---

## 5. Threats to validity (acknowledged)

- **Distribution shift confound.** The metric panel rewards models
  that happen to overfit shared masking patterns. The masked-regex
  preprocessor (LogLLM-style) is the same across systems, so this
  is the cleanest baseline.
- **OpenStack size imbalance.** With ~80× fewer lines than Thunderbird,
  the OpenStack fold is biased toward higher variance. The protocol
  reports it in the Sensitivity-LOSO group separately from the
  size-comparable Core-LOSO.
- **Anomaly definition heterogeneity.** Each system labels different
  failure modes as anomalous. A model with strong overall macro-F1 may
  still struggle on one fold for reasons that are domain-specific
  rather than methodological.

---

## 6. Mapping of protocol artefacts to repository paths

| Artefact | Path |
|---|---|
| Per-fold metrics | `reports/phase4/runs/{run_name}/{T}/metrics.json` |
| Confusion matrix | `reports/phase4/runs/{run_name}/{T}/confusion.{csv,png,txt}` |
| Leakage audit | `reports/phase4/runs/{run_name}/{T}/leakage.json` |
| Per-prediction file | `reports/phase4/runs/{run_name}/{T}/predictions.jsonl` |
| Aggregated summary | `reports/phase4/runs/{run_name}/summary.json` |
| Published baselines | `reports/phase4/published_numbers.yaml` |
| Head-to-head table (rendered) | `reports/phase4/cross_system.md` |
| Protocol (this document) | `reports/phase4/loso_protocol.md` |
