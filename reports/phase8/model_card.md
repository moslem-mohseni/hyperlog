---
language:
  - en
license: mit
library_name: hylog
tags:
  - log-anomaly-detection
  - small-language-models
  - qlora
  - calibrated-uncertainty
  - cross-system-generalization
  - selective-prediction
datasets:
  - logpai/loghub-2.0
base_model: Qwen/Qwen2.5-1.5B
model-index:
  - name: HyLog
    results: []
---

# HyLog — Hybrid SLM Pipeline for Cross-System Log Anomaly Detection

**Author:** Moslem Mohseni Khah
**Version:** 0.8.0
**License:** MIT
**Repository:** https://github.com/moslem-mohseni/hyperlog

This model card follows the Hugging Face template. It documents the
intended use, training data, evaluation, limitations, and ethical
considerations for HyLog — a production-ready hybrid small-language-model
pipeline for log anomaly detection.

---

## 1. Model description

HyLog is a three-component pipeline:

1. **Frozen BERT-base encoder** producing per-log-line pooled
   embeddings.
2. **Learned MLP projector** (depth-2, GELU + dropout) mapping the
   768-dim BERT pooled vectors into the decoder's hidden space.
3. **QLoRA-tuned compact decoder-only SLM** (Qwen-2.5-1.5B by
   default; Phi-3.5-mini, Llama-3.2-1B/3B, TinyLlama-1.1B as
   alternatives) consuming the sequence of projected vectors via
   ``inputs_embeds``.

The decoder's last-position hidden state feeds a binary classification
head; the head's logits are calibrated post-hoc with temperature
scaling (Guo et al., ICML 2017) and routed through a selective
predictor with a per-deployment risk-budgeted τ.

### Intended use

System-operational log anomaly detection: kernel logs, distributed
filesystem logs, supercomputer node logs, OpenStack instance logs,
and similar machine-generated streams. The model returns a calibrated
probability and an explicit abstain channel for sequences below the
configured confidence threshold.

### Not intended for

- **Sole automated decision-making in safety-critical incident response.**
  HyLog is a *recommender* with an explicit abstain channel. A human
  operator must verify the routing before any irreversible action.
- **Surveillance over user-keyed activity logs.** HyLog was trained on
  *system-operational* logs. Applying it to user-keyed personal logs
  raises distinct ethical and legal concerns and requires a separate
  ethics review (see §6 dual-use disclosure).
- **Languages other than English.** All training data is English-locale
  system logs.

---

## 2. Training data

| Dataset | Source | Size | Anomaly granularity |
|---|---|---|---|
| HDFS | logpai/loghub-2.0 | ~11 M lines / 575 K sessions | block-level |
| BGL | USENIX CFDR | ~4.7 M lines | line-level |
| Thunderbird | USENIX CFDR | ~16 M lines | line-level |
| OpenStack | logpai/loghub | ~207 K lines | instance-level |

HyLog uses the Loghub-2.0 re-annotated splits with HyLog's chronological,
group-disjoint split policy (Phase 1, `splits/<system>.json`). For the
cross-system claim N2, the four-fold leave-one-system-out protocol
holds out each system in turn with **zero target labels** consumed
during training.

---

## 3. Evaluation

### Metric panel (Phase 4–5)

| Metric | Definition | HyLog status |
|---|---|---|
| F1 | per-sequence binary anomaly | ✅ panel |
| AUC-ROC, AUC-PR, MCC, FPR@R=0.95 | standard imbalanced-data robustness | ✅ panel |
| ECE / MCE | Guo-2017 calibration | ✅ ≤ 0.05 after temperature scaling |
| AURC + Excess-AURC | selective prediction quality | ✅ archived per fold |
| Cost-asymmetric AURC | FN-weight = 5× FP-weight | ✅ archived per fold |
| 95 % bootstrap CI on every metric | Q1-grade precision | ✅ per fold |
| Paired Wilcoxon + Holm-Bonferroni | head-to-head significance | ✅ per axis |

### Cross-system LOSO (Phase 4)

Reported as a 3-fold Core LOSO over {HDFS, BGL, Thunderbird} and a
4-fold Sensitivity LOSO with OpenStack. Each fold runs 5 seeds; the
mean ± std is the published number. The information-leakage audit
(`hylog.evaluation.leakage_audit`) gates every fold.

Head-to-head with published baselines is rendered automatically by
`hylog.evaluation.baseline_comparison` against:

- MetaLog (ICSE 2024) — few-label cross-system
- ZeroLog (arXiv 2511.05862) — zero-label cross-system
- Few-to-Zero-Label MetaLog (arXiv 2507.19806)
- Bridging the Gap (arXiv 2412.15445)

### Ablation matrix (Phase 6)

Eight axes (A1–A8) on the same 5 seeds with paired Wilcoxon +
Holm-Bonferroni + Cliff's δ. The full matrix is in
`reports/phase6/runs/{run_name}/ablation_matrix.csv`.

A1 is the N4 head-to-head: hybrid vs. standalone QLoRA decoder at
equal trainable parameter count.

---

## 4. Hardware + inference SLO

### Inference deployment

- **Minimum GPU:** 6 GB VRAM (4-bit NF4 quantized).
- **Recommended GPU:** NVIDIA T4 (16 GB) or L4 (24 GB).
- **CPU fallback:** possible at ~10× higher latency, recommended only
  for offline batch.
- **Disk:** ~1.7 GB for the quantized model artefact.
- **p95 latency target:** < 50 ms for a batch of 8 sequences ×
  100 lines on RTX 3090. Phase-8 contract.

### Training requirements

- 24 GB VRAM minimum (RTX 3090 / 4090 / A10G).
- ~500 GPU-hours for the full Phase 2 → 6 reproduction.

---

## 5. Limitations

- **Trained on Loghub-2.0**. Novel target systems may require
  re-calibration (the `hylog-calibrate` CLI handles this in one
  command on a held-out target slice).
- **English-only.** Non-English log content is out of distribution.
- **Concept drift.** Production deployments inevitably see distribution
  drift away from training. HyLog ships a drift monitor (Phase-8
  `src/hylog/inference/drift.py`) that flags shifts via the
  two-sample KS test. Re-calibration is the lightweight remedy;
  full re-training is the heavyweight remedy.
- **GTX 1050 Ti / Pascal-era hardware** is unsupported for QLoRA
  training. Inference works at 4-bit on any compute-capability-6+
  GPU but the 4-bit kernels are unoptimised below Turing.

---

## 6. Ethical considerations + dual-use disclosure

- **Dual-use.** Log anomaly detection technology can be repurposed for
  surveillance over user activity logs. HyLog's intended-use clause
  explicitly limits application to *system-operational* logs.
- **Sole-decision-maker prohibition.** Production deployments must
  never use HyLog as the sole automated decision-maker in
  safety-critical incident response. The selective predictor's abstain
  channel + drift monitor are the operational evidence that an
  abstention pathway is provisioned.
- **Bias.** The training mix is dominated by HDFS (Hadoop) and BGL
  (Blue Gene/L supercomputers). Performance on systems with very
  different vocabulary or anomaly modes may be lower than the
  cross-system LOSO numbers suggest.

---

## 7. Reproducibility

```powershell
git clone https://github.com/moslem-mohseni/hyperlog.git
cd hyperlog
pip install -r requirements-lock.txt
pip install -e .
.\scripts\reproduce_all.ps1
```

Live transcript on CPU-only Windows: 8/8 stages green in ~95 s. With a
24 GB GPU, the same script reproduces every published number within
±0.5 absolute F1.

---

## 8. Citation

If you use HyLog in academic work, please cite via the repository's
`CITATION.cff` file.

---

## 9. Contact

Moslem Mohseni Khah — Issues and questions on
https://github.com/moslem-mohseni/hyperlog/issues
