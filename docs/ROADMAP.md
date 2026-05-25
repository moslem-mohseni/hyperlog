# HyLog — Hybrid Small-Language-Model Pipeline for Cross-System Log Anomaly Detection with Calibrated Uncertainty

> **Project codename:** **HyLog** (Hybrid Log). An earlier working name "HyperLog" was retired because of a naming clash with HyperLogLog, the well-known probabilistic data structure.

**Author:** Moslem Mohseni Khah
**Document version:** v0.4 (post round-3 critique — final pre-implementation)
**Target venues:** IEEE Transactions on Network and Service Management (Q1), IEEE Open Journal of the Computer Society (Q1), IEEE Transactions on Reliability (Q1)
**Status:** Planning. No code written yet. Time-agnostic phasing.

---

## 0. How to read this document

The roadmap is **phase-gated**: each phase has an entry contract, deliverables, a test/checklist block, and an explicit **exit gate**. No phase begins until the previous one's exit gate is signed off. Every phase also lists a **kill-switch / fallback** so that the project never silently fails.

Numbered §-references inside phases (e.g. "see §11.3") point to risk-register entries that the planning phase has already pre-analyzed.

---

## 1. Project Description

### 1.1 Elevator pitch (one paragraph)
HyLog is a research-grade and production-grade system that detects anomalies in system logs across heterogeneous software systems (Hadoop/HDFS, Blue Gene/L, Thunderbird, OpenStack, …) using a **three-stage hybrid encoder–decoder pipeline** — a frozen masked language encoder produces semantic embeddings of individual log lines, a small **learned projector** aligns those embeddings into the input space of a **modern compact decoder-only language model in the 1–4 B parameter band** (Qwen-2.5-1.5B primary, Phi-3.5-mini-instruct 3.8 B secondary, Llama-3.2-1B / 3B as additional configurations), and that decoder is **parameter-efficiently fine-tuned with QLoRA (4-bit NF4)** to emit anomaly judgments over log sequences. The pipeline is evaluated under a **strict cross-system protocol** (train on a source system, test on a held-out target system with zero target labels) and equipped with **post-hoc temperature-scaling calibration** plus **selective prediction with risk-coverage guarantees**. The end deliverable is (i) a reproducible open-source codebase on GitHub, (ii) a deployable inference service with a documented latency / VRAM budget on commodity 24 GB GPUs, and (iii) a Q1 journal paper.

> **Wording note (corrected after critique):** earlier drafts said "sub-2 B decoder". Phi-3.5-mini is 3.8 B, so the precise descriptor is "**compact decoders in the 1–4 B band**". Sub-2 B is the *primary* operating point (Qwen-2.5-1.5B); larger configurations are reported as secondary points to study scaling.

### 1.2 Why this project, why now
The literature review of 25 papers (2017–2026) identified a tri-junction gap that no published work simultaneously fills:

1. **Modern compact decoder-only SLMs** (Qwen-2.5, Phi-3.5, Llama-3.2 — all released 2024 or later) are largely unexplored as the *decoder half of a hybrid* LAD pipeline. LogLLM uses Llama-2/3-7B; LogFiT uses Longformer/RoBERTa encoders only; LogTinyLLM uses tiny decoders in isolation (no hybrid); LogADReft uses LoRA on RoBERTa/GPT-2/Llama-3 but as standalone fine-tuning, not in a frozen-encoder + projector + decoder architecture.
2. **Cross-system generalization** (train on system A, deploy on system B with zero target labels) remains weak — MetaLog requires a few target labels, ZeroLog reaches ~80 % F1 but uses traditional embeddings without modern SLM decoders.
3. **Calibration and uncertainty quantification** are absent from essentially every published LAD pipeline. Reliability diagrams, ECE, and selective prediction are unused in the dominant baselines, despite being prerequisites for any safety-critical deployment.

### 1.3 Novelty claims (with explicit scoping)
- **N1 (architectural).** First hybrid frozen-encoder ↔ learned-projector ↔ QLoRA-tuned **modern compact (1–4 B)** decoder-only SLM pipeline for log anomaly detection. *Scope guardrail:* the closest prior art is LogLLM (Llama-7B decoder, no QLoRA) and LogADReft (LoRA on standalone LLMs, no hybrid encoder+projector); HyLog is the intersection.
- **N2 (empirical).** First systematic **leave-one-system-out** evaluation of a hybrid SLM LAD pipeline over ≥4 public datasets with **zero target labels**, head-to-head against MetaLog and ZeroLog under their own evaluation protocols.
- **N3 (uncertainty).** First LAD pipeline that ships with explicit post-hoc temperature-scaling calibration, ECE / MCE / reliability diagrams, and a risk-coverage selective-prediction curve.
- **N4 (ablation).** A clean component-isolation ablation answering "**is the hybrid encoder + projector + decoder architecture better than a single standalone QLoRA-tuned decoder with the same trainable parameter budget?**" — a question prior work has not answered head-to-head.

A **defensive scoop-check** (§11.1) is run at the start of every phase to verify that no new arXiv preprint has invalidated these claims.

### 1.4 Out of scope (explicit)
- Real-time streaming log ingestion at petabyte scale.
- Root-cause analysis, dependency-graph reasoning, multimodal logs.
- Federated or differentially private training (left to future work; see DP-FlogTinyLLM).
- Intrusion-log datasets (AIT-LDS, LANL) — listed as optional extension only.

---

## 2. Related Work (links re-verified during planning)

### 2.1 Foundational baselines
| # | Paper | Year | Link | Code |
|---|---|---|---|---|
| 1 | **DeepLog** — Du et al., CCS '17 | 2017 | https://dl.acm.org/doi/10.1145/3133956.3134015 | https://github.com/Thijsvanede/DeepLog (community) |
| 2 | **LogBERT** — Guo, Yuan, Wu | 2021 | https://arxiv.org/abs/2103.04475 | — |
| 3 | **LogFiT** — Almodovar et al., IEEE TNSM | 2024 | https://doi.org/10.1109/TNSM.2024.3358730 | — |

### 2.2 Hybrid / large-LM LAD (direct ancestor of HyLog)
| # | Paper | Year | Link | Code |
|---|---|---|---|---|
| 4 | **LogLLM** — Guan, Cao, Qian, Gao, Ouyang | 2024 | https://arxiv.org/abs/2411.08561 | https://github.com/guanwei49/LogLLM |

### 2.3 PEFT for log anomaly detection
| # | Paper | Year | Link | Code |
|---|---|---|---|---|
| 5 | **LogADReft** — Lim, Zhu, Pang; PAKDD '25 | 2025 | https://arxiv.org/abs/2503.08045 | https://github.com/mala-lab/LogADReft |
| 6 | **LogTinyLLM** | 2025 | https://arxiv.org/abs/2507.11071 | — |
| 7 | **AdaptiveLog** (LLM+SLM collaboration; ACM TOSEM) | 2025 | https://arxiv.org/abs/2501.11031 | — |

### 2.4 Cross-system generalization
| # | Paper | Year | Link | Code |
|---|---|---|---|---|
| 8 | **MetaLog** — ICSE '24 | 2024 | https://dl.acm.org/doi/10.1145/3597503.3639205 | — |
| 9 | **ZeroLog** | 2025 | https://arxiv.org/abs/2511.05862 | — |
| 10 | **From Few-Label to Zero-Label** | 2025 | https://arxiv.org/abs/2507.19806 | — |

### 2.5 Adjacent / benchmarks
| # | Paper | Year | Link | Code |
|---|---|---|---|---|
| 11 | **ADALog** | 2025 | https://arxiv.org/abs/2505.13496 | — |
| 12 | **AIOps for LAD — Systematic Review** | 2025 | https://www.sciencedirect.com/science/article/pii/S2667305325001346 | — |
| 13 | **Loghub-2.0** — ISSTA '24 | 2024 | https://github.com/logpai/loghub-2.0 | repo |

### 2.6 Methodological foundations
| # | Paper / Resource | Year | Link |
|---|---|---|---|
| 14 | **LoRA** — Hu et al. | 2021 | https://arxiv.org/abs/2106.09685 |
| 15 | **QLoRA** — Dettmers et al., NeurIPS '23 | 2023 | https://arxiv.org/abs/2305.14314 |
| 16 | **On Calibration of Modern Neural Networks** — Guo et al., ICML '17 | 2017 | https://arxiv.org/abs/1706.04599 |
| 17 | **Drain** — He et al., ICWS '17 | 2017 | https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf |

### 2.7 Model cards
| # | Model | Card |
|---|---|---|
| 18 | Qwen-2.5-1.5B (base) | https://huggingface.co/Qwen/Qwen2.5-1.5B |
| 19 | Qwen-2.5-1.5B-Instruct | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| 20 | Phi-3.5-mini-instruct (3.8 B) | https://huggingface.co/microsoft/Phi-3.5-mini-instruct |
| 21 | bitsandbytes (Windows-native ≥ 0.43.0) | https://github.com/bitsandbytes-foundation/bitsandbytes |

---

## 3. Datasets and dataset hygiene

| Dataset | System | Size | Anomaly labels | Use |
|---|---|---|---|---|
| HDFS | Hadoop distributed FS | ~11 M lines | block-level | Source / target |
| BGL | Blue Gene/L supercomputer | ~4.7 M lines | line-level | Source / target |
| Thunderbird | Sandia supercomputer | ~16 M lines | line-level | Source / target |
| OpenStack | OpenStack infra | ~207 K lines | instance-level | Sensitivity study only (§3.2) |
| Liberty | Sandia | ~265 M lines | line-level | Reported because LogLLM reports on Liberty; included for head-to-head completeness in Phase 3. Excluded from cross-system Phase 4 due to size imbalance. |

### 3.1 Splits and label-leakage hygiene
Two well-known pathologies of the legacy HDFS split must be avoided:
- **Temporal leakage** — random splits put future blocks in train and past blocks in test, inflating F1.
- **Block-ID leakage** — the same `blk_id` appearing in both train and test.

HyLog's split policy:
- **Chronological split** (LogLLM convention): 8 : 1 : 1 train / val / test in arrival order, with no shuffling that crosses session boundaries.
- **Disjoint group keys**: for HDFS, the `blk_id` is hashed and assigned to exactly one split. A unit test enforces this.
- **Loghub-2.0 versions** of every dataset are used (annotations re-verified in ISSTA '24). Legacy Loghub-1 splits are not used.

### 3.2 OpenStack imbalance caveat
OpenStack at ~207 K lines is ~80× smaller than Thunderbird. Including it as an equal LOSO fold biases averages. HyLog reports LOSO **two ways**:
- **Core LOSO (3 systems):** HDFS / BGL / Thunderbird (size-comparable).
- **Sensitivity LOSO (+ OpenStack):** included as a fourth fold but reported separately with a size-imbalance caveat.

### 3.3 Sequence granularity (per dataset)
LogLLM and the broader literature differ in how they group log lines into sequences. HyLog standardizes:

| Dataset | Grouping | Sequence definition | Typical sequence length (lines) |
|---|---|---|---|
| HDFS | **Session window** on `blk_id` | All lines for one block | mean ≈ 19, p95 ≈ 50 |
| BGL | **Fixed-stride sliding window** | 100-line window, stride 20 | exactly 100 |
| Thunderbird | **Fixed-stride sliding window** | 100-line window, stride 20 | exactly 100 |
| OpenStack | **Session window** on instance ID | All lines per instance | varies; truncate to 100 |

The 100-line window is the LogLLM default. Each line is encoded individually by BERT; the decoder consumes the resulting sequence of projected vectors, so the **decoder's token budget is the number of lines, not the number of sub-word tokens**. For Qwen-2.5-1.5B with up to 128 lines per sequence, the decoder context is two orders of magnitude below the model's 128 K limit — VRAM is gated by activation memory, not by context length.

### 3.4 Anomaly granularity (what HyLog detects)
HyLog produces a **per-sequence binary label** (anomaly / normal). This matches the labelling convention of LogLLM, LogFiT, MetaLog, and ZeroLog and is the convention all comparison numbers in the literature use. Per-line anomaly localization is **explicitly out of scope** for v1.0 and listed as future work in the paper.

### 3.5 Licensing
- HDFS: Apache 2.0 (logpai distribution).
- BGL / Thunderbird: USENIX CFDR distribution; redistribution within research artifacts is permitted with citation.
- OpenStack: logpai re-distribution under research-use.
- HyLog **does not redistribute raw data**; the `scripts/download_data.ps1` script fetches archives from the original Loghub-2.0 mirrors at run time. License attribution is auto-emitted into `data/LICENSES.txt`.

---

## 4. Technical Architecture

```
                    raw log line
                          │
                          ▼
              ┌─────────────────────┐
              │  regex preprocessor │  (LogLLM-style, parser-free)
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  frozen BERT encoder│  (bert-base-uncased; 768-d)
              └─────────────────────┘
                          │  (one vector per log line)
                          ▼
              ┌─────────────────────┐
              │ learned projector   │  (2-layer MLP; trainable)
              │  768 → d_model_SLM  │
              └─────────────────────┘
                          │  (sequence of N projected vectors)
                          ▼
              ┌─────────────────────┐
              │ QLoRA-tuned SLM     │  (Qwen-2.5-1.5B base; 4-bit NF4
              │ decoder (1–4 B)     │   + LoRA r=8/16, α=16/32,
              │                     │   target = q_proj,k_proj,v_proj,o_proj)
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ classification head │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ temperature scaler  │  (post-hoc, fitted on calibration set)
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ selective predictor │  (risk-coverage; abstain below τ)
              └─────────────────────┘
                          │
                          ▼
                normal / anomaly / abstain
```

**Why parser-free regex (no Drain)?** Drain (#17) is excellent but introduces a brittle templatization step that loses arguments and is hard to keep stable across systems. LogLLM and LogFiT empirically show that masked-regex preprocessing performs as well or better while staying simpler — HyLog follows that line and reports a Drain ablation in Phase 6 (A7).

**Three training stages** (LogLLM-derived), with explicit losses:
1. **Stage 1 — Projector warm-up.** Encoder frozen, decoder frozen. The decoder head emits class logits; loss is **weighted cross-entropy** (class weights inversely proportional to support, capped at 10× to avoid degenerate gradients). Only the projector receives gradient.
2. **Stage 2 — Joint adapter training.** Projector + QLoRA adapters in the decoder train jointly. Same weighted cross-entropy. Encoder remains frozen.
3. **Stage 3 — End-to-end refinement.** LoRA + projector at 0.1× the Stage-2 learning rate; encoder remains frozen. Early stop on validation F1.

**Inference decoding strategy.** Classification head sits on top of the decoder's last hidden state at the final position; HyLog does *not* use autoregressive token sampling. The output is a single softmax over {normal, anomaly}. This makes the calibration story rigorous (temperature scaling on logits is exactly the Guo-2017 setting) and avoids the sampling-variance trap that hits generative scoring schemes.

### 4.1 VRAM budget (sanity-checked, not aspirational)
Rough back-of-the-envelope for Qwen-2.5-1.5B with **128 line-vectors per sequence** (decoder input length = 128 projected vectors) and micro-batch size 4:
- Frozen BERT-base in fp16: ~0.4 GB.
- Qwen-2.5-1.5B in 4-bit NF4: ~1.2 GB weights.
- Decoder activations at L=128 (vectors, not sub-word tokens), B=4: ~3–4 GB.
- BERT-encoder activations for 4 × 128 = 512 lines (each truncated to 64 sub-word tokens): ~3–4 GB.
- LoRA adapters + optimizer state (Adam, fp32 masters): ~1–2 GB.
- Gradient activations + workspace: ~3–5 GB.
- **Working budget: ~12–17 GB. Target hardware: 24 GB (RTX 3090 / 4090 / A5000). Headroom: 7–12 GB.**

Note on the unit of "sequence length": throughout HyLog, **L = number of log lines** entering the decoder (each line is one projected vector). The decoder's native sub-word token length matters only inside the BERT encoder per-line; for that we cap at 64 sub-word tokens per line, which covers > 99 % of LAD log lines without truncation.

If empirical VRAM exceeds budget in Phase 3, fallback knobs (in order): reduce micro-batch to 2 with grad-accum 2; reduce L from 128 lines to 64; switch to gradient checkpointing; drop projector second layer. All four are pre-budgeted in the trainer config.

### 4.2 Compute budget (planning estimate, not aspirational)
Per single-seed Phase-3 training on one (decoder × dataset) cell with Stage-1+2+3 ≈ ~6 hours on RTX 3090 (estimated from LogLLM training-time reports scaled to 1.5 B parameters with QLoRA's ~3× speedup over full fine-tune).
- Phase 2 reproduction (1 model × 2 datasets × 5 seeds) ≈ 60 h.
- Phase 3 core (1 decoder × 2 datasets × 5 seeds) ≈ 60 h.
- Phase 4 LOSO (1 decoder × 4 folds × 5 seeds) ≈ 120 h.
- Phase 6 ablation (8 axes × ~3 cells × 5 seeds, with shared runs amortized) ≈ 250 h.
- **Total ≈ 500 GPU-hours on a single 24 GB GPU.** This is a heavy but tractable budget on one machine; the kill-switch is to reduce seed count from 5 → 3 if budget overruns by > 30 %.

### 4.3 Hyperparameter search strategy
- **Stage-1 sweep:** projector depth ∈ {1, 2}, projector hidden ∈ {1024, 2048}, lr ∈ {1e-4, 5e-4} — 8 cells, 1 seed, narrowed to a single Stage-1 config before Stage-2 begins.
- **Stage-2 sweep:** LoRA rank ∈ {8, 16}, α ∈ {16, 32}, lr ∈ {1e-4, 5e-5} — 8 cells, 1 seed, narrowed to a single config.
- After narrowing, the *full ablation grid* (§Phase 6) runs with 5 seeds on the locked configs.
- All sweeps are tracked in MLflow; no hand-tuning after the sweep is locked.

---

## 4bis. Evaluation metrics (the full set)

F1 alone is insufficient for severely imbalanced LAD data. HyLog reports a **fixed metric panel** on every experiment:

| Metric | Why it is in the panel |
|---|---|
| **Precision / Recall / F1** | Convention; head-to-head comparability with prior LAD work. |
| **AUC-ROC** | Threshold-independent ranking quality. |
| **AUC-PR** | More informative than AUC-ROC under heavy class imbalance. |
| **MCC** (Matthews correlation coefficient) | Balanced single-number score robust to imbalance. |
| **FPR @ Recall = 0.95** | Operational metric — how many false alarms to catch 95 % of anomalies. |
| **ECE / MCE** | Calibration (Phase 5). |
| **AURC** (area under risk-coverage curve) | Selective prediction standard metric — see §11.4. |
| **Excess-AURC (E-AURC)** | AURC normalized against an oracle selector — fair across base accuracies. |

The headline number in the paper is **macro-F1** for comparability; AURC is the headline for the selective-prediction story.

---

## 4ter. Baseline ladder (what HyLog must beat or match)

HyLog reports against a *ladder* of baselines of increasing sophistication. Every baseline is re-implemented or run from official code on **the same splits**:

1. **B0 — Trivial.** Predict majority class. Sets the floor.
2. **B1 — Classical ML.** TF-IDF on Drain templates → Logistic Regression and Isolation Forest. (If a Drain+LR baseline already gets F1 = 99 on HDFS, the entire SLM stack must justify its cost.)
3. **B2 — DeepLog** (LSTM next-event prediction).
4. **B3 — LogBERT** (self-supervised BERT).
5. **B4 — LogFiT** (fine-tuned Longformer).
6. **B5 — LogLLM** (the direct ancestor; Phase 2 reproduction).
7. **B6 — Standalone QLoRA-tuned decoder** (Qwen-2.5-1.5B without the BERT encoder + projector hybrid). This is the **N4 head-to-head** baseline.

For cross-system (Phase 4) the ladder narrows to **MetaLog, ZeroLog, and B6** — the only published methods that report leave-one-system-out at this scale.

---

## 5. Phase Plan (no time deadlines)

Each phase: **entry contract → deliverables → tests/checklist → exit gate → kill-switch**.

### Phase 0 — Foundation & Environment

**Entry contract:** Windows 10/11 dev machine with an NVIDIA GPU ≥ 24 GB VRAM, CUDA 12.x driver, Python 3.11.

**Deliverables:**
- `pyproject.toml` (PEP 621) is the **single** source of dependency truth; no `requirements.txt`. Pinned versions: `torch==2.4.*`, `transformers>=4.45,<4.50`, `peft>=0.12,<0.14`, `bitsandbytes>=0.43.0` (this is the first version with **official** Windows wheels — Critical: see §11.2), `accelerate>=0.34`, `datasets`, `mlflow`, `pytest`, `ruff`, `mypy`, `hydra-core` (config), `omegaconf`.
- `src/hylog/` skeleton with empty modules: `data/`, `models/`, `training/`, `evaluation/`, `calibration/`, `inference/`, `cli/`.
- `tests/` mirroring `src/`.
- `.gitignore`, `.editorconfig`, `LICENSE` (MIT), `CITATION.cff`, `README.md` (author: Moslem Mohseni Khah).
- GitHub Actions `ci.yml`: `windows-latest` and `ubuntu-latest` matrix → ruff → mypy → pytest (CPU-only tests).
- Pre-commit hooks (`.pre-commit-config.yaml`).
- First commit pushed to GitHub (user provides PAT).

**Tests / Checklist:**
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
- [ ] `python -c "import bitsandbytes as bnb; print(bnb.__version__)"` → `>= 0.43.0`, no DLL error.
- [ ] `python -c "from bitsandbytes.nn import Linear4bit"` succeeds.
- [ ] A smoke training step on a 2-line toy dataset with `BitsAndBytesConfig(load_in_4bit=True)` and a 125 M parameter test model completes without error.
- [ ] `pytest` runs with zero failures and ≥ 1 passing smoke test.
- [ ] `ruff check src tests` returns zero violations.
- [ ] `mypy src` returns zero errors (strict mode on public APIs).
- [ ] GitHub Actions CI green on both OS matrices.
- [ ] Tag `v0.0.1-skeleton` pushed.

**Exit gate:** all checklist items ticked; no item is "best-effort".

**Kill-switch:** if bitsandbytes on Windows cannot be imported after a documented installation procedure (§11.2), the project pivots to a **WSL2 + Ubuntu development path** while keeping the published reproduction script in PowerShell. This is the single biggest Windows-specific risk and is decided **before** any modeling work.

### Phase 1 — Data Pipeline

**Entry contract:** Phase 0 exit gate passed.

**Deliverables:**
- `src/hylog/data/loaders/` with one module per dataset.
- `LogDataset` abstraction (PyTorch `Dataset`) returning `(sequence_of_lines, label, group_id)`.
- Regex preprocessor `src/hylog/data/preprocess.py` — LogLLM regex set, every regex documented inline with the original LogLLM source citation.
- Sliding-window sequencer (configurable window length and stride).
- `splits/` manifest directory with deterministic JSON files: byte-level reproducibility.
- `scripts/download_data.ps1` (Windows PowerShell) and `scripts/download_data.sh` (mirror).
- `data/LICENSES.txt` auto-emitted from a `licenses.yaml`.

**Tests / Checklist:**
- [ ] Each loader unit-tested against a 100-line synthetic fixture: count, label distribution, regex output match.
- [ ] Loading the same dataset twice yields **byte-identical** split manifests (SHA-256 check).
- [ ] **Group-disjointness invariant**: `set(train.group_ids) ∩ set(test.group_ids) == ∅` enforced by a unit test on every dataset.
- [ ] Per-split anomaly % recorded in `splits/manifest.json` and within ±0.2 % of LogLLM-reported numbers; deviation auto-flagged.
- [ ] Property test: union of sliding windows reconstructs the raw file up to stride.
- [ ] Loghub-2.0 SHA-256 of every downloaded archive is verified against a committed `checksums.txt`.
- [ ] Tag `v0.1.0-data` pushed.

**Exit gate:** all checklist items ticked; split manifests committed to git.

**Kill-switch:** if Loghub-2.0 mirrors are unreachable, fall back to a documented local-mirror procedure with the same SHA-256s.

### Phase 2 — Feasibility check + Baseline reproduction (LogLLM)

**This is two sub-phases.** The feasibility check is new (added after round-1 critique).

#### Phase 2A — Feasibility check (gating)
- Clone https://github.com/guanwei49/LogLLM. Run their training script unmodified on a small HDFS subset. **Goal:** demonstrate that the upstream code runs end-to-end on Windows in our environment.
- Time-box: if the upstream code does not run after a documented adaptation effort, escalate. Two pre-decided escape paths: (a) run upstream on a Linux WSL2 mirror to obtain reference numbers and continue HyLog development on Windows; (b) fall back to LogFiT (https://doi.org/10.1109/TNSM.2024.3358730) as the architectural ancestor instead, which is simpler.

#### Phase 2B — Faithful reproduction
- `src/hylog/models/baselines/logllm.py` — a re-implementation cross-validated against the upstream repo, with explicit parity comments citing upstream line numbers.
- Run LogLLM (their reported configuration) on HDFS and BGL.

**Tightened tolerance after critique:** F1 must reproduce within **±1.0** absolute points on HDFS (paper-reported ≥ 99 means our tolerance is ~1 % relative) and **±2.0** on BGL (where seed variance is empirically larger).

**Tests / Checklist:**
- [ ] Phase 2A: upstream LogLLM repo executes the published training script in our environment (Windows-native or WSL2) without code changes.
- [ ] Phase 2B: F1 on HDFS within ±1.0 of LogLLM-reported.
- [ ] Phase 2B: F1 on BGL within ±2.0 of LogLLM-reported.
- [ ] All loss curves and gradient norms archived in MLflow.
- [ ] `reports/phase2/reproduction.md` written, including any documented deviations and their justifications.
- [ ] Tag `v0.2.0-baseline` pushed.

**Exit gate:** F1 tolerances met **or** a documented and reviewer-defensible deviation note.

**Kill-switch:** if reproduction fails on both datasets, pivot to LogFiT-style ancestor (a simpler, fully-published baseline) and update novelty claim N1's scoping accordingly. Project does not silently abandon.

### Phase 3 — Backbone Substitution (HyLog core)

**Goal:** Replace the Llama-7B decoder with Qwen-2.5-1.5B (primary), Phi-3.5-mini-instruct (secondary), Llama-3.2-1B / 3B (additional points for scaling study).

**Deliverables:**
- `src/hylog/models/decoder.py` — pluggable decoder registry.
- `src/hylog/models/projector.py` — projector with `hidden_size` auto-discovery from `decoder.config`.
- `src/hylog/training/qlora_trainer.py` — three-stage trainer.
- Per-decoder Hydra configs under `configs/decoders/`.

**Tests / Checklist:**
- [ ] In-domain F1 on HDFS (Qwen-2.5-1.5B) **≥ Phase-2 LogLLM-on-HDFS F1 − 1.0**.
- [ ] In-domain F1 on BGL (Qwen-2.5-1.5B) **≥ Phase-2 LogLLM-on-BGL F1 − 1.0**.
- [ ] Peak VRAM during training **≤ 22 GB** on RTX 3090 / 4090 (headroom in 24 GB).
- [ ] Trainable parameter count (projector + LoRA) < 5 % of decoder full count.
- [ ] All three training stages converge (val loss monotone non-increasing across the last 20 % of each stage).
- [ ] 5-seed runs; standard deviation per metric reported.
- [ ] Tag `v0.3.0-core` pushed.

**Exit gate:** all checklist items ticked.

**Kill-switch:** if Qwen-2.5-1.5B underperforms by > 1.0 absolute F1 across both datasets and 5 seeds, escalate. Options: (a) switch primary decoder to Phi-3.5-mini (3.8 B); (b) increase LoRA rank; (c) increase projector capacity. Each option is pre-budgeted in configs.

### Phase 4 — Cross-System Evaluation Protocol (LOSO)

**Goal:** Demonstrate generalization. Train on three of {HDFS, BGL, Thunderbird}, test on the fourth with zero target labels. OpenStack reported separately (§3.2).

**Deliverables:**
- `src/hylog/evaluation/cross_system.py`.
- Result matrices in `reports/phase4/` for **Core LOSO (3-fold)** and **Sensitivity LOSO (+OpenStack)**.
- Head-to-head tables vs. MetaLog and ZeroLog using **the same evaluation protocol** they used (we recompute, not just cite).

**Tests / Checklist:**
- [ ] Core-LOSO macro-F1 (3 folds) **≥ ZeroLog macro-F1 on overlapping datasets** (≥ 80 %).
- [ ] Information-leakage unit test: target-system raw lines hash-set check against training-batch hash-set — must be disjoint.
- [ ] Per-fold confusion matrices archived as PNG + CSV.
- [ ] 5-seed runs per fold; reported as mean ± std.
- [ ] Tag `v0.4.0-crosssys` pushed.

**Exit gate:** all items ticked.

**Kill-switch:** if Core-LOSO macro-F1 < 70 %, run the diagnostic ablation set early (Phase 6 partial) to locate the bottleneck before publishing claims. Options to recover: (a) augment training with a small fraction of target-system **unlabeled** logs via masked self-prediction; (b) add a domain-adversarial loss à la ZeroLog. Both are pre-coded paths.

### Phase 5 — Calibration & Selective Prediction

**Goal:** Make HyLog the first LAD pipeline shipped with usable uncertainty.

**Deliverables:**
- `src/hylog/calibration/temperature.py` — Guo-2017 temperature scaling.
- `src/hylog/calibration/metrics.py` — ECE, MCE, reliability diagrams.
- `src/hylog/inference/selective.py` — risk-coverage curves with an **auto-τ selector** that targets a configurable risk budget.
- Reliability diagrams as PNG + bin data as CSV for every (backbone, dataset, fold) tuple.

**Threshold-selection rigor (post-critique):**
- The calibration set is **disjoint from train, val, and test**; a 5 % held-out slice of train is used.
- For **cross-system** evaluation, τ is fitted on the **source** system's calibration slice and frozen — this honors the zero-label constraint. A sensitivity analysis reports how τ behaves under target-system distribution shift (Phase 6 A8).

**Calibration target — justified, not arbitrary:**
- **ECE ≤ 0.05** matches the typical "well-calibrated" threshold cited in Guo et al. 2017 (post-temp-scaling ECE on CIFAR/SVHN is typically 0.01–0.04).

**Tests / Checklist:**
- [ ] Post-calibration ECE ≤ 0.05 on every (backbone, dataset, fold) tuple.
- [ ] Reliability diagrams archived as PNG + bin CSV per (backbone, dataset, fold).
- [ ] Risk-coverage curves monotone non-increasing in **risk** as coverage decreases (the correct monotonicity check).
- [ ] **AURC strictly lower than B6 (standalone QLoRA decoder) AURC** on at least 3 of 4 LOSO folds — this is the rigorous "selective prediction helps" test (replaces the earlier informal "F1 at 80 % coverage" check).
- [ ] **Excess-AURC** reported alongside AURC.
- [ ] Cost-asymmetric variant: selective error computed with FN-weight = 5× FP-weight (operational LAD cost asymmetry) reported in addition to symmetric AURC.
- [ ] All calibration artefacts reproducible from a single seed (deterministic).
- [ ] Tag `v0.5.0-calibrated` pushed.

**Exit gate:** all items ticked.

**Kill-switch:** if temperature scaling fails to bring ECE below 0.05 on some folds, escalate to **Platt scaling** and **vector scaling**; both are pre-implemented as alternatives. Worst case, ship per-fold reliability data and report it honestly — the *honest* reliability diagram is itself a contribution over the silent baselines.

### Phase 6 — Ablation & Statistical Validation

**Goal:** Defend N4 head-to-head and quantify every design choice.

**Ablation matrix:**
- **A1 (the head-to-head for N4):** hybrid (encoder + projector + decoder) vs. **standalone QLoRA-tuned decoder of equal trainable parameter count**. Trainable-parameter equality is enforced by configuring the standalone decoder's LoRA rank such that its trainable count matches projector + LoRA.
- **A2:** LoRA rank ∈ {4, 8, 16, 32}.
- **A3:** target modules ∈ {Q only, QV, QKVO}.
- **A4:** projector depth ∈ {1, 2, 3 layers}.
- **A5:** encoder frozen vs. encoder LoRA-tuned.
- **A6:** with vs. without temperature scaling (ECE comparison).
- **A7:** regex preprocessor vs. Drain templatization.
- **A8:** τ-selection source vs. target system (calibration-shift study).

**Statistical rigor:**
- All conditions run with the **same 5 seeds**.
- Wilcoxon signed-rank paired tests across seeds.
- Effect sizes: Cliff's δ.
- Multiple-comparisons correction: Holm–Bonferroni across the 8 ablation axes.

**Tests / Checklist:**
- [ ] Every ablation cell completed for all 5 seeds.
- [ ] All p-values, corrected p-values, and effect sizes in a single CSV.
- [ ] **A1** shows a statistically significant improvement (Holm-corrected p < 0.05, |Cliff's δ| > 0.33) **or** is reported as a negative result with full transparency. A negative result does not invalidate the project — it would still be a publishable contribution given the calibration novelty.
- [ ] Tag `v0.6.0-ablation` pushed.

**Exit gate:** all items ticked; honest reporting of negatives.

**Kill-switch:** if A1 is a clear negative (hybrid loses to standalone), pivot the paper's framing: lead with N3 (calibration + selective prediction) instead of N1, and present A1 as a *valuable negative result*. The codebase still ships; the paper still publishes; the venue might shift (e.g. an empirical-software-engineering venue more receptive to negative results).

### Phase 7 — Hardening & Reproducibility

**Goal:** Every result re-runnable on a fresh Windows machine in one command.

**Deliverables:**
- `scripts/reproduce_all.ps1` — one-shot Windows reproduction.
- `scripts/reproduce_all.sh` — Linux mirror.
- `run_manifest.json` per training run: git SHA, config hash, dataset hash, library versions, GPU model, CUDA version, seeds, peak VRAM, wallclock.
- `environment.yml` (conda) and locked `pip` requirements.
- MLflow tracking config + HTML exporter.
- Docker image (Linux) for cloud-side reproducibility.
- Self-hosted GPU CI runner executing one smoke training step.

**Tests / Checklist:**
- [ ] Fresh-clone + `reproduce_all.ps1` on a clean Windows VM regenerates every headline number within ±0.5 absolute F1.
- [ ] `run_manifest.json` schema validated by JSON Schema in CI.
- [ ] GPU CI runner green.
- [ ] Docker image builds and runs.
- [ ] Tag `v0.7.0-reproducible` pushed.

**Exit gate:** all items ticked; an independent re-run on a colleague's Windows machine succeeds.

### Phase 8 — Production Inference Service

**Goal:** Ship a deployable REST service.

**Deliverables:**
- `src/hylog/cli/` with two entry points: `hylog-train` and `hylog-predict` (Click-based; documented in README). Exposed as `[project.scripts]` in `pyproject.toml`.
- `src/hylog/inference/server.py` — FastAPI service.
- Export path: ONNX or TorchScript; vLLM optional.
- Documented SLO: p95 latency < 50 ms for a 100-line batch of size 8 on RTX 3090.
- Dockerfile (Linux) + Windows-native run script.
- Model card following the HF template, authored by Moslem Mohseni Khah.
- `clients/python/` SDK example.

**Tests / Checklist:**
- [ ] End-to-end integration test passes (HTTP request → response with calibrated probability + abstention flag).
- [ ] Load test: 100 req/s sustained for 5 min, p95 < 50 ms.
- [ ] OpenAPI spec auto-generated and committed.
- [ ] Model card includes LOSO matrix, ECE, intended use, dual-use disclosure (§11.8), explicit "not a sole decision-maker" clause, drift-monitoring guidance.
- [ ] Security checklist (§11.9) implemented and unit-tested: input cap, rate limit, no-echo errors, API-key auth.
- [ ] Response schema matches §11.7 exactly; contract test enforced.
- [ ] CLI smoke test: `hylog-predict --input fixtures/sample.jsonl` exits 0 and returns schema-valid output.
- [ ] Tag `v0.8.0-service` pushed.

**Exit gate:** all items ticked.

### Phase 9 — Paper Artefacts & Public Release

**Goal:** Manuscript + tagged 1.0.0.

**Deliverables:**
- LaTeX manuscript in `paper/` (IEEE template).
- Auto-regenerated figures via `scripts/build_figures.py`.
- Reproducibility appendix: every table cell traceable to a JSON under `reports/`.
- Zenodo-archived release with DOI.
- GitHub repo public.

**Tests / Checklist:**
- [ ] Every numerical claim has a backing JSON in `reports/`.
- [ ] `latexmk` builds with zero warnings beyond IEEE template defaults.
- [ ] README "Reproducing the paper" section with exact commands.
- [ ] Zenodo DOI minted.
- [ ] Tag `v1.0.0` pushed; GitHub release auto-generated.

**Exit gate:** all items ticked; repository is public.

---

## 6. Path to "production-grade, working, innovative"

- **Production-grade** = reached at Phase 8 (deployable service with SLOs, calibrated outputs, model card).
- **Innovation** = N1 (architectural), N2 (cross-system empirical), N3 (calibration first), N4 (clean ablation).
- **Windows-first** = a Windows-native dev + reproduction path is itself an under-served contribution.

A **measurable v1.0.0 success contract** (concrete numerical targets, not adjectives):

| # | Metric | Target | Rationale |
|---|---|---|---|
| S1 | In-domain F1, HDFS, Qwen-2.5-1.5B | ≥ 99.0 | LogLLM reports ~99.5; we must be within 0.5 of SOTA. |
| S2 | In-domain F1, BGL, Qwen-2.5-1.5B | ≥ 96.0 | LogLLM-class performance band. |
| S3 | In-domain F1, Thunderbird | ≥ 95.0 | LogFiT-class performance band. |
| S4 | Core-LOSO macro-F1 (3 folds) | ≥ 80.0 | Matches ZeroLog's zero-label number. |
| S5 | Post-calibration ECE (every fold) | ≤ 0.05 | Guo-2017-class calibration. |
| S6 | AURC vs. B6 standalone decoder | strictly lower on ≥ 3 of 4 folds | Selective prediction is value-add over a single decoder. |
| S7 | p95 inference latency, RTX 3090, batch 8 × 100 lines | ≤ 50 ms | Production-grade SLO. |
| S8 | Peak training VRAM | ≤ 22 GB | Fits in consumer 24 GB GPU with headroom. |
| S9 | One-command reproduction on a clean Windows VM | green | "Reproducible" is a checklist, not a promise. |
| S10 | Public GitHub repo + Zenodo DOI | both live | Release discipline. |

A miss on any single target does not kill the project, but every miss is explicitly reported in the paper and a kill-switch from §10 is invoked.

---

## 7. Repository layout (target state at v1.0.0)

```
E:\Project\
├── docs\
│   ├── 00_initial_materials\          (existing — frozen)
│   ├── ROADMAP.md
│   ├── ARCHITECTURE.md
│   └── REPRODUCING.md
├── src.hylog.
│   ├── data\
│   ├── models\
│   ├── training\
│   ├── evaluation\
│   ├── calibration\
│   ├── inference\
│   └── cli\
├── tests\
├── configs\hydra-based
├── experiments\
├── reports\
├── scripts\
│   ├── download_data.ps1
│   ├── reproduce_all.ps1
│   └── reproduce_all.sh
├── paper\
├── .github\workflows\
├── pyproject.toml
├── README.md
├── LICENSE
└── CITATION.cff
```

---

## 8. GitHub deployment plan

- User provides PAT with `repo` scope.
- Suggested repo: `hylog-lad`.
- Branching: `main` (always green), `dev` (integration), feature branches `feat/phase-N-…`.
- Each phase ends with a signed tag `v0.N.0-…` and a release draft.
- Conventional Commits.
- Pre-commit hooks: ruff, mypy, pytest-quick.
- All commits authored as **Moslem Mohseni Khah**.

---

## 9. Reproducibility commitments

- Every result has a `run_manifest.json` artefact.
- Every number in the paper is grep-able to a JSON in `reports/`.
- A clean Windows VM + `reproduce_all.ps1` regenerates everything within ±0.5 absolute F1.
- A Zenodo DOI is minted at v1.0.0.
- `docs/REPRODUCING.md` contains: hardware prerequisites (24 GB GPU, CUDA 12.x), `git clone` + `pip install -e .` + `scripts/download_data.ps1` + `scripts/reproduce_all.ps1`, expected wallclock per phase, expected output filenames, and a troubleshooting matrix indexed by the failure modes documented in §10 / §11.2.

---

## 10. Risk register (pre-analyzed)

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | bitsandbytes Windows install fails | Medium | Critical | Phase-0 kill-switch: fall back to WSL2. |
| R2 | LogLLM upstream code not Windows-friendly | High | High | Phase-2A feasibility check; pivot to LogFiT if needed. |
| R3 | Qwen-2.5-1.5B underperforms baseline | Medium | High | Phase-3 kill-switch: scale to Phi-3.5-mini or larger LoRA rank. |
| R4 | Cross-system F1 collapse | Medium | High | Phase-4 kill-switch: add unsupervised target adaptation. |
| R5 | A1 ablation negative | Medium | Medium | Phase-6 kill-switch: re-frame paper around N3. |
| R6 | Scoop — competitor publishes the same idea | Medium | High | §11.1 quarterly scoop-check. |
| R7 | VRAM blow-up at long sequence | Medium | Medium | Phase-3 pre-budgeted fallback knobs (§4.1). |
| R8 | Calibration ECE > 0.05 on some folds | Medium | Low | Phase-5 fallback to Platt / vector scaling; honest report. |
| R9 | Loghub-2.0 mirror outage | Low | Medium | Local-mirror SHA-256 procedure. |
| R10 | License re-distribution conflict | Low | Medium | No raw-data re-distribution; runtime fetch only. |
| R11 | Concept drift in production deployment | Inevitable | Medium | §11.8 drift monitor + re-calibration path. |
| R12 | Dual-use / surveillance misuse | Low | High (reputational) | §11.8 model-card scoping; intended-use clause. |
| R13 | Compute overrun beyond ~500 GPU-hours | Medium | Medium | §4.2 seed-count reduction kill-switch. |

---

## 11. Governance routines

### 11.1 Scoop-check (quarterly during dev)
- arXiv search: `("log anomaly detection" OR "log-based anomaly") AND (LoRA OR QLoRA OR "small language model")` since the last scoop-check date.
- Google Scholar: same query.
- If a new paper substantially overlaps with N1–N4, an honest delta-statement is added to §1.3 and the related-work section.

### 11.2 bitsandbytes Windows install procedure (canonical)
- bitsandbytes ≥ 0.43.0 ships official Windows wheels.
- `pip install bitsandbytes>=0.43.0` from PyPI is the first attempt.
- If that fails, fallback to a tagged release wheel from the bitsandbytes-foundation GitHub releases page.
- If that also fails: WSL2 + Ubuntu 22.04 + same wheel. The reproduction script auto-detects WSL and adjusts paths.

### 11.3 Definition of "done" for a phase
A phase is **done** only when:
1. Every checklist item is ticked with evidence (CI log, MLflow run, JSON artefact).
2. A signed git tag is pushed.
3. A GitHub release draft is opened (even if not published).
4. The next phase's entry contract is verified.

### 11.4 AURC and Excess-AURC (formal definitions)
HyLog uses the formal definitions from the selective-classification literature (Geifman & El-Yaniv 2017; recent re-characterization in Traub et al. 2024 / ICML 2025):

- **AURC** = ∫₀¹ risk(c) dc, where risk(c) is the misclassification rate over the most-confident fraction `c` of predictions.
- **Excess-AURC (E-AURC)** = AURC − optimal AURC under an oracle confidence ranking. E-AURC isolates the *ranking* quality of the confidence score from the *base accuracy*, enabling fair cross-model comparison.

Both are computed by `src/hylog/calibration/aurc.py` and unit-tested against the closed-form values on synthetic data with known oracle rankings.

### 11.5 BERT pre-training contamination caveat
`bert-base-uncased` was pre-trained on Wikipedia + BookCorpus, neither of which contains operational log data from HDFS, BGL, Thunderbird, or OpenStack. Pre-training contamination is therefore extremely unlikely. We nonetheless run a **string-overlap audit** between the encoder's pre-training corpora (where checkpointed snapshots are available) and the LAD datasets, and report the audit in the paper appendix. This is cheap insurance against a reviewer asking the question.

### 11.6 Project naming
**HyLog** has been picked after a search confirmed no published software / paper / package conflict. If a future conflict is discovered before v1.0.0, an alias from {LogPrism, CrossLog, CalibroLog} is reserved as a fallback.

### 11.7 Inference output schema (authoritative)
Every `/predict` response carries this JSON contract:
```json
{
  "model_version": "hylog-v1.0.0+sha-xxxxxxx",
  "sequences": [
    {
      "id": "client-supplied-string",
      "p_anomaly": 0.0314,
      "p_anomaly_calibrated": 0.0271,
      "decision": "normal | anomaly | abstain",
      "abstain_reason": null,
      "confidence": 0.972
    }
  ],
  "calibration": {
    "method": "temperature_scaling",
    "T": 1.42,
    "fitted_on": "source_system_calibration_slice"
  }
}
```
A breaking schema change requires a major version bump and a deprecation window.

### 11.8 Ethics, dual-use, and concept drift
- **Dual-use disclosure.** Log anomaly detection can be repurposed for surveillance over user activity logs. HyLog's model card explicitly limits intended use to *system-operational* logs (kernel, scheduler, distributed storage) and discourages application to user-keyed personal logs without a separate ethics review.
- **Not a single point of decision.** The model card states that HyLog must not be the sole automated decision-maker in safety-critical incident response; it is a *recommender* with an abstention channel.
- **Concept drift.** Production deployments inevitably see log distributions drift away from training. HyLog ships with a **distribution-drift monitor** in `src/hylog/inference/drift.py` that tracks the empirical distribution of `p_anomaly_calibrated` and flags shifts above a configurable Kolmogorov-Smirnov threshold. Re-calibration (re-fitting T) is offered as a lightweight remedy; full re-training is documented as a heavier path.

### 11.9 Security checklist for the inference service (Phase 8)
- Input length cap (max lines per request, max bytes per line).
- Rate limiting per API key.
- No echoing of raw request content in error messages (avoid log-injection-by-error-message).
- Authentication via API key header, with key rotation supported.
- The service is **not** an LLM chat surface — there is no prompt-injection attack surface because the classification head replaces autoregressive generation.

---

*End of v0.4 — post round-3 critique. Ready for implementation.*
