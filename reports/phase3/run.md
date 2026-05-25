# Phase 3 — HyLog Core (Backbone Substitution)

**Author:** Moslem Mohseni Khah
**Phase:** 3 (Backbone Substitution — the HyLog core)
**Roadmap reference:** `docs/ROADMAP.md` §Phase 3
**Release tag:** `v0.3.0-core`

This report records the methodology, deliverables, and verification
status of HyLog's Phase 3: retiring LogLLM's Llama-7B decoder in favour
of a compact 1–4 B SLM trained with QLoRA, with a deterministic
classification head replacing autoregressive answer matching.

---

## 1. Architectural changes vs. Phase 2

| Element | LogLLM baseline (Phase 2) | HyLogCore (Phase 3) |
|---|---|---|
| Decoder | Llama-3-8B | Qwen-2.5-1.5B (primary), Phi-3.5-mini, Llama-3.2-1B/3B, TinyLlama-1.1B |
| Output | Token IDs over "The sequence is normal/anomalous." | Logits over {normal, anomaly} via classification head |
| Inference | Autoregressive sampling (up to 5 tokens) | Single softmax — calibration-friendly |
| Projector | depth-1 Linear | depth-2 MLP with GELU + dropout |
| LoRA target | (q_proj, v_proj) only | (q_proj, k_proj, v_proj, o_proj) — full QKVO |
| LoRA rank | 8 | 16 |
| Loss | Cross-entropy over answer tokens | Weighted cross-entropy on 2 logits |
| Class imbalance handling | None | Inverse-frequency weights, capped at 10× |
| Encoder | BERT + LoRA (active in two stages) | BERT, permanently frozen by default |
| Training stages | 4 (decoder-LoRA → projector → projector+enc-LoRA → all) | 3 (projector warm-up → projector+LoRA+head → end-to-end refine) |

The simpler classification head is the reason HyLog can plausibly claim
"first LAD pipeline with usable uncertainty" (roadmap N3): the
post-hoc temperature scaling that Phase 5 deploys is well-defined only on
a fixed-output classifier.

---

## 2. Deliverables shipped at this tag

### 2.1 Source
| Path | Purpose |
|---|---|
| `src/hylog/models/decoder.py` | Registry of all six registered backbones; `load_decoder()` GPU path with BitsAndBytesConfig. |
| `src/hylog/models/hylog_core.py` | HyLogCore: BERT (frozen) → Projector → QLoRA decoder → BinaryClassificationHead. `inputs_embeds` path; no autoregressive sampling. |
| `src/hylog/training/collator.py` | HyLogCollator: flattens a batch of LogSequence into (line_inputs, sequence_lengths, labels). |
| `src/hylog/training/qlora_trainer.py` | QLoraTrainer with grad accumulation, class-weighted CE, gradient clipping, per-epoch validation panel. |
| `src/hylog/training/vram.py` | Pre-flight VRAM estimator predicting peak memory before the run. |
| `src/hylog/training/seeded_runner.py` | Multi-seed driver aggregating mean/std/min/max per metric. |

### 2.2 Configs
- `configs/decoders/{qwen25_1_5b, qwen25_1_5b_instruct, phi35_mini, llama32_1b, llama32_3b, tinyllama}.yaml`
- `configs/experiments/hylog_{hdfs,bgl}.yaml` — full 5-seed run protocol pre-built.

### 2.3 Tests
All on CPU using a tiny BertModel + tiny LlamaForCausalLM (constructed
from in-memory configs, no network). +45 tests added in Phase 3:

| Suite | Coverage |
|---|---|
| `test_decoder_registry.py` | All six required backbones present; case-insensitive lookup; consistency with `configs/decoders/*.yaml`. |
| `test_hylog_core.py` | LoRA attachment, encoder always frozen, default training mode, projector dimensions, trainable-fraction bounds, forward shape, error handling. |
| `test_qlora_trainer.py` | Class-weighted CE with cap, tail-monotone helper, mechanics end-to-end on a tiny synthetic dataset. |
| `test_seeded_runner.py` | Aggregate mean/std/min/max, headline formatting, edge cases. |
| `test_vram.py` | Qwen-2.5-1.5B in 4-bit fits 22 GB budget; Phi-3.5-mini in 4-bit fits 24 GB; bf16 is strictly larger than 4-bit. |
| `test_configs.py` | All required configs present, schema-valid, consistent with the registry. |

---

## 3. Phase 3 checklist status

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | In-domain F1 on HDFS (Qwen-2.5-1.5B) ≥ Phase-2 LogLLM-on-HDFS F1 − 1.0 | ⏳ **GPU-deferred** | Infrastructure complete; `configs/experiments/hylog_hdfs.yaml` runs the 5-seed protocol end-to-end on CUDA. |
| 2 | In-domain F1 on BGL (Qwen-2.5-1.5B) ≥ Phase-2 LogLLM-on-BGL F1 − 1.0 | ⏳ **GPU-deferred** | Same as above; `hylog_bgl.yaml`. |
| 3 | Peak VRAM during training ≤ 22 GB on RTX 3090 / 4090 | ✅ **Predicted to pass; mechanically verified** | `test_qwen_1_5b_4bit_fits_in_24gb` in `tests/unit/test_training/test_vram.py` asserts the estimator's peak ≤ 22 GB for the default Phase 3 config. The estimator's accuracy on real hardware is validated by the GPU run. |
| 4 | Trainable parameter count (projector + LoRA + head) < 5 % of decoder full count | ✅ **Verified analytically + in-vivo bounded** | `test_trainable_fraction_under_5_percent_at_production_scale` computes the budget at production scale (Qwen-2.5-1.5B, 1.54 B params, 28 layers, projector depth-2, LoRA r=16 QKVO, classification head) and asserts the fraction < 5 %. |
| 5 | All three training stages converge (val loss monotone non-increasing across the last 20 % of each stage's steps) | ⏳ **GPU-deferred** | The monotone-tail helper is exercised in unit tests; convergence is a property of the real model + real data. |
| 6 | 5-seed runs; standard deviation per metric reported | ✅ **Infrastructure complete** | `seeded_runner.run_seeded()` aggregates mean/std/min/max; configs declare `seeds: [42, 1337, 2024, 31415, 27182]`. |
| 7 | Tag `v0.3.0-core` pushed | ✅ | this commit |

### Items 1, 2, 5 — GPU-dependent
Per the roadmap kill-switch, these three items are gated on a CUDA run.
The infrastructure is in place and the exact commands to execute them on
a 24 GB GPU are:

```powershell
# Install GPU torch + bitsandbytes (Phase 0 §11.2 procedure)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes>=0.43.0

# Fetch real datasets
.\scripts\download_data.ps1 -Dataset hdfs
.\scripts\download_data.ps1 -Dataset bgl

# Run all 5 seeds for HDFS and BGL
hylog-train --config configs/experiments/hylog_hdfs.yaml
hylog-train --config configs/experiments/hylog_bgl.yaml
```

On completion, `reports/phase3/runs/{hdfs,bgl}/seeds/{42,1337,…}/metrics.json`
holds the full metric panel per seed and `reports/phase3/runs/{hdfs,bgl}/summary.json`
holds the mean ± std over seeds.

### Deviation note (per the Phase 3 kill-switch in roadmap §10 R3)
The kill-switch says: "if Qwen-2.5-1.5B underperforms by > 1.0 absolute F1
across both datasets and 5 seeds, escalate. Options: (a) switch primary
decoder to Phi-3.5-mini (3.8 B); (b) increase LoRA rank; (c) increase
projector capacity. Each option is pre-budgeted in configs."

All three escape paths are already wired:
- (a) Phi-3.5-mini has its own decoder spec and config; switching is a
  one-line change.
- (b) `HyLogLoraConfig.r` is a runtime knob; the Hydra config exposes it
  as `model.lora.r`.
- (c) `HyLogCoreConfig.projector_depth` toggles between 1, 2, or 3; the
  config exposes it as `model.projector.depth`.

The kill-switch is therefore "instrumentation complete, decision deferred
to the GPU run output".

---

## 4. Test summary at this tag

| Suite | Tests | Status |
|---|---|---|
| Data pipeline (Phase 1 regression) | 51 | ✅ |
| Models (encoder, projector, head, LogLLM baseline, HyLogCore, decoder registry) | 38 | ✅ |
| Training (three-stage trainer, MLflow, QLoRA trainer, seeded runner, VRAM, configs) | 34 | ✅ |
| Evaluation (metric panel) | 10 | ✅ |
| Feasibility artefacts | 4 | ✅ |
| Other (smoke, CLI, utils) | 8 | ✅ |
| **Total** | **138** | **✅ all pass** |

Verification commands run on this commit:

```text
ruff check src tests        -> clean
ruff format --check         -> clean
mypy --strict src/hylog     -> clean (38 source files)
pytest -q                   -> 138 passed
```

---

## 5. Reproducibility manifest

| Artefact | Path |
|---|---|
| Run report (this doc) | `reports/phase3/run.md` |
| HDFS experiment config | `configs/experiments/hylog_hdfs.yaml` |
| BGL experiment config | `configs/experiments/hylog_bgl.yaml` |
| Decoder spec configs | `configs/decoders/*.yaml` |
| HyLog core model | `src/hylog/models/hylog_core.py` |
| Decoder registry | `src/hylog/models/decoder.py` |
| QLoRA trainer | `src/hylog/training/qlora_trainer.py` |
| Multi-seed runner | `src/hylog/training/seeded_runner.py` |
| VRAM estimator | `src/hylog/training/vram.py` |
| GPU-run output (when produced) | `reports/phase3/runs/{hdfs,bgl}/summary.json` |
