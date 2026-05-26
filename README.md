# HyperLog

**Hybrid Small-Language-Model Pipeline for Cross-System Log Anomaly Detection with Calibrated Uncertainty**

[![CI](https://github.com/moslem-mohseni/hyperlog/actions/workflows/ci.yml/badge.svg)](https://github.com/moslem-mohseni/hyperlog/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v1.0.0-green.svg)](https://github.com/moslem-mohseni/hyperlog/releases)

> Author: **Moslem Mohseni Khah** — Status: **v1.0.0 released**

HyperLog detects anomalies in system logs across heterogeneous software
systems using a three-stage hybrid encoder-decoder pipeline. A frozen
BERT encoder produces semantic embeddings of individual log lines; a
learned projector aligns them into the input space of a compact
decoder-only small language model (Qwen-2.5-1.5B primary,
Phi-3.5-mini-instruct secondary); the decoder is parameter-efficiently
fine-tuned with QLoRA (4-bit NF4). The pipeline is evaluated under a
strict leave-one-system-out cross-system protocol with zero target
labels and ships with post-hoc temperature-scaling calibration plus
risk-coverage selective prediction.

## What v1.0.0 ships

- **Code:** 70 source files; 420 unit + integration tests passing.
- **CLIs:** `hylog-train`, `hylog-predict`, `hylog-loso`, `hylog-calibrate`, `hylog-ablation`.
- **Inference service:** FastAPI REST (`/v1/predict`, `/v1/drift`, `/v1/model-info`, `/healthz`) with API-key auth, per-key rate limiting, drift monitor, OpenAPI spec.
- **Reproducibility:** one-command `reproduce_all.{ps1,sh}`, conda + pip lockfiles, Dockerfile, GPU CI workflow.
- **Paper:** LaTeX manuscript in `paper/`, auto-regenerated figures, full reproducibility appendix.
- **Model card:** HF-template `reports/phase8/model_card.md` with intended use, dual-use disclosure, drift guidance.

## Quick start (development)

> Requires Python 3.11+, optional NVIDIA GPU for training.
> Windows is the primary development platform; Linux works identically.

```powershell
# Clone
git clone https://github.com/moslem-mohseni/hyperlog.git
cd hyperlog

# Install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e ".[dev]"

# One-command verification (no GPU required; ~95 s on a fresh machine)
.\scripts\reproduce_all.ps1
```

The reproduce-all script runs `verify_install` → `ruff` →
`ruff format --check` → `mypy` → `pytest` (420 tests) → mock
`hylog-loso` → `hylog-calibrate` → `hylog-ablation`. On exit 0 you have
a fully-verified development environment.

## Reproducing the paper

Every numerical claim in `paper/main.tex` is backed by a JSON artefact
under `reports/`. The complete mapping is in
[`reports/phase9/reproducibility_appendix.md`](reports/phase9/reproducibility_appendix.md).

**CPU-only (claims that do not depend on GPU runs):**

```powershell
.\scripts\reproduce_all.ps1     # passes 420 tests
.\scripts\build_paper.ps1       # paper/main.pdf
```

**GPU run (24 GB VRAM):**

```powershell
pip install -r requirements-lock.txt

# Phase 2: faithful LogLLM reproduction.
hylog-train --config configs/baselines/logllm_hdfs.yaml
hylog-train --config configs/baselines/logllm_bgl.yaml

# Phase 3: HyperLog core in-domain.
hylog-train --config configs/experiments/hylog_hdfs.yaml
hylog-train --config configs/experiments/hylog_bgl.yaml

# Phase 4: cross-system LOSO.
hylog-loso --config configs/experiments/loso_hdfs_held.yaml
hylog-loso --config configs/experiments/loso_bgl_held.yaml
hylog-loso --config configs/experiments/loso_thunderbird_held.yaml
hylog-loso --config configs/experiments/loso_openstack_held.yaml

# Phase 5: calibration.
hylog-calibrate --predictions reports/phase4/runs/<run>/<fold>/predictions.jsonl `
                --out-dir reports/phase5/runs/<fold>

# Phase 6: full ablation matrix.
hylog-ablation --all-axes configs/ablation --out-dir reports/phase6/runs

# Rebuild the paper with filled-in numbers.
.\scripts\build_paper.ps1
```

Estimated total compute: **~500 GPU-hours** on a 24 GB GPU.

## Inference service

```powershell
# Start the service with a mock predictor (no GPU needed):
python -c "from hylog.inference.server import *; from hylog.inference.auth import *; import uvicorn; uvicorn.run(create_app(ServerConfig(api_key_store=APIKeyStore.from_plaintext({'demo':'demo'}))), port=8000)"
```

```bash
curl -H "X-API-Key: demo" \
     -H "Content-Type: application/json" \
     -d '{"sequences":[{"id":"r1","lines":["FATAL kernel panic"]}]}' \
     http://127.0.0.1:8000/v1/predict
```

Production deployment uses the included Dockerfile (CUDA 12.1, Python 3.11):

```bash
docker build -t hyperlog:v1.0.0 .
docker run --gpus all -p 8000:8000 hyperlog:v1.0.0
```

## Repository layout

```
.
├── docs/
│   ├── ROADMAP.md                       # the 10-phase plan
│   └── 00_initial_materials/            # original literature review + proposal
├── src/hylog/
│   ├── data/                            # ingestion, preprocessing, splits
│   ├── models/                          # encoder, projector, decoder registry, HyLogCore, LogLLM baseline, standalone-decoder baseline
│   ├── training/                        # 3-stage QLoRA trainer, seeded runner, VRAM estimator, MLflow logger + HTML exporter
│   ├── evaluation/                      # metric panel, bootstrap, Wilcoxon, Holm-Bonferroni, Cliff's δ, LOSO orchestrator, ablation matrix, OOD distance, leakage audit, run manifest
│   ├── calibration/                     # temperature / Platt / vector scaling, ECE, reliability, AURC
│   ├── inference/                       # FastAPI server, schemas, auth, rate limit, drift monitor, selective predictor
│   ├── cli/                             # hylog-train / hylog-predict / hylog-loso / hylog-calibrate / hylog-ablation
│   └── utils/
├── tests/                               # 420 unit + integration tests
├── configs/                             # Hydra configs (decoders, experiments, ablation, baselines)
├── reports/                             # JSON artefacts per phase; LaTeX cites these directly
├── scripts/                             # download_data, reproduce_all, verify_install, build_figures, build_paper, ...
├── paper/                               # IEEE-TNSM LaTeX manuscript + figures + references
├── clients/python/                      # Python SDK example
├── .github/workflows/                   # CI matrix (CPU) + GPU smoke
├── Dockerfile                           # CUDA 12.1 production container
├── environment.yml                      # conda env
├── requirements-lock.txt                # pip lock
├── pyproject.toml
├── README.md
├── LICENSE                              # MIT
└── CITATION.cff
```

## Documentation

- **[Roadmap](docs/ROADMAP.md)** — the 10-phase plan, deliverables, tests, risk register.
- **[Reproducibility appendix](reports/phase9/reproducibility_appendix.md)** — every paper claim → JSON file.
- **[Model card](reports/phase8/model_card.md)** — HF-template card with intended use, dual-use disclosure, drift guidance.
- **[LOSO protocol](reports/phase4/loso_protocol.md)** — reviewer-facing cross-system protocol spec.
- Per-phase reports: `reports/phase{2..9}/*.md`.

## Citing

Cite via the [CITATION.cff](CITATION.cff) metadata or the Zenodo DOI
(minted at v1.0.0):

```bibtex
@software{mohsenikhah_hyperlog_2026,
  title   = {HyperLog: Hybrid Small-Language-Model Pipeline for
             Cross-System Log Anomaly Detection with Calibrated
             Uncertainty},
  author  = {Mohseni Khah, Moslem},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/moslem-mohseni/hyperlog},
}
```

## License

[MIT](LICENSE) © 2026 Moslem Mohseni Khah
