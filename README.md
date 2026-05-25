# HyLog

**Hybrid Small-Language-Model Pipeline for Cross-System Log Anomaly Detection with Calibrated Uncertainty**

[![CI](https://github.com/moslem-mohseni/hyperlog/actions/workflows/ci.yml/badge.svg)](https://github.com/moslem-mohseni/hyperlog/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> Author: **Moslem Mohseni Khah** · Status: **pre-alpha (Phase 0 — scaffolding)**

HyLog detects anomalies in system logs across heterogeneous software systems
(HDFS, Blue Gene/L, Thunderbird, OpenStack, …) using a three-stage hybrid
encoder-decoder pipeline. A frozen BERT encoder produces semantic embeddings of
individual log lines; a learned projector aligns them into the input space of a
**compact decoder-only small language model in the 1–4 B parameter band** (Qwen-2.5-1.5B
primary, Phi-3.5-mini-instruct secondary); the decoder is parameter-efficiently
fine-tuned with **QLoRA (4-bit NF4)**. The pipeline is evaluated under a strict
**leave-one-system-out** cross-system protocol with **zero target labels** and
shipped with **post-hoc temperature-scaling calibration** plus **risk-coverage
selective prediction**.

## Project status

This repository is currently in **Phase 0 — Foundation & Environment**. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the full 10-phase plan, related work,
risk register, and success contract.

## Quick start (development)

> Requires Python 3.11+, an NVIDIA GPU with at least 24 GB VRAM, and CUDA 12.x.
> Windows is the primary development platform; Linux works identically.

```powershell
# Clone
git clone https://github.com/moslem-mohseni/hyperlog.git
cd hyperlog

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Verify
ruff check src tests
mypy src
pytest
```

## Documentation

- **[Roadmap](docs/ROADMAP.md)** — phases, deliverables, tests, success contract.
- **`docs/00_initial_materials/`** — the original literature review and research
  proposal that motivated this project (preserved verbatim, separate from the
  source tree).
- **`docs/ARCHITECTURE.md`** *(Phase 7 deliverable)*
- **`docs/REPRODUCING.md`** *(Phase 7 deliverable)*

## Repository layout

```
.
├── docs/
│   ├── ROADMAP.md
│   └── 00_initial_materials/        # literature review + proposal (frozen)
├── src/hylog/
│   ├── data/                        # ingestion, preprocessing, splits
│   ├── models/                      # encoder, projector, decoder, baselines
│   ├── training/                    # three-stage QLoRA trainer
│   ├── evaluation/                  # metrics panel, LOSO protocol
│   ├── calibration/                 # temperature scaling, ECE, AURC
│   ├── inference/                   # FastAPI service, selective predictor
│   ├── cli/                         # hylog-train, hylog-predict
│   └── utils/                       # seeding, manifests, logging
├── tests/                           # mirrors src/
├── configs/                         # Hydra configs
├── reports/                         # per-phase result artefacts
├── scripts/                         # download_data, reproduce_all
├── paper/                           # LaTeX manuscript (Phase 9)
└── pyproject.toml
```

## Citing

If you use HyLog in academic work, please cite it via the metadata in
[`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) © 2026 Moslem Mohseni Khah
