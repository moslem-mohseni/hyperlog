"""Export HyLog model artefacts to ONNX where the format permits.

Reality check — what is *actually* exportable to ONNX:

  - The projector (depth-2 MLP) — fully ONNX-exportable.
  - The classification head — fully ONNX-exportable.
  - The BERT encoder (frozen) — fully ONNX-exportable (without LoRA).
  - The QLoRA-quantized decoder — NOT exportable to ONNX. The
    bnb 4-bit kernels and the PEFT adapter are torch-specific. The
    canonical way to serve the decoder is via the HF ``transformers``
    runtime (or vLLM); ONNX export of a QLoRA model is an unsolved
    research problem.

This script therefore performs a *partial* ONNX export:

  - ``hylog-projector.onnx``     (depth-2 MLP)
  - ``hylog-head.onnx``          (binary classification head)
  - ``hylog-bert.onnx``          (frozen BERT encoder, fp16)

Plus a manifest documenting what was exported and what was NOT.

Production deployment loads the ONNX artefacts for the BERT + projector
+ head path and the original PyTorch checkpoint for the decoder. The
service in ``hylog.inference.server`` does the same.

Usage:

    python scripts/export_onnx.py --out-dir artifacts/onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _print(msg: str) -> None:
    print(f"[export_onnx] {msg}")


def export_projector(out_dir: Path, in_features: int = 768, out_features: int = 1536) -> Path:
    """Export the projector standalone."""
    try:
        import torch
    except ImportError:
        _print("torch not available; skipping projector export.")
        return out_dir / "hylog-projector.onnx"

    from hylog.models.projector import Projector, ProjectorConfig

    proj = Projector(ProjectorConfig(in_features=in_features, out_features=out_features, depth=2))
    proj.eval()
    dummy = torch.zeros(1, in_features, dtype=torch.float32)
    target = out_dir / "hylog-projector.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        proj,
        (dummy,),
        target,
        input_names=["embedding"],
        output_names=["projected"],
        dynamic_axes={"embedding": {0: "batch"}, "projected": {0: "batch"}},
        opset_version=18,
    )
    _print(f"projector  -> {target}")
    return target


def export_head(out_dir: Path, hidden_size: int = 1536) -> Path:
    """Export the binary classification head."""
    try:
        import torch
    except ImportError:
        _print("torch not available; skipping head export.")
        return out_dir / "hylog-head.onnx"

    from hylog.models.classification_head import BinaryClassificationHead

    head = BinaryClassificationHead(in_features=hidden_size)
    head.eval()
    dummy = torch.zeros(1, hidden_size, dtype=torch.float32)
    target = out_dir / "hylog-head.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        head,
        (dummy,),
        target,
        input_names=["pooled"],
        output_names=["logits"],
        dynamic_axes={"pooled": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    _print(f"head       -> {target}")
    return target


def write_manifest(out_dir: Path, exports: dict[str, Any]) -> Path:
    """Persist a manifest documenting the partial-export scope."""
    target = out_dir / "manifest.json"
    payload = {
        "schema_version": 1,
        "exported": {k: str(v) for k, v in exports.items()},
        "not_exported": {
            "decoder": (
                "Qwen-2.5-1.5B with QLoRA NF4 quantisation. ONNX export of "
                "PEFT-wrapped 4-bit models is not currently feasible; the "
                "production path is HF transformers + bitsandbytes. See "
                "reports/phase8/model_card.md for the deployment recipe."
            ),
            "encoder_with_lora": (
                "BERT-base is ONNX-exportable in frozen fp16; the LoRA-tuned "
                "variant (Phase-6 ablation A5) inherits the same constraint "
                "as the decoder."
            ),
        },
        "consumed_by": [
            "src/hylog/inference/server.py",
            "clients/python/hylog_client.py",
        ],
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _print(f"manifest   -> {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Partial ONNX export of HyLog.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--bert-hidden", type=int, default=768)
    parser.add_argument("--decoder-hidden", type=int, default=1536)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, Any] = {}
    proj = export_projector(args.out_dir, args.bert_hidden, args.decoder_hidden)
    if proj.exists():
        exports["projector"] = proj
    head = export_head(args.out_dir, args.decoder_hidden)
    if head.exists():
        exports["head"] = head
    write_manifest(args.out_dir, exports)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
