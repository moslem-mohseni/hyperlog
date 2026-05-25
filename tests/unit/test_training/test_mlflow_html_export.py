"""Tests for the MLflow HTML exporter."""

from __future__ import annotations

import json
from pathlib import Path

from hylog.training.mlflow_html_export import (
    RunRecord,
    export_html,
    render_html,
    runs_to_json,
)


def _sample_runs() -> list[RunRecord]:
    return [
        RunRecord(
            run_id="abcd1234",
            experiment_name="phase4-loso-core",
            start_time="2026-05-26T10:00:00Z",
            end_time="2026-05-26T10:30:00Z",
            status="FINISHED",
            params={"lr": "5e-5", "batch_size": "16"},
            metrics={"f1": 0.91, "auc_roc": 0.93},
            tags={"seed": "42"},
        ),
        RunRecord(
            run_id="efgh5678",
            experiment_name="phase4-loso-core",
            start_time="2026-05-26T11:00:00Z",
            end_time="",
            status="RUNNING",
            params={"lr": "1e-4"},
            metrics={},
            tags={},
        ),
        RunRecord(
            run_id="ijkl9012",
            experiment_name="phase5-calibration",
            start_time="2026-05-26T09:00:00Z",
            end_time="2026-05-26T09:05:00Z",
            status="FINISHED",
            params={},
            metrics={"ece": 0.04},
            tags={"method": "temperature_scaling"},
        ),
    ]


def test_render_html_groups_by_experiment() -> None:
    html = render_html(title="t1", runs=_sample_runs())
    assert "phase4-loso-core" in html
    assert "phase5-calibration" in html
    # Group counts.
    assert "(2 runs)" in html or "(2 runs" in html
    assert "(1 runs)" in html or "(1 runs" in html


def test_render_html_escapes_special_chars() -> None:
    runs = [
        RunRecord(
            run_id="xyz",
            experiment_name="<bad>name",
            start_time="t",
            end_time="t",
            status="OK",
            params={"k": "<v>"},
            metrics={},
            tags={},
        )
    ]
    html = render_html(title="<title>", runs=runs)
    assert "&lt;bad&gt;name" in html
    assert "&lt;title&gt;" in html


def test_export_html_round_trip(tmp_path: Path) -> None:
    out = export_html(out_path=tmp_path / "report.html", runs=_sample_runs(), title="x")
    body = out.read_text(encoding="utf-8")
    assert body.startswith("<!DOCTYPE html>")
    assert body.strip().endswith("</html>")


def test_runs_to_json_is_deterministic() -> None:
    a = runs_to_json(_sample_runs())
    b = runs_to_json(_sample_runs())
    assert a == b
    payload = json.loads(a)
    assert len(payload) == 3


def test_empty_runs_renders_message() -> None:
    html = render_html(title="empty", runs=[])
    assert "0 run(s)" in html


def test_kv_collapsing_for_many_params() -> None:
    """When a run has > 4 params, the renderer collapses them into a
    <details> element so the table stays readable."""
    runs = [
        RunRecord(
            run_id="abc",
            experiment_name="e",
            start_time="t",
            end_time="t",
            status="OK",
            params={f"k{i}": str(i) for i in range(10)},
            metrics={},
            tags={},
        )
    ]
    html = render_html(title="t", runs=runs)
    assert "<details>" in html
    assert "more" in html
