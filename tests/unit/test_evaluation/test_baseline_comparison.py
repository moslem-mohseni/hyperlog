"""Tests for the published-baselines registry."""

from __future__ import annotations

from pathlib import Path

from hylog.evaluation.baseline_comparison import (
    DEFAULT_PUBLISHED_PATH,
    load_published_baselines,
    render_macro_comparison,
    render_per_fold_comparison,
)


def test_published_yaml_loads() -> None:
    baselines = load_published_baselines()
    assert baselines, "expected at least one published baseline"
    methods = {b.method for b in baselines}
    # Roadmap §2.4 lists at least these:
    assert "MetaLog" in methods
    assert "ZeroLog" in methods


def test_each_baseline_has_required_fields() -> None:
    for b in load_published_baselines():
        assert b.method
        assert b.paper_link.startswith(("http://", "https://"))
        assert b.year >= 2017
        assert b.protocol


def test_render_macro_table_includes_hylog_row() -> None:
    table = render_macro_comparison(
        hylog_macro_f1=0.85,
        baselines=load_published_baselines(),
    )
    assert "HyLog" in table
    assert "MetaLog" in table
    assert "85.00" in table  # hylog macro F1 in percent


def test_render_macro_table_with_unknown_hylog_value() -> None:
    table = render_macro_comparison(
        hylog_macro_f1=None,
        baselines=load_published_baselines(),
    )
    # NaN HyLog row should render as the placeholder.
    assert "—" in table or "—" in table


def test_render_per_fold_table_columns() -> None:
    table = render_per_fold_comparison(
        hylog_per_fold={"HDFS": 0.99, "BGL": 0.96, "Thunderbird": 0.95, "OpenStack": None},
        baselines=load_published_baselines(),
    )
    for fold in ("HDFS", "BGL", "Thunderbird", "OpenStack"):
        assert fold in table
    assert "HyLog" in table


def test_default_path_resolves_to_repo() -> None:
    assert isinstance(DEFAULT_PUBLISHED_PATH, Path)
    assert DEFAULT_PUBLISHED_PATH.exists()
