"""Tests for the Wilcoxon hypothesis-test helpers."""

from __future__ import annotations

import pytest

from hylog.evaluation.statistical_tests import (
    holm_bonferroni,
    wilcoxon_one_sample,
    wilcoxon_paired,
)


def test_wilcoxon_one_sample_against_mean_with_no_difference() -> None:
    # All values equal expected -> p must be 1.0 (cannot reject).
    result = wilcoxon_one_sample([0.5, 0.5, 0.5], expected=0.5)
    assert result.p_value == pytest.approx(1.0)
    assert not result.rejected_at_alpha_0_05


def test_wilcoxon_one_sample_detects_clear_difference() -> None:
    result = wilcoxon_one_sample(
        [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97],
        expected=0.5,
    )
    assert result.p_value < 0.05
    assert result.rejected_at_alpha_0_05


def test_wilcoxon_paired_with_equal_arrays() -> None:
    result = wilcoxon_paired([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert result.p_value == pytest.approx(1.0)


def test_wilcoxon_paired_detects_systematic_offset() -> None:
    a = [0.90, 0.91, 0.92, 0.93, 0.94]
    b = [0.80, 0.81, 0.82, 0.83, 0.84]
    result = wilcoxon_paired(a, b)
    assert result.p_value < 0.10  # 5 samples, one-tailed feel ok at 10%


def test_wilcoxon_paired_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        wilcoxon_paired([1.0, 2.0], [1.0])


def test_holm_bonferroni_rejects_smallest() -> None:
    p_values = [0.001, 0.04, 0.06, 0.5]
    rejected = holm_bonferroni(p_values, alpha=0.05)
    # The smallest p is 0.001 vs alpha/4=0.0125 -> rejected.
    assert rejected[0] is True
    # 0.04 vs alpha/3 = 0.0166 -> NOT rejected (Holm stops here).
    assert rejected[1] is False
    assert rejected[2] is False
    assert rejected[3] is False


def test_holm_bonferroni_empty() -> None:
    assert holm_bonferroni([]) == []


def test_wilcoxon_alternative_greater_vs_less() -> None:
    values = [0.9, 0.91, 0.92, 0.93, 0.94]
    greater = wilcoxon_one_sample(values, expected=0.5, alternative="greater")
    less_result = wilcoxon_one_sample(values, expected=0.5, alternative="less")
    assert greater.p_value < 0.5
    assert less_result.p_value > 0.5
