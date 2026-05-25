"""Tests for the LogLLM-style preprocessor."""

from __future__ import annotations

from hylog.data.preprocess import Preprocessor, default_preprocessor


def test_masks_ipv4_with_port() -> None:
    pp = default_preprocessor()
    out = pp("dest: /10.250.19.102:50010 sending")
    assert "10.250.19.102:50010" not in out
    assert "<IP>:<PORT>" in out


def test_masks_hdfs_block() -> None:
    pp = default_preprocessor()
    out = pp("Receiving block blk_-1608999687919862906 src")
    assert "blk_-1608999687919862906" not in out
    assert "<BLK>" in out


def test_masks_timestamp_compact() -> None:
    pp = default_preprocessor()
    out = pp("081109 203518 143 INFO message")
    assert "081109 203518" not in out
    assert "<TS>" in out


def test_masks_iso_timestamp() -> None:
    pp = default_preprocessor()
    out = pp("2005-06-03 15:42:50 RAS KERNEL INFO")
    assert "<TS>" in out or ("<DATE>" in out and "<TIME>" in out)


def test_masks_hex_long() -> None:
    pp = default_preprocessor()
    out = pp("Address 0xDEADBEEF cafefacefeedbeef offset")
    # 0x… and the standalone long hex token both masked.
    assert "0xDEADBEEF" not in out
    assert "cafefacefeedbeef" not in out
    assert out.count("<HEX>") >= 1


def test_masks_url() -> None:
    pp = default_preprocessor()
    out = pp("Connection refused https://example.com/api/v1?x=1 retrying")
    assert "https://example.com" not in out
    assert "<URL>" in out


def test_collapses_whitespace() -> None:
    pp = default_preprocessor()
    out = pp("   leading   and    trailing   spaces   \n")
    assert out == "leading and trailing spaces"


def test_deterministic_two_runs() -> None:
    pp1 = default_preprocessor()
    pp2 = default_preprocessor()
    line = "081109 203518 143 INFO dfs.DataNode: block blk_42 src /10.0.0.1:5000"
    assert pp1(line) == pp2(line)


def test_preprocess_many_preserves_order() -> None:
    pp = default_preprocessor()
    lines = ["a 1", "b 2", "c 3"]
    out = pp.preprocess_many(lines)
    assert out == ["a <NUM>", "b <NUM>", "c <NUM>"]


def test_lowercase_option() -> None:
    pp = Preprocessor(lowercase=True)
    out = pp("INFO MESSAGE")
    assert out == "info message"
