"""Parser-free regex preprocessor for log lines.

The regex set is modeled after LogLLM (Guan et al., 2024;
https://arxiv.org/abs/2411.08561; https://github.com/guanwei49/LogLLM).
Each line is normalized by replacing volatile arguments (IPs, hex addresses,
numeric IDs, timestamps, paths) with stable placeholder tokens. The resulting
template-like form lets a frozen BERT encoder produce a stable semantic
embedding without invoking a heuristic log parser such as Drain.

The regex order matters: more specific patterns (timestamps, IPs) run before
generic numeric patterns to avoid premature masking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern


@dataclass(frozen=True)
class _Rule:
    """A single regex-replacement rule."""

    name: str
    pattern: Pattern[str]
    replacement: str


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern)


# LogLLM-style mask token vocabulary. Kept distinct so an ablation can collapse
# them into a single <NUM> placeholder if desired (Phase 6, ablation A?).
_MASK_IP = "<IP>"
_MASK_PORT = "<PORT>"
_MASK_HEX = "<HEX>"
_MASK_PATH = "<PATH>"
_MASK_URL = "<URL>"
_MASK_UUID = "<UUID>"
_MASK_BLOCK = "<BLK>"
_MASK_TIMESTAMP = "<TS>"
_MASK_DATE = "<DATE>"
_MASK_TIME = "<TIME>"
_MASK_NUM = "<NUM>"


# Order: most specific to most generic.
_DEFAULT_RULES: tuple[_Rule, ...] = (
    # ISO-like timestamps: 2005-06-03-15.42.50.675872 or 2005-06-03 15:42:50
    _Rule(
        "timestamp_iso",
        _compile(r"\b\d{4}[-/.]\d{2}[-/.]\d{2}[ T\-]\d{2}[:.]\d{2}[:.]\d{2}(?:[.,]\d+)?\b"),
        _MASK_TIMESTAMP,
    ),
    # Compact YYYYMMDD timestamps used by HDFS: 081109 203518
    _Rule(
        "timestamp_compact",
        _compile(r"\b\d{6}\s+\d{6}\b"),
        _MASK_TIMESTAMP,
    ),
    # Standalone dates and times.
    _Rule("date", _compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"), _MASK_DATE),
    _Rule("time", _compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), _MASK_TIME),
    # UUIDs.
    _Rule(
        "uuid",
        _compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        _MASK_UUID,
    ),
    # HDFS block IDs: blk_-1608999687919862906
    _Rule("hdfs_block", _compile(r"\bblk_-?\d+\b"), _MASK_BLOCK),
    # IPv4 with optional port: 10.250.19.102:54106
    _Rule(
        "ipv4_port",
        _compile(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}\b"),
        f"{_MASK_IP}:{_MASK_PORT}",
    ),
    _Rule("ipv4", _compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), _MASK_IP),
    # IPv6 (simplified — full RFC matching is intentionally avoided for speed).
    _Rule(
        "ipv6",
        _compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"),
        _MASK_IP,
    ),
    # URLs.
    _Rule(
        "url",
        _compile(r"\bhttps?://[^\s\"'>]+"),
        _MASK_URL,
    ),
    # Hex addresses (8+ hex digits, often with 0x).
    _Rule("hex_0x", _compile(r"\b0x[0-9a-fA-F]+\b"), _MASK_HEX),
    _Rule("hex_long", _compile(r"\b[0-9a-fA-F]{8,}\b"), _MASK_HEX),
    # Filesystem-like paths.
    _Rule(
        "unix_path",
        _compile(r"(?<![A-Za-z0-9])/(?:[\w\-.]+/)+[\w\-.]*"),
        _MASK_PATH,
    ),
    _Rule(
        "windows_path",
        _compile(r"\b[A-Za-z]:\\(?:[\w\-. ]+\\)*[\w\-. ]*"),
        _MASK_PATH,
    ),
    # Generic standalone numbers (run last so it does not eat timestamp parts).
    _Rule("number", _compile(r"\b\d+\b"), _MASK_NUM),
)


@dataclass(frozen=True)
class Preprocessor:
    """Apply a fixed sequence of regex masking rules to a log line.

    The default rule set mirrors the LogLLM preprocessor. The class is frozen
    and the rules are immutable so that two preprocessors constructed with the
    same arguments are guaranteed to produce byte-identical output, which is a
    prerequisite for the byte-identical split-manifest invariant tested in
    Phase 1.
    """

    rules: tuple[_Rule, ...] = field(default=_DEFAULT_RULES)
    lowercase: bool = False

    def __call__(self, line: str) -> str:
        return self.preprocess(line)

    def preprocess(self, line: str) -> str:
        """Normalize a single log line."""
        text = line.rstrip("\r\n")
        for rule in self.rules:
            text = rule.pattern.sub(rule.replacement, text)
        text = " ".join(text.split())
        if self.lowercase:
            text = text.lower()
        return text

    def preprocess_many(self, lines: list[str]) -> list[str]:
        return [self.preprocess(ln) for ln in lines]


def default_preprocessor() -> Preprocessor:
    """The canonical preprocessor used across the HyLog pipeline."""
    return Preprocessor()


__all__ = ["Preprocessor", "default_preprocessor"]
