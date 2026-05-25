"""Tests for the API-key auth module."""

from __future__ import annotations

from hylog.inference.auth import (
    APIKeyStore,
    constant_time_match,
    hash_key,
)


def test_hash_key_is_deterministic_hex() -> None:
    h = hash_key("secret-1")
    assert isinstance(h, str)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
    assert h == hash_key("secret-1")
    assert h != hash_key("secret-2")


def test_constant_time_match_handles_unicode() -> None:
    assert constant_time_match("abc", "abc")
    assert not constant_time_match("abc", "abd")
    # Unicode safe.
    assert constant_time_match("سلام", "سلام")
    assert not constant_time_match("سلام", "hi")


def test_apikeystore_from_plaintext_round_trip() -> None:
    store = APIKeyStore.from_plaintext({"alice-key": "alice", "bob-key": "bob"})
    assert store.authenticate("alice-key") == "alice"
    assert store.authenticate("bob-key") == "bob"
    assert store.authenticate("eve-key") is None


def test_apikeystore_from_env_parses_pairs(monkeypatch) -> None:
    h = hash_key("k1")
    monkeypatch.setenv("HYLOG_API_KEYS", f"alice:{h}")
    store = APIKeyStore.from_env()
    assert store.authenticate("k1") == "alice"


def test_apikeystore_empty_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("HYLOG_API_KEYS", raising=False)
    store = APIKeyStore.from_env()
    assert store.empty()


def test_apikeystore_ignores_malformed_env_entries(monkeypatch) -> None:
    monkeypatch.setenv("HYLOG_API_KEYS", "alice,bob:," + f"carol:{hash_key('k')}")
    store = APIKeyStore.from_env()
    assert store.authenticate("k") == "carol"
