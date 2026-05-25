"""API-key authentication for the HyLog inference service.

Phase-8 §11.9 security checklist: every request to ``/v1/*`` must carry
a valid API key in the ``X-API-Key`` header. Keys are stored as
SHA-256 digests so the on-disk artefact does not contain plaintext
secrets.

The auth surface is deliberately small: there is no user model, no
session state, no JWT. One key = one client. Rate limiting (see
``rate_limit.py``) is keyed on the same identifier.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

HEADER_NAME = "X-API-Key"


@dataclass(frozen=True, slots=True)
class APIKey:
    """One registered key (identified by its hashed prefix)."""

    name: str
    hashed: str
    """Lower-case hex SHA-256 of the plaintext key. Stored on disk."""


def hash_key(plaintext: str) -> str:
    """Return the SHA-256 hex digest used as the on-disk identifier."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def constant_time_match(a: str, b: str) -> bool:
    """``hmac.compare_digest`` wrapper that handles unicode and length."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


@dataclass(slots=True)
class APIKeyStore:
    """Read-only registry of accepted API keys.

    The store is a frozen mapping from the SHA-256 hash to a human name.
    Construct from a YAML or env var; never accept plaintext keys at
    runtime.
    """

    keys: Mapping[str, str]
    """``{hashed: human_name}``."""

    @classmethod
    def from_env(cls, env_var: str = "HYLOG_API_KEYS") -> APIKeyStore:
        """Parse ``name1:hash1,name2:hash2,...`` from an env var.

        Example:
            HYLOG_API_KEYS=alice:abc123...,bob:def456...
        """
        raw = os.environ.get(env_var, "")
        keys: dict[str, str] = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                continue
            name, hashed = entry.split(":", 1)
            keys[hashed.strip().lower()] = name.strip()
        return cls(keys=keys)

    @classmethod
    def from_plaintext(cls, plaintext_map: Mapping[str, str]) -> APIKeyStore:
        """Hash every plaintext key. Used in tests and the dev server."""
        return cls(keys={hash_key(k): v for k, v in plaintext_map.items()})

    def authenticate(self, plaintext: str) -> str | None:
        """Return the name of the matching key, or ``None`` on no match.

        Iterates with constant-time comparison so a timing attack cannot
        reveal *which* key was supplied.
        """
        target = hash_key(plaintext)
        for hashed, name in self.keys.items():
            if constant_time_match(hashed, target):
                return name
        return None

    def empty(self) -> bool:
        return not self.keys


# FastAPI dependency. The store is injected via ``app.state.api_key_store``
# at startup so tests can override it.


def require_api_key(
    request_header: str | None = Header(default=None, alias=HEADER_NAME),
) -> str:
    """FastAPI dependency that validates ``X-API-Key``.

    Returns the client's display name on success. Raises 401 on missing
    or invalid keys. The error message is intentionally vague (Phase-8
    §11.9: no-echo errors).
    """
    from fastapi import Request

    def _dep(
        request: Request,
        header: str | None = Header(default=None, alias=HEADER_NAME),
    ) -> str:
        store: APIKeyStore | None = getattr(request.app.state, "api_key_store", None)
        if store is None or store.empty():
            # Auth not configured -> deny by default.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="server has no API keys configured",
            )
        if not header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid API key",
                headers={"WWW-Authenticate": HEADER_NAME},
            )
        client = store.authenticate(header)
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid API key",
                headers={"WWW-Authenticate": HEADER_NAME},
            )
        return client

    return _dep  # type: ignore[return-value]


__all__ = [
    "HEADER_NAME",
    "APIKey",
    "APIKeyStore",
    "constant_time_match",
    "hash_key",
    "require_api_key",
]
