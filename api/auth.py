"""Optional accounts — real signup/login with hashed passwords and signed bearer tokens.

Stdlib only (no external auth service, no extra dependency): PBKDF2-HMAC-SHA256 for
passwords and an HMAC-signed, expiring token for sessions. Accounts are **optional** —
the anonymous persona demo still works unchanged. Set ``CP_REQUIRE_AUTH=1`` to require a
valid bearer token to create a session; otherwise auth is available but not enforced.

This mirrors NexusAI's Supabase-Auth capability (signup → token → verify → protected
endpoints) without taking on a third-party dependency.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

_PBKDF2_ROUNDS = 120_000


def _secret() -> bytes:
    return os.environ.get("CP_AUTH_SECRET", "dev-insecure-secret-change-me").encode()


# ---- password hashing ----
def hash_password(password: str) -> tuple[str, str]:
    """Return (salt_hex, hash_hex) for a new password."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return hmac.compare_digest(digest.hex(), hash_hex)


# ---- signed bearer tokens ----
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(email: str, *, ttl_seconds: int = 24 * 3600) -> str:
    payload = {"sub": email, "exp": int(time.time()) + ttl_seconds}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(token: str) -> str | None:
    """Return the email if the token is valid and unexpired, else None."""
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    expected = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload.get("sub")


def require_auth_enabled() -> bool:
    return os.environ.get("CP_REQUIRE_AUTH", "").strip() in ("1", "true", "yes")


def bearer_email(authorization: str) -> str | None:
    """Extract and verify the email from an ``Authorization: Bearer <token>`` header."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return verify_token(authorization[7:].strip())
