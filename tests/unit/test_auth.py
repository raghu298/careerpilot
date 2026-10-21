"""Optional accounts: password hashing, signed tokens, and the signup/login endpoints."""

import os

from api.auth import hash_password, make_token, verify_password, verify_token


def test_password_hash_roundtrip():
    salt, h = hash_password("hunter2")
    assert verify_password("hunter2", salt, h) is True
    assert verify_password("wrong", salt, h) is False


def test_token_sign_verify_and_tamper():
    tok = make_token("jane@acme.io")
    assert verify_token(tok) == "jane@acme.io"
    # tamper with the signature → rejected
    body, sig = tok.split(".", 1)
    assert verify_token(f"{body}.{sig[:-2]}xx") is None
    assert verify_token("garbage") is None


def test_expired_token_rejected():
    tok = make_token("jane@acme.io", ttl_seconds=-1)
    assert verify_token(tok) is None


def test_auth_endpoints_signup_login_me():
    os.environ["CP_DB_PATH"] = ":memory:"
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as c:
        r = c.post("/auth/signup", json={"email": "A@Acme.io", "password": "secret1"})
        assert r.status_code == 200
        token = r.json()["token"]
        assert r.json()["email"] == "a@acme.io"  # normalized
        # duplicate signup
        assert c.post("/auth/signup", json={"email": "a@acme.io", "password": "secret1"}).status_code == 409
        # wrong password
        assert c.post("/auth/login", json={"email": "a@acme.io", "password": "nope"}).status_code == 401
        # right password
        assert c.post("/auth/login", json={"email": "a@acme.io", "password": "secret1"}).status_code == 200
        # /auth/me with bearer
        me = c.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200 and me.json()["email"] == "a@acme.io"
        assert c.get("/auth/me").status_code == 401
