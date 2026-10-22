"""
Authentication & roles
======================
One client company, two roles:
  engineer - datasets, column mapping, training, model activation, belt types, + everything a planner can do
  planner  - belt fleet, inspections, predictions, recording failures/replacements

Passwords: scrypt (stdlib). Sessions: random token in an HttpOnly, SameSite=Strict cookie;
only its SHA-256 is stored, so a leaked database does not leak live sessions.
"""

import hashlib
import hmac
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request

from config import SESSION_HOURS
from db import get_conn, now_iso

COOKIE_NAME = "belt_session"
MIN_PASSWORD_LENGTH = 10
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}

# Brute-force protection: max failed logins per client IP inside the window.
_MAX_FAILURES, _WINDOW_S = 10, 15 * 60
_failures = defaultdict(list)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), dklen=32, n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(calc.hex(), digest)
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def check_rate_limit(ip: str) -> None:
    now = time.time()
    _failures[ip] = [t for t in _failures[ip] if now - t < _WINDOW_S]
    if len(_failures[ip]) >= _MAX_FAILURES:
        raise HTTPException(status_code=429, detail="Too many failed logins. Try again in 15 minutes.")


def record_failure(ip: str) -> None:
    _failures[ip].append(time.time())


def authenticate(username: str, password: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? AND active = 1", (username,)).fetchone()
    if row is None:
        hash_password(password)  # equalise timing so usernames can't be probed
        return None
    return dict(row) if verify_password(password, row["password_hash"]) else None


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now_iso(),))
        conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?,?,?,?)",
                     (_token_hash(token), user_id, expires.isoformat(timespec="seconds"), now_iso()))
    return token


def delete_session(token: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not signed in.")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT u.id, u.username, u.role, u.tours_seen FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ? AND u.active = 1",
            (_token_hash(token), now_iso()),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Session expired. Please sign in again.")
    return dict(row)


def require_engineer(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "engineer":
        raise HTTPException(status_code=403, detail="Engineer role required.")
    return user


def create_user(username: str, password: str, role: str) -> int:
    if role not in ("engineer", "planner"):
        raise ValueError("role must be 'engineer' or 'planner'")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                           (username.strip(), hash_password(password), role, now_iso()))
        return cur.lastrowid
