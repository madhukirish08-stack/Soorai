"""
Runtime configuration
=====================
All mutable state (client datasets, approved mappings, model versions, the SQLite
database) lives under INSTANCE_DIR, which is git-ignored. Client data must never be
committed to the code repository.

Environment variables (also read from <repo>/.env):
  GEMINI_API_KEY          - enables AI column-mapping proposals
  GEMINI_MODEL            - default gemini-3.8-flash
  BELT_RUL_INSTANCE_DIR   - default <repo>/instance
  BELT_RUL_DEV_MODE       - "1" enables synthetic-data generation, /docs, non-HTTPS cookies
  BELT_RUL_COOKIE_SECURE  - override the Secure cookie flag ("1"/"0")
  BELT_RUL_SESSION_HOURS  - login session lifetime, default 12
"""

import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path: str) -> None:
    """Minimal .env loader (KEY=VALUE lines). Real environment variables take precedence."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(REPO_DIR, ".env"))


def _flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val.strip().lower() in ("1", "true", "yes", "on")


DEV_MODE      = _flag("BELT_RUL_DEV_MODE", False)
# Secure cookies by default; only plain-HTTP local development may turn them off.
COOKIE_SECURE = _flag("BELT_RUL_COOKIE_SECURE", not DEV_MODE)
SESSION_HOURS = float(os.environ.get("BELT_RUL_SESSION_HOURS", "12"))

INSTANCE_DIR  = os.path.abspath(os.environ.get("BELT_RUL_INSTANCE_DIR", os.path.join(REPO_DIR, "instance")))
DATASETS_DIR  = os.path.join(INSTANCE_DIR, "datasets")
MAPPINGS_DIR  = os.path.join(INSTANCE_DIR, "mappings")
MODELS_DIR    = os.path.join(INSTANCE_DIR, "models")
DB_PATH       = os.path.join(INSTANCE_DIR, "belt_rul.db")

FRONTEND_DIR  = os.path.join(REPO_DIR, "frontend")
GENERATOR_SCRIPT = os.path.join(REPO_DIR, "training", "generate_synthetic_vbelts.py")

MAX_UPLOAD_MB = int(os.environ.get("BELT_RUL_MAX_UPLOAD_MB", "100"))

for _d in (DATASETS_DIR, MAPPINGS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)
