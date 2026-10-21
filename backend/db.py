"""
SQLite persistence (stdlib only)
================================
Users/sessions, belt types, the belt fleet, inspection history, model versions and
which model version is active for each belt type.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('engineer', 'planner')),
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS belt_types (
    id                   INTEGER PRIMARY KEY,
    name                 TEXT NOT NULL UNIQUE COLLATE NOCASE,
    family               TEXT NOT NULL,
    aliases              TEXT NOT NULL DEFAULT '',
    cord_material        TEXT NOT NULL DEFAULT '',
    notes                TEXT NOT NULL DEFAULT '',
    thresholds_json      TEXT NOT NULL,
    thresholds_confirmed INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS belts (
    id           INTEGER PRIMARY KEY,
    tag          TEXT NOT NULL UNIQUE COLLATE NOCASE,
    belt_type_id INTEGER NOT NULL REFERENCES belt_types(id),
    location     TEXT NOT NULL DEFAULT '',
    installed_on TEXT,
    static_json  TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'in_service' CHECK (status IN ('in_service', 'failed', 'replaced')),
    end_hour     REAL,
    end_reason   TEXT,
    created_at   TEXT NOT NULL,
    created_by   INTEGER REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS inspections (
    id            INTEGER PRIMARY KEY,
    belt_id       INTEGER NOT NULL REFERENCES belts(id) ON DELETE CASCADE,
    hour          REAL NOT NULL,
    readings_json TEXT NOT NULL,
    predicted_rul REAL,
    health_score  REAL,
    model_id      INTEGER REFERENCES models(id),
    created_at    TEXT NOT NULL,
    created_by    INTEGER REFERENCES users(id),
    UNIQUE (belt_id, hour)
);
CREATE TABLE IF NOT EXISTS models (
    id          INTEGER PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'training' CHECK (status IN ('training', 'ready', 'failed')),
    notes       TEXT NOT NULL DEFAULT '',
    review_note TEXT NOT NULL DEFAULT '',
    info_json   TEXT NOT NULL DEFAULT '{}',
    error       TEXT,
    created_at  TEXT NOT NULL,
    created_by  INTEGER REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS active_models (
    belt_type_id INTEGER PRIMARY KEY REFERENCES belt_types(id) ON DELETE CASCADE,
    model_id     INTEGER NOT NULL REFERENCES models(id),
    activated_at TEXT NOT NULL,
    activated_by INTEGER REFERENCES users(id)
);
"""

# Placeholder failure limits. They are NOT JK Fenner figures - every type is seeded as
# unconfirmed and the UI flags it until an engineer confirms or corrects the limits.
UNCONFIRMED_NOTE = "Default failure limits are placeholders - confirm with JK Fenner / site engineering."
SEED_BELT_TYPES = [
    {"name": "Classical wrapped", "family": "classical_wrapped", "cord_material": "Polyester",
     "aliases": "Poly-F Plus, Poly-F Plus PB, Ecodrive, Ecodrive Plus, JK Transpower, A, B, C, D, E",
     "thresholds": {"elongation_pct": 3.0, "wear_pct": 100.0, "crack_depth_mm": 2.0}},
    {"name": "Wedge (SpacesaVer)", "family": "wedge", "cord_material": "Polyester",
     "aliases": "SpacesaVer, SPZ, SPA, SPB, SPC, 3V, 5V, 8V",
     "thresholds": {"elongation_pct": 2.0, "wear_pct": 100.0, "crack_depth_mm": 1.5}},
    {"name": "Raw-edge cogged", "family": "raw_edge_cogged", "cord_material": "Polyester",
     "aliases": "AX, BX, CX, SPZX, SPAX, SPBX, SPCX, XPZ, XPA, XPB, XPC",
     "thresholds": {"elongation_pct": 2.0, "wear_pct": 100.0, "crack_depth_mm": 2.5}},
]

BELT_FAMILIES = ["classical_wrapped", "wedge", "raw_edge_cogged", "poly_v", "timing", "specialty"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM belt_types").fetchone()[0] == 0:
            for t in SEED_BELT_TYPES:
                conn.execute(
                    "INSERT INTO belt_types (name, family, aliases, cord_material, notes, thresholds_json,"
                    " thresholds_confirmed, created_at, updated_at) VALUES (?,?,?,?,?,?,0,?,?)",
                    (t["name"], t["family"], t["aliases"], t["cord_material"], UNCONFIRMED_NOTE,
                     json.dumps(t["thresholds"]), now_iso(), now_iso()),
                )


def row_to_dict(row) -> dict:
    if row is None:
        return None
    d = dict(row)
    for key in list(d):
        if key.endswith("_json"):
            d[key[:-5]] = json.loads(d.pop(key) or "null")
    return d


# ── Belt types ───────────────────────────────────────────────────────────────

def list_belt_types(conn) -> list:
    rows = conn.execute("SELECT * FROM belt_types ORDER BY id").fetchall()
    types = [row_to_dict(r) for r in rows]
    for t in types:
        t["thresholds_confirmed"] = bool(t["thresholds_confirmed"])
        t["aliases"] = [a.strip() for a in t["aliases"].split(",") if a.strip()]
    return types


def belt_type_thresholds(conn) -> dict:
    """{belt_type_id: {degradation_field: failure_threshold}}"""
    return {t["id"]: t["thresholds"] for t in list_belt_types(conn)}


def match_belt_type(types: list, value: str):
    """Resolve a dataset's belt-type value (name, section or alias) to a belt type id."""
    v = str(value).strip().lower()
    for t in types:
        if v == t["name"].lower() or v in (a.lower() for a in t["aliases"]):
            return t["id"]
    return None
