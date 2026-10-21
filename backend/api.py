"""
Belt RUL API
============
FastAPI backend + static frontend for one client company (engineer / planner roles).

  /api/auth/*         login, logout, current user
  /api/belt-types     belt-type profiles with their own failure limits   (edit: engineer)
  /api/belts          fleet, inspections, recorded failures/replacements (engineer + planner)
  /api/datasets       client datasets + column mapping review            (engineer)
  /api/models         model versions: train, activate, compare, delete   (engineer)
  /api/users          user accounts                                      (engineer)
  /api/dev/*          synthetic data generation                          (engineer, DEV_MODE only)

Run: python backend/api.py   (first user: python backend/manage.py create-user ...)
"""

import io
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auth
import model_registry as reg
import schema_mapper as sm
from config import (COOKIE_SECURE, DATASETS_DIR, DEV_MODE, FRONTEND_DIR, GENERATOR_SCRIPT,
                    MAPPINGS_DIR, MAX_UPLOAD_MB, SESSION_HOURS)
from db import BELT_FAMILIES, get_conn, init_db, list_belt_types, match_belt_type, now_iso, row_to_dict

log = logging.getLogger("belt_rul")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# API docs expose every endpoint - only in development.
app = FastAPI(title="Belt RUL API", lifespan=lifespan, docs_url="/docs" if DEV_MODE else None,
              redoc_url=None, openapi_url="/openapi.json" if DEV_MODE else None)
# No CORS middleware on purpose: the frontend is served from this same origin.


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    # Never leak tracebacks to the client; log them server-side.
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.exception_handler(reg.TrainingError)
async def training_error(request: Request, exc: reg.TrainingError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def bad_request(msg: str):
    raise HTTPException(status_code=400, detail=msg)


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


@app.post("/api/auth/login")
def login(body: LoginBody, request: Request, response: Response):
    ip = request.client.host if request.client else "unknown"
    auth.check_rate_limit(ip)
    user = auth.authenticate(body.username, body.password)
    if user is None:
        auth.record_failure(ip)
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    token = auth.create_session(user["id"])
    response.set_cookie(auth.COOKIE_NAME, token, max_age=int(SESSION_HOURS * 3600), httponly=True,
                        secure=COOKIE_SECURE, samesite="strict", path="/")
    return {"id": user["id"], "username": user["username"], "role": user["role"], "dev_mode": DEV_MODE}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(auth.COOKIE_NAME)
    if token:
        auth.delete_session(token)
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(auth.current_user)):
    return {**user, "dev_mode": DEV_MODE}


# ─────────────────────────────────────────────────────────────────────────────
# Users (engineer)
# ─────────────────────────────────────────────────────────────────────────────

class UserBody(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(min_length=auth.MIN_PASSWORD_LENGTH, max_length=200)
    role: str = Field(pattern="^(engineer|planner)$")


@app.get("/api/users")
def list_users(user: dict = Depends(auth.require_engineer)):
    with get_conn() as conn:
        rows = conn.execute("SELECT id, username, role, active, created_at FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/users")
def add_user(body: UserBody, user: dict = Depends(auth.require_engineer)):
    try:
        return {"id": auth.create_user(body.username, body.password, body.role)}
    except Exception as e:
        bad_request("Username already exists." if "UNIQUE" in str(e) else str(e))


@app.put("/api/users/{user_id}/active")
def set_user_active(user_id: int, body: Dict[str, bool], user: dict = Depends(auth.require_engineer)):
    if user_id == user["id"]:
        bad_request("You cannot deactivate your own account.")
    with get_conn() as conn:
        conn.execute("UPDATE users SET active = ? WHERE id = ?", (1 if body.get("active") else 0, user_id))
        if not body.get("active"):
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Schema & belt types
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/schema")
def get_schema(user: dict = Depends(auth.current_user)):
    return sm.load_canonical()


DEGRADATION_FIELDS = sm.fields_by_role(sm.load_canonical(), "degradation")


class BeltTypeBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    family: str
    aliases: List[str] = []
    cord_material: str = Field(default="", max_length=80)
    notes: str = Field(default="", max_length=1000)
    thresholds: Dict[str, float]
    thresholds_confirmed: bool = False


def _validate_type(body: BeltTypeBody) -> dict:
    if body.family not in BELT_FAMILIES:
        bad_request(f"family must be one of {BELT_FAMILIES}")
    thr = {k: float(v) for k, v in body.thresholds.items() if k in DEGRADATION_FIELDS and v is not None}
    if not thr or any(v <= 0 for v in thr.values()):
        bad_request("Give at least one positive failure limit (elongation, wear or crack depth).")
    return thr


@app.get("/api/belt-types")
def get_belt_types(user: dict = Depends(auth.current_user)):
    active = reg.active_model_ids()
    with get_conn() as conn:
        types = list_belt_types(conn)
        counts = dict(conn.execute("SELECT belt_type_id, COUNT(*) FROM belts GROUP BY belt_type_id").fetchall())
    for t in types:
        t["active_model_id"] = active.get(t["id"])
        t["belt_count"] = counts.get(t["id"], 0)
    return {"types": types, "families": BELT_FAMILIES, "degradation_fields": DEGRADATION_FIELDS}


@app.post("/api/belt-types")
def create_belt_type(body: BeltTypeBody, user: dict = Depends(auth.require_engineer)):
    thr = _validate_type(body)
    try:
        with get_conn() as conn:
            new_id = conn.execute(
                "INSERT INTO belt_types (name, family, aliases, cord_material, notes, thresholds_json, "
                "thresholds_confirmed, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (body.name.strip(), body.family, ", ".join(a.strip() for a in body.aliases if a.strip()),
                 body.cord_material, body.notes, json.dumps(thr), int(body.thresholds_confirmed), now_iso(), now_iso()),
            ).lastrowid
    except Exception as e:
        bad_request("A belt type with this name already exists." if "UNIQUE" in str(e) else str(e))
    return {"id": new_id}


@app.put("/api/belt-types/{type_id}")
def update_belt_type(type_id: int, body: BeltTypeBody, user: dict = Depends(auth.require_engineer)):
    thr = _validate_type(body)
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE belt_types SET name=?, family=?, aliases=?, cord_material=?, notes=?, thresholds_json=?, "
            "thresholds_confirmed=?, updated_at=? WHERE id=?",
            (body.name.strip(), body.family, ", ".join(a.strip() for a in body.aliases if a.strip()),
             body.cord_material, body.notes, json.dumps(thr), int(body.thresholds_confirmed), now_iso(), type_id))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Belt type not found.")
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Belts & inspections (engineer + planner)
# ─────────────────────────────────────────────────────────────────────────────

STATIC_FIELDS = [k for k, v in sm.load_canonical()["fields"].items() if v.get("group") == "static"]


class BeltBody(BaseModel):
    tag: str = Field(min_length=1, max_length=60)
    belt_type_id: int
    location: str = Field(default="", max_length=120)
    installed_on: Optional[str] = Field(default=None, max_length=10)
    static: Dict[str, Optional[float]] = {}


class InspectionBody(BaseModel):
    hour: float = Field(ge=0, le=200000)
    readings: Dict[str, Optional[float]]


class EndOfLifeBody(BaseModel):
    status: str = Field(pattern="^(failed|replaced|in_service)$")
    hour: Optional[float] = Field(default=None, ge=0, le=200000)
    reason: str = Field(default="", max_length=300)


def _clean_static(static: Dict) -> Dict:
    return {k: float(v) for k, v in static.items() if k in STATIC_FIELDS and v is not None}


def _belt(conn, belt_id: int) -> dict:
    row = conn.execute("SELECT * FROM belts WHERE id = ?", (belt_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Belt not found.")
    return row_to_dict(row)


def _inspections(conn, belt_id: int) -> List[dict]:
    rows = conn.execute("SELECT i.*, u.username AS created_by_name FROM inspections i "
                        "LEFT JOIN users u ON u.id = i.created_by WHERE belt_id = ? ORDER BY hour", (belt_id,)).fetchall()
    return [row_to_dict(r) for r in rows]


def _type_map(conn) -> Dict[int, dict]:
    return {t["id"]: t for t in list_belt_types(conn)}


@app.get("/api/belts")
def list_belts(user: dict = Depends(auth.current_user)):
    active = reg.active_model_ids()
    with get_conn() as conn:
        types = _type_map(conn)
        belts = [row_to_dict(r) for r in conn.execute("SELECT * FROM belts ORDER BY tag").fetchall()]
        latest = {r["belt_id"]: row_to_dict(r) for r in conn.execute(
            "SELECT i.* FROM inspections i JOIN (SELECT belt_id, MAX(hour) h FROM inspections GROUP BY belt_id) x "
            "ON x.belt_id = i.belt_id AND x.h = i.hour").fetchall()}
        n_insp = dict(conn.execute("SELECT belt_id, COUNT(*) FROM inspections GROUP BY belt_id").fetchall())
    out = []
    for b in belts:
        t = types[b["belt_type_id"]]
        last = latest.get(b["id"])
        pred = None
        if last and b["status"] == "in_service":
            # Live with the CURRENT active model (it may have changed since the inspection).
            pred = reg.predict_inspections(b, t, [{"hour": last["hour"], "readings": last["readings"]}],
                                           active.get(b["belt_type_id"]))[0]
        out.append({**b, "belt_type_name": t["name"], "inspection_count": n_insp.get(b["id"], 0),
                    "last_hour": last["hour"] if last else None, "last_inspected_at": last["created_at"] if last else None,
                    "prediction": pred, "has_active_model": b["belt_type_id"] in active})
    return out


@app.post("/api/belts")
def create_belt(body: BeltBody, user: dict = Depends(auth.current_user)):
    with get_conn() as conn:
        if body.belt_type_id not in _type_map(conn):
            bad_request("Unknown belt type.")
        try:
            new_id = conn.execute(
                "INSERT INTO belts (tag, belt_type_id, location, installed_on, static_json, created_at, created_by) "
                "VALUES (?,?,?,?,?,?,?)",
                (body.tag.strip(), body.belt_type_id, body.location.strip(), body.installed_on,
                 json.dumps(_clean_static(body.static)), now_iso(), user["id"])).lastrowid
        except Exception as e:
            bad_request("A belt with this tag already exists." if "UNIQUE" in str(e) else str(e))
    return {"id": new_id}


@app.put("/api/belts/{belt_id}")
def update_belt(belt_id: int, body: BeltBody, user: dict = Depends(auth.current_user)):
    with get_conn() as conn:
        _belt(conn, belt_id)
        if body.belt_type_id not in _type_map(conn):
            bad_request("Unknown belt type.")
        conn.execute("UPDATE belts SET tag=?, belt_type_id=?, location=?, installed_on=?, static_json=? WHERE id=?",
                     (body.tag.strip(), body.belt_type_id, body.location.strip(), body.installed_on,
                      json.dumps(_clean_static(body.static)), belt_id))
    return {"ok": True}


@app.get("/api/belts/{belt_id}")
def belt_detail(belt_id: int, user: dict = Depends(auth.current_user)):
    active = reg.active_model_ids()
    with get_conn() as conn:
        belt = _belt(conn, belt_id)
        btype = _type_map(conn)[belt["belt_type_id"]]
        insps = _inspections(conn, belt_id)
    model_id = active.get(belt["belt_type_id"])
    preds = reg.predict_inspections(belt, btype, [{"hour": i["hour"], "readings": i["readings"]} for i in insps], model_id)
    model = reg.get_model(model_id) if model_id else None
    return {
        "belt": belt, "belt_type": btype, "inspections": insps, "predictions": preds,
        "model": {"id": model["id"], "feature_cols": model["info"]["feature_cols"],
                  "feature_stats": model["info"]["feature_stats"], "metrics": model["info"]["metrics"],
                  "rul_estimated": model["info"]["rul_estimated"]} if model else None,
    }


@app.post("/api/belts/{belt_id}/inspections")
def add_inspection(belt_id: int, body: InspectionBody, user: dict = Depends(auth.current_user)):
    fields = sm.load_canonical()["fields"]
    readings = {k: float(v) for k, v in body.readings.items()
                if v is not None and (k in fields or k.startswith(sm.EXTRA_PREFIX))}
    active = reg.active_model_ids()
    with get_conn() as conn:
        belt = _belt(conn, belt_id)
        if belt["status"] != "in_service":
            bad_request("This belt is no longer in service.")
        btype = _type_map(conn)[belt["belt_type_id"]]
        model_id = active.get(belt["belt_type_id"])
        prior = [{"hour": i["hour"], "readings": i["readings"]} for i in _inspections(conn, belt_id) if i["hour"] != body.hour]
        history = sorted(prior + [{"hour": body.hour, "readings": readings}], key=lambda i: i["hour"])
        pred = reg.predict_inspections(belt, btype, history, model_id)[[h["hour"] for h in history].index(body.hour)]
        # Store the prediction made at inspection time (model version included) - this is the
        # audit trail for later comparing predictions with the belt's actual failure.
        conn.execute(
            "INSERT INTO inspections (belt_id, hour, readings_json, predicted_rul, health_score, model_id, created_at, created_by) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(belt_id, hour) DO UPDATE SET readings_json=excluded.readings_json, "
            "predicted_rul=excluded.predicted_rul, health_score=excluded.health_score, model_id=excluded.model_id, "
            "created_at=excluded.created_at, created_by=excluded.created_by",
            (belt_id, body.hour, json.dumps(readings), pred["rul_hours"], pred["health_score"], model_id, now_iso(), user["id"]))
    return belt_detail(belt_id, user)


@app.delete("/api/belts/{belt_id}/inspections/{inspection_id}")
def delete_inspection(belt_id: int, inspection_id: int, user: dict = Depends(auth.current_user)):
    with get_conn() as conn:
        conn.execute("DELETE FROM inspections WHERE id = ? AND belt_id = ?", (inspection_id, belt_id))
    return belt_detail(belt_id, user)


@app.put("/api/belts/{belt_id}/end-of-life")
def end_of_life(belt_id: int, body: EndOfLifeBody, user: dict = Depends(auth.current_user)):
    if body.status != "in_service" and body.hour is None:
        bad_request("Enter the operating hour at which the belt failed / was replaced.")
    with get_conn() as conn:
        _belt(conn, belt_id)
        conn.execute("UPDATE belts SET status=?, end_hour=?, end_reason=? WHERE id=?",
                     (body.status, None if body.status == "in_service" else body.hour,
                      None if body.status == "in_service" else body.reason, belt_id))
    return belt_detail(belt_id, user)


# ─────────────────────────────────────────────────────────────────────────────
# Datasets & column mapping (engineer)
# ─────────────────────────────────────────────────────────────────────────────

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def dataset_path(name: Optional[str]) -> str:
    """Datasets always resolve inside DATASETS_DIR (blocks path traversal)."""
    path = os.path.join(DATASETS_DIR, os.path.basename(name or ""))
    if not name or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return path


def _dataset_summary(fn: str) -> dict:
    full = os.path.join(DATASETS_DIR, fn)
    try:
        cols = list(pd.read_csv(full, nrows=0).columns)
    except Exception:
        cols = []
    saved = sm.load_mapping(MAPPINGS_DIR, fn, cols)
    mapping = saved["mapping"] if saved else (sm.identity_mapping(cols) if sm.is_fully_canonical(cols) else None)
    return {
        "filename": fn, "size_mb": round(os.path.getsize(full) / 1_048_576, 2), "columns": cols,
        "mapping_status": "approved" if saved else "not_needed" if mapping else "required",
        "has_belt_type_column": bool(mapping and any(m["target"] == "belt_type" for m in mapping)),
    }


@app.get("/api/datasets")
def list_datasets(user: dict = Depends(auth.require_engineer)):
    files = sorted(f for f in os.listdir(DATASETS_DIR) if f.lower().endswith(".csv"))
    return [_dataset_summary(f) for f in files]


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), user: dict = Depends(auth.require_engineer)):
    stem = SAFE_NAME.sub("_", os.path.splitext(os.path.basename(file.filename or ""))[0]).strip("._")[:80]
    if not stem or not (file.filename or "").lower().endswith(".csv"):
        bad_request("Only .csv files are accepted.")
    contents = await file.read(MAX_UPLOAD_MB * 1_048_576 + 1)
    if len(contents) > MAX_UPLOAD_MB * 1_048_576:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB.")
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        bad_request("Could not parse the file as CSV.")
    if df.empty or len(df.columns) < 2:
        bad_request("The CSV has no data.")
    # Never overwrite: an existing file may be referenced by approved mappings and model versions.
    name, n = f"{stem}.csv", 2
    while os.path.exists(os.path.join(DATASETS_DIR, name)):
        name, n = f"{stem}_{n}.csv", n + 1
    with open(os.path.join(DATASETS_DIR, name), "wb") as f:
        f.write(contents)
    return {**_dataset_summary(name), "rows": int(len(df))}


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str, user: dict = Depends(auth.require_engineer)):
    path = dataset_path(name)
    os.remove(path)
    mapping = os.path.join(MAPPINGS_DIR, f"{os.path.basename(path)}.mapping.json")
    if os.path.exists(mapping):
        os.remove(mapping)
    return {"ok": True}


class ProposeBody(BaseModel):
    filename: str


class ApproveBody(BaseModel):
    filename: str
    mapping: List[Dict]
    proposal_source: Optional[str] = "manual"
    dry_run: bool = False


@app.post("/api/schema/propose")
def propose_schema(body: ProposeBody, user: dict = Depends(auth.require_engineer)):
    path = dataset_path(body.filename)
    raw = pd.read_csv(path)
    saved = sm.load_mapping(MAPPINGS_DIR, path, raw.columns)
    if saved:
        proposal = {"source": "approved", "mapping": saved["mapping"], "profile": sm.profile_dataframe(raw), "warnings": []}
    else:
        proposal = sm.propose_mapping(raw, MAPPINGS_DIR)
    proposal["already_approved"] = saved is not None
    proposal["canonical_fields"] = sm.load_canonical()["fields"]
    return proposal


@app.post("/api/schema/approve")
def approve_schema(body: ApproveBody, user: dict = Depends(auth.require_engineer)):
    path = dataset_path(body.filename)
    raw = pd.read_csv(path)
    errors, warnings, rul_info = sm.validate_mapping(raw, body.mapping)
    if not errors:
        # Belt-type values must match a configured type (or alias) before training.
        bt = next((m["source"] for m in body.mapping if m.get("target") == "belt_type"), None)
        if bt:
            with get_conn() as conn:
                types = list_belt_types(conn)
            unknown = sorted({str(v) for v in raw[bt].dropna().unique() if match_belt_type(types, v) is None})
            if unknown:
                warnings.append(f"Belt-type values not configured yet: {unknown[:10]}. Add them under Belt Types "
                                "(as a new type or an alias) before training.")
    if errors:
        return JSONResponse(status_code=422, content={"saved": False, "errors": errors, "warnings": warnings})
    if body.dry_run:
        # UI validates first so the reviewer sees warnings BEFORE anything is saved
        return {"saved": False, "errors": [], "warnings": warnings, "rul_source": rul_info}
    sm.save_mapping(MAPPINGS_DIR, path, raw.columns, body.mapping, body.proposal_source or "manual")
    return {"saved": True, "errors": [], "warnings": warnings, "rul_source": rul_info}


# ─────────────────────────────────────────────────────────────────────────────
# Model versions (engineer)
# ─────────────────────────────────────────────────────────────────────────────

class TrainDataset(BaseModel):
    filename: str
    belt_type_id: Optional[int] = None


class TrainBody(BaseModel):
    datasets: List[TrainDataset] = Field(min_length=1)
    notes: str = Field(default="", max_length=500)
    method: Literal["aft", "regression"] = "aft"   # survival (median + interval) or legacy point regression


class ActivateBody(BaseModel):
    belt_type_ids: Optional[List[int]] = None


class ReviewBody(BaseModel):
    review_note: str = Field(default="", max_length=1000)


@app.get("/api/models")
def get_models(user: dict = Depends(auth.require_engineer)):
    return reg.list_models()


@app.get("/api/models/compare")
def compare_models(a: int, b: int, user: dict = Depends(auth.require_engineer)):
    return reg.compare(a, b)


@app.get("/api/models/{model_id}")
def get_model(model_id: int, user: dict = Depends(auth.require_engineer)):
    m = reg.get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    return m


@app.post("/api/models/train", status_code=202)
def train(body: TrainBody, user: dict = Depends(auth.require_engineer)):
    for d in body.datasets:
        dataset_path(d.filename)
    specs = [{"filename": d.filename, "belt_type_id": d.belt_type_id} for d in body.datasets]
    return {"id": reg.start_training(specs, body.notes, user["id"], body.method), "status": "training"}


@app.post("/api/models/{model_id}/activate")
def activate_model(model_id: int, body: ActivateBody, user: dict = Depends(auth.require_engineer)):
    return {"active_for": reg.activate(model_id, user["id"], body.belt_type_ids)}


@app.put("/api/models/{model_id}/review")
def review_model(model_id: int, body: ReviewBody, user: dict = Depends(auth.require_engineer)):
    with get_conn() as conn:
        conn.execute("UPDATE models SET review_note = ? WHERE id = ?", (body.review_note, model_id))
    return {"ok": True}


@app.delete("/api/models/{model_id}")
def delete_model(model_id: int, user: dict = Depends(auth.require_engineer)):
    reg.delete_model(model_id)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Development only: synthetic V-belt data
# ─────────────────────────────────────────────────────────────────────────────

class GenerateBody(BaseModel):
    belts: int = Field(default=300, ge=10, le=3000)
    step: int = Field(default=250, ge=10, le=2000)
    seed: int = Field(default=42, ge=0, le=2**31)
    censor_frac: float = Field(default=0.0, ge=0.0, le=0.9)


@app.post("/api/dev/generate-data")
def generate_data(body: GenerateBody, user: dict = Depends(auth.require_engineer)):
    if not DEV_MODE:
        raise HTTPException(status_code=404, detail="Not found.")
    out = os.path.join(DATASETS_DIR, f"synthetic_vbelts_seed{body.seed}.csv")
    # Argument list, no shell: nothing from the request can be interpreted as a command.
    proc = subprocess.run(
        [sys.executable, GENERATOR_SCRIPT, "--belts", str(body.belts), "--step", str(body.step),
         "--seed", str(body.seed), "--censor-frac", str(body.censor_frac), "--output", out],
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        log.error("Generator failed: %s", proc.stderr)
        raise HTTPException(status_code=500, detail="Data generation failed (see server log).")
    return {"filename": os.path.basename(out), "log": proc.stdout[-2000:]}


# Frontend (mounted last so /api routes take precedence)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=os.environ.get("BELT_RUL_HOST", "127.0.0.1"),
                port=int(os.environ.get("BELT_RUL_PORT", "8001")), reload=False)
