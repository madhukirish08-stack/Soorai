"""
Fleet operations
================
  - store_inspection()   the ONE place a prediction made at inspection time is written
  - import_csv()         bulk import of belts / inspections with a validate-first (dry run) step
  - accuracy_report()    backtest: what each model version predicted vs. what actually happened
  - actual_remaining()   ground truth for one inspection of an ended belt
  - export_fleet_dataset() turn recorded fleet history into a training dataset (closes the loop)

Invariants (keep):
- Import is ALL-OR-NOTHING: any invalid row blocks the commit, so bad data never reaches the
  database half-loaded. Re-importing a corrected file is safe (inspections upsert on belt+hour).
- Accuracy counts only belts that FAILED. A belt that was replaced is censored (it lasted at least
  that long) and is reported separately - treating replacements as failures would reward
  pessimistic models and punish good ones.
- Predictions are stored at inspection time (with model version and interval). The backtest uses
  those stored values, never a re-prediction with today's model.
"""

import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import model_registry as reg
import schema_mapper as sm
from db import list_belt_types, match_belt_type, now_iso, row_to_dict

MAX_IMPORT_ROWS = 20000
MAX_ERRORS_SHOWN = 100
MIN_FAILED_FOR_CONFIDENCE = 10     # fewer failed belts -> accuracy figures are anecdotal
HORIZONS = [(0, 500), (500, 2000), (2000, 5000), (5000, float("inf"))]   # actual remaining hours

TAG_ALIASES = {"belt_tag", "tag", "belt", "belt_id", "id", "asset", "asset_tag", "asset_id"}
HOUR_ALIASES = {"hour", "hours", "operating_hours", "op_hours", "run_hours", "run_hrs", "hour_meter", "hrs"}
BELT_ATTR_ALIASES = {
    "__belt_type__": {"belt_type", "type", "section", "belt_section"},
    "__location__": {"location", "drive", "position", "equipment"},
    "__installed_on__": {"installed_on", "installed", "install_date", "installation_date"},
}
FIELD_ALIASES = {"temp": "temperature_c", "temp_c": "temperature_c", "temperature": "temperature_c", "load": "load_pct", "slip": "slip_pct",
                 "vibration": "vibration_mms", "rpm": "driver_rpm", "elongation": "elongation_pct", "wear": "wear_pct",
                 "crack": "crack_depth_mm", "crack_depth": "crack_depth_mm", "tension": "belt_tension_n",
                 "side_wear": "wear_pct", "sidewall": "wear_pct", "stretch": "elongation_pct"}


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(h).strip().lower()).strip("_")


# ─────────────────────────────────────────────────────────────────────────────
# Storing a prediction made at inspection time
# ─────────────────────────────────────────────────────────────────────────────

def store_inspection(conn, belt_id: int, hour: float, readings: Dict, pred: Dict, model_id: Optional[int], user_id: int) -> None:
    """Insert/update an inspection together with the prediction made NOW (model version + interval).
    This is the audit trail the accuracy backtest relies on. Used by single entry AND bulk import."""
    conn.execute(
        "INSERT INTO inspections (belt_id, hour, readings_json, predicted_rul, predicted_low, predicted_high, "
        "health_score, model_id, created_at, created_by) VALUES (?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(belt_id, hour) DO UPDATE SET readings_json=excluded.readings_json, "
        "predicted_rul=excluded.predicted_rul, predicted_low=excluded.predicted_low, predicted_high=excluded.predicted_high, "
        "health_score=excluded.health_score, model_id=excluded.model_id, created_at=excluded.created_at, created_by=excluded.created_by",
        (belt_id, hour, json.dumps(readings), pred["rul_hours"], pred["rul_low"], pred["rul_high"],
         pred["health_score"], model_id, now_iso(), user_id))


# ─────────────────────────────────────────────────────────────────────────────
# Bulk import
# ─────────────────────────────────────────────────────────────────────────────

def template_csv(kind: str) -> str:
    fields = sm.load_canonical()["fields"]
    if kind == "belts":
        cols = ["belt_tag", "belt_type", "location", "installed_on"] + [k for k, v in fields.items() if v.get("group") == "static"]
        example = ["EXAMPLE-DELETE-ME", "SPB", "Cooling tower pump 7", "2025-01-15", "1800", "112", "250", "", "2", "1440", "15"]
    else:
        cols = (["belt_tag", "hour"] + [k for k, v in fields.items() if v["role"] == "degradation"]
                + [k for k, v in fields.items() if v.get("group") == "operating"])
        example = ["EXAMPLE-DELETE-ME", "3000"] + [""] * (len(cols) - 2)
    example += [""] * (len(cols) - len(example))
    return ",".join(cols) + "\n" + ",".join(example[:len(cols)]) + "\n"


def _resolve_columns(columns: List[str], kind: str, fields: Dict) -> Dict:
    """Map each CSV header to a role: 'tag', 'hour', a belt attribute, or a canonical/extra field."""
    by_label = {_norm(v["label"]): k for k, v in fields.items()}
    resolved, seen = {}, set()
    for c in columns:
        n = _norm(c)
        target = None
        if n in TAG_ALIASES:
            target = "__tag__"
        elif kind == "inspections" and n in HOUR_ALIASES:
            target = "hour"
        elif kind == "belts" and any(n in names for names in BELT_ATTR_ALIASES.values()):
            target = next(t for t, names in BELT_ATTR_ALIASES.items() if n in names)
        elif n in fields and fields[n]["role"] in ("feature", "degradation"):
            target = n
        elif n in by_label and fields[by_label[n]]["role"] in ("feature", "degradation"):
            target = by_label[n]
        elif n in FIELD_ALIASES:
            target = FIELD_ALIASES[n]
        elif n.startswith(sm.EXTRA_PREFIX) and kind == "inspections":
            target = n
        if target and not target.startswith("__") and target != "hour":
            # Belt-level details belong in the belts file, per-inspection measurements in the inspections file.
            is_static = fields.get(target, {}).get("group") == "static"
            if is_static != (kind == "belts"):
                target = None
        if target and target in seen:
            target = None        # first column wins; the duplicate is reported as ignored
        if target:
            seen.add(target)
        resolved[c] = target
    return resolved


def _read(data: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        raise reg.TrainingError("Could not read the file as CSV.")
    if df.empty:
        raise reg.TrainingError("The file has no data rows.")
    if len(df) > MAX_IMPORT_ROWS:
        raise reg.TrainingError(f"Too many rows ({len(df):,}); the limit is {MAX_IMPORT_ROWS:,} per file. Split the file.")
    return df.apply(lambda col: col.str.strip())


def import_csv(conn, kind: str, data: bytes, dry_run: bool, user_id: int, default_belt_type_id: Optional[int] = None) -> Dict:
    """Validate (and unless dry_run, commit) a belts or inspections CSV. Returns a report dict."""
    if kind not in ("belts", "inspections"):
        raise reg.TrainingError("Unknown import kind.")
    fields = sm.load_canonical()["fields"]
    df = _read(data)
    colmap = _resolve_columns(list(df.columns), kind, fields)
    if "__tag__" not in colmap.values():
        raise reg.TrainingError("The file needs a belt tag column (e.g. 'belt_tag'). Download the template for the expected layout.")
    if kind == "inspections" and "hour" not in colmap.values():
        raise reg.TrainingError("The file needs an operating-hours column (e.g. 'hour').")

    report = {"kind": kind, "rows": int(len(df)), "errors": [], "error_count": 0, "warnings": [],
              "columns": [{"source": c, "field": t, "unit": fields.get(t, {}).get("unit")} for c, t in colmap.items()],
              "committed": False}
    report["ignored_columns"] = [c for c, t in colmap.items() if t is None]
    errors = report["errors"]

    def err(row, msg):
        report["error_count"] += 1
        if len(errors) < MAX_ERRORS_SHOWN:
            errors.append({"row": int(row), "message": msg})

    types = list_belt_types(conn)
    inv = {t: c for c, t in colmap.items() if t}
    handler = _import_inspections if kind == "inspections" else _import_belts
    parsed = handler(conn, df, inv, fields, types, report, err, default_belt_type_id)

    if report["error_count"]:
        report["warnings"].append(f"{report['error_count']} row(s) have errors. Nothing is imported until the file is fixed.")
    elif not dry_run and parsed:
        _commit(conn, kind, parsed, user_id, report)
        report["committed"] = True
    return report


def _num(val: str, col: str, row: int, err, minimum: Optional[float] = None) -> Optional[float]:
    if val == "":
        return None
    try:
        x = float(val)
    except ValueError:
        # Commas are never guessed at: "1,200" could be 1200 or 1.2. Ask for a plain number.
        err(row, f"'{col}': “{val}” is not a plain number (no thousands separators; use a dot for decimals).")
        return None
    if not np.isfinite(x):
        err(row, f"'{col}': “{val}” is not a valid number.")
        return None
    if minimum is not None and x < minimum:
        err(row, f"'{col}': {x:g} must be at least {minimum:g}.")
        return None
    return x


def _range_warnings(rows: List[Dict], fields: Dict, report: Dict, key: str) -> None:
    """Out-of-range values usually mean a wrong unit (Fahrenheit, inches, fraction instead of %)."""
    per_col: Dict[str, List[float]] = {}
    for r in rows:
        for k, v in r[key].items():
            per_col.setdefault(k, []).append(v)
    for col, vals in per_col.items():
        rng = fields.get(col, {}).get("range")
        if not rng:
            continue
        out = sum(1 for v in vals if v < rng[0] or v > rng[1])
        if out:
            report["warnings"].append(
                f"'{col}': {out} of {len(vals)} values are outside the expected {rng[0]:g}–{rng[1]:g} {fields[col]['unit'] or ''}. "
                "Check the units (the template uses the unit shown for each field).")


def _import_inspections(conn, df, inv, fields, types, report, err, _default):
    belts = {r["tag"].lower(): row_to_dict(r) for r in conn.execute("SELECT * FROM belts").fetchall()}
    rows, seen, skipped_empty = [], set(), 0
    unknown_tags = set()
    for i, rec in enumerate(df.to_dict("records")):
        line = i + 2
        tag = rec[inv["__tag__"]]
        if not tag:
            err(line, "Belt tag is empty.")
            continue
        belt = belts.get(tag.lower())
        if belt is None:
            unknown_tags.add(tag)
            err(line, f"Belt “{tag}” is not registered. Register it first (or import the belts file).")
            continue
        if belt["status"] != "in_service":
            err(line, f"Belt “{tag}” is no longer in service ({belt['status']}).")
            continue
        hour = _num(rec[inv["hour"]], inv["hour"], line, err, minimum=0)
        if hour is None:
            if rec[inv["hour"]] == "":
                err(line, "Operating hour is empty.")
            continue
        if hour > 200000:
            err(line, f"Operating hour {hour:g} is implausibly large.")
            continue
        readings = {}
        for field in [t for t in inv if t not in ("__tag__", "hour")]:
            val = _num(rec[inv[field]], inv[field], line, err)
            if val is None:
                continue
            if fields.get(field, {}).get("role") == "degradation" and val < 0:
                err(line, f"'{inv[field]}': a degradation measurement cannot be negative ({val:g}).")
                continue
            readings[field] = val
        if not readings:
            skipped_empty += 1
            continue
        key = (belt["id"], hour)
        if key in seen:
            err(line, f"Belt “{tag}” appears twice at hour {hour:g} in this file.")
            continue
        seen.add(key)
        rows.append({"line": line, "belt": belt, "hour": hour, "readings": readings})

    if skipped_empty:
        report["warnings"].append(f"{skipped_empty} row(s) had no measurements and were skipped.")
    existing = {(r["belt_id"], r["hour"]) for r in conn.execute("SELECT belt_id, hour FROM inspections").fetchall()}
    report["to_update"] = sum(1 for r in rows if (r["belt"]["id"], r["hour"]) in existing)
    report["to_create"] = len(rows) - report["to_update"]
    report["belts_affected"] = len({r["belt"]["id"] for r in rows})
    report["unknown_tags"] = sorted(unknown_tags)[:20]
    if report["to_update"]:
        report["warnings"].append(f"{report['to_update']} inspection(s) already exist for the same belt and hour and will be replaced.")
    _range_warnings(rows, fields, report, "readings")
    report["preview"] = [{"row": r["line"], "tag": r["belt"]["tag"], "hour": r["hour"], "readings": r["readings"]} for r in rows[:8]]
    return rows


def _import_belts(conn, df, inv, fields, types, report, err, default_type_id):
    existing = {r["tag"].lower() for r in conn.execute("SELECT tag FROM belts").fetchall()}
    static_cols = [t for t in inv if t not in ("__tag__", "__belt_type__", "__location__", "__installed_on__")]
    if "__belt_type__" not in inv and default_type_id is None:
        raise reg.TrainingError("The file has no belt-type column. Add one, or choose a default belt type for the whole file.")
    rows, seen, skipped_existing = [], set(), 0
    for i, rec in enumerate(df.to_dict("records")):
        line = i + 2
        tag = rec[inv["__tag__"]]
        if not tag:
            err(line, "Belt tag is empty.")
            continue
        if tag.lower() in existing:
            skipped_existing += 1
            continue
        if tag.lower() in seen:
            err(line, f"Belt “{tag}” appears twice in this file.")
            continue
        seen.add(tag.lower())
        if "__belt_type__" in inv and rec[inv["__belt_type__"]]:
            tid = match_belt_type(types, rec[inv["__belt_type__"]])
            if tid is None:
                err(line, f"Belt type “{rec[inv['__belt_type__']]}” is not configured. Add it under Belt Types (as a type or alias).")
                continue
        elif default_type_id is not None:
            tid = default_type_id
        else:
            err(line, "Belt type is empty.")
            continue
        installed = rec.get(inv.get("__installed_on__", ""), "")
        if installed:
            try:
                datetime.strptime(installed, "%Y-%m-%d")      # a real calendar date, not just a 4-2-2 digit shape
            except ValueError:
                err(line, f"Installed date “{installed}” is not a valid date (use 2025-01-31).")
                continue
        static = {}
        for field in static_cols:
            val = _num(rec[inv[field]], inv[field], line, err)
            if val is not None:
                static[field] = val
        rows.append({"line": line, "tag": tag, "belt_type_id": tid, "static": static,
                     "location": rec.get(inv.get("__location__", ""), ""), "installed_on": installed or None})
    if skipped_existing:
        report["warnings"].append(f"{skipped_existing} belt(s) are already registered and were left unchanged.")
    report["to_create"] = len(rows)
    report["to_update"] = 0
    _range_warnings(rows, fields, report, "static")
    report["preview"] = [{"row": r["line"], "tag": r["tag"], "type": next(t["name"] for t in types if t["id"] == r["belt_type_id"]),
                          "location": r["location"]} for r in rows[:8]]
    return rows


def _commit(conn, kind: str, rows: List[Dict], user_id: int, report: Dict) -> None:
    if kind == "belts":
        for r in rows:
            conn.execute("INSERT INTO belts (tag, belt_type_id, location, installed_on, static_json, created_at, created_by) "
                         "VALUES (?,?,?,?,?,?,?)", (r["tag"], r["belt_type_id"], r["location"][:120], r["installed_on"],
                                                    json.dumps(r["static"]), now_iso(), user_id))
        return

    active = reg.active_model_ids()
    types = {t["id"]: t for t in list_belt_types(conn)}
    by_belt: Dict[int, List[Dict]] = {}
    for r in rows:
        by_belt.setdefault(r["belt"]["id"], []).append(r)
    for belt_id, new_rows in by_belt.items():
        belt = new_rows[0]["belt"]
        new_hours = {r["hour"] for r in new_rows}
        prior = [{"hour": i["hour"], "readings": json.loads(i["readings_json"])} for i in conn.execute(
            "SELECT hour, readings_json FROM inspections WHERE belt_id = ?", (belt_id,)).fetchall() if i["hour"] not in new_hours]
        history = sorted(prior + [{"hour": r["hour"], "readings": r["readings"]} for r in new_rows], key=lambda h: h["hour"])
        model_id = active.get(belt["belt_type_id"])
        preds = {p["hour"]: p for p in reg.predict_inspections(belt, types[belt["belt_type_id"]], history, model_id)}
        for r in new_rows:
            store_inspection(conn, belt_id, r["hour"], r["readings"], preds[float(r["hour"])], model_id, user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Predictions vs. what actually happened
# ─────────────────────────────────────────────────────────────────────────────

def actual_remaining(belt: Dict, hour: float) -> Optional[Dict]:
    """Ground truth for an inspection of an ended belt: exact for a failure, a lower bound for a replacement."""
    if belt["status"] not in ("failed", "replaced") or belt.get("end_hour") is None or hour > belt["end_hour"]:
        return None
    return {"actual_rul": round(float(belt["end_hour"]) - float(hour), 0), "kind": belt["status"]}


def interval_hit(low, high, actual) -> Optional[bool]:
    return None if low is None or high is None else bool(low <= actual <= high)


def accuracy_report(conn) -> Dict:
    rows = conn.execute(
        "SELECT i.belt_id, i.hour, i.predicted_rul AS pred, i.predicted_low AS low, i.predicted_high AS high, i.model_id, "
        "b.tag, b.status, b.end_hour, b.end_reason, b.belt_type_id, t.name AS type_name "
        "FROM inspections i JOIN belts b ON b.id = i.belt_id JOIN belt_types t ON t.id = b.belt_type_id "
        "WHERE b.status IN ('failed','replaced') AND b.end_hour IS NOT NULL AND i.hour <= b.end_hour "
        "AND i.predicted_rul IS NOT NULL AND i.model_id IS NOT NULL ORDER BY i.belt_id, i.hour").fetchall()

    models: Dict[int, Dict] = {}
    belts_out, replaced_out = [], []
    last_per_belt: Dict[tuple, object] = {}
    for r in rows:
        actual = float(r["end_hour"]) - float(r["hour"])
        last_per_belt[(r["model_id"], r["belt_id"])] = r        # rows are ordered by hour -> the latest wins
        if r["status"] != "failed":
            continue
        m = models.setdefault(r["model_id"], {"belts": set(), "points": []})
        m["belts"].add(r["belt_id"])
        m["points"].append((actual, float(r["pred"]), r["low"], r["high"]))

    out_models = []
    for mid, m in sorted(models.items()):
        pts = m["points"]
        err = np.array([p - a for a, p, _, _ in pts])
        banded = [(a, lo, hi) for a, _, lo, hi in pts if lo is not None and hi is not None]
        buckets = []
        for lo_h, hi_h in HORIZONS:
            sel = [(a, p, lo, hi) for a, p, lo, hi in pts if lo_h <= a < hi_h]
            if not sel:
                continue
            e = np.array([p - a for a, p, _, _ in sel])
            b2 = [(a, lo, hi) for a, _, lo, hi in sel if lo is not None and hi is not None]
            buckets.append({"label": f"{lo_h:,}–{int(hi_h):,} h left" if hi_h != float("inf") else f"over {lo_h:,} h left",
                            "n": len(sel), "mae_hours": round(float(np.abs(e).mean()), 0), "bias_hours": round(float(e.mean()), 0),
                            "coverage": round(sum(1 for a, lo, hi in b2 if lo <= a <= hi) / len(b2), 3) if b2 else None})
        out_models.append({
            "model_id": mid, "failed_belts": len(m["belts"]), "points": len(pts),
            "mae_hours": round(float(np.abs(err).mean()), 0), "bias_hours": round(float(err.mean()), 0),
            "interval_coverage": round(sum(1 for a, lo, hi in banded if lo <= a <= hi) / len(banded), 3) if banded else None,
            "buckets": buckets,
            "low_sample": len(m["belts"]) < MIN_FAILED_FOR_CONFIDENCE,
        })

    for (mid, belt_id), r in last_per_belt.items():
        actual = float(r["end_hour"]) - float(r["hour"])
        item = {"belt_id": belt_id, "tag": r["tag"], "type": r["type_name"], "model_id": mid, "end_hour": r["end_hour"],
                "reason": r["end_reason"], "last_inspection_hour": r["hour"], "actual_rul": round(actual, 0),
                "predicted_rul": r["pred"], "low": r["low"], "high": r["high"]}
        if r["status"] == "failed":
            item["error_hours"] = round(float(r["pred"]) - actual, 0)
            item["in_interval"] = interval_hit(r["low"], r["high"], actual)
            belts_out.append(item)
        else:
            # replaced: the belt lasted AT LEAST `actual` more hours. An interval that ends before that was too pessimistic.
            item["too_pessimistic"] = None if r["high"] is None else bool(r["high"] < actual)
            replaced_out.append(item)

    belts_out.sort(key=lambda x: -x["end_hour"])
    return {
        "models": out_models, "failed_belts": belts_out[:200], "replaced_belts": replaced_out[:200],
        "total_failed_belts": len(belts_out), "total_replaced_belts": len(replaced_out),
        "min_failed_for_confidence": MIN_FAILED_FOR_CONFIDENCE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fleet history -> training dataset
# ─────────────────────────────────────────────────────────────────────────────

def export_fleet_dataset(conn, datasets_dir: str) -> Dict:
    """Write recorded fleet history as a canonical dataset an engineer can train on directly.

    failure_hour is filled ONLY for belts recorded as FAILED. Replaced and in-service belts are left
    empty so the survival model treats them as censored (they lasted at least until their last
    inspection) - do not fill failure_hour for replacements."""
    fields = sm.load_canonical()["fields"]
    belts = {r["id"]: row_to_dict(r) for r in conn.execute("SELECT * FROM belts").fetchall()}
    types = {t["id"]: t["name"] for t in list_belt_types(conn)}
    rows, used = [], {"failed": 0, "replaced": 0, "in_service": 0}
    for ins in conn.execute("SELECT * FROM inspections ORDER BY belt_id, hour").fetchall():
        b = belts[ins["belt_id"]]
        if b["end_hour"] is not None and ins["hour"] > b["end_hour"]:
            continue
        readings = {k: v for k, v in json.loads(ins["readings_json"]).items() if k in fields}   # extras (x_*) can't be exported
        rows.append({"belt_id": b["tag"], "belt_type": types[b["belt_type_id"]], "hour": ins["hour"],
                     **{k: v for k, v in b["static"].items() if k in fields}, **readings,
                     "failure_hour": b["end_hour"] if b["status"] == "failed" else None, "_status": b["status"]})
    if not rows:
        raise reg.TrainingError("There are no inspections recorded yet.")
    df = pd.DataFrame(rows)
    for status, n in df.groupby("_status")["belt_id"].nunique().items():
        used[status] = int(n)
    df = df.drop(columns=["_status"])
    df = df.dropna(axis=1, how="all")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    name, n = f"fleet_records_{stamp}.csv", 2
    while os.path.exists(os.path.join(datasets_dir, name)):
        name, n = f"fleet_records_{stamp}_{n}.csv", n + 1
    df.to_csv(os.path.join(datasets_dir, name), index=False)
    return {"filename": name, "rows": int(len(df)), "belts": int(df["belt_id"].nunique()), **{f"{k}_belts": v for k, v in used.items()}}
