"""Phase 4 regression test: bulk import, prediction-vs-actual backtest, fleet -> dataset, DB migration.

Run from the repo root:   python tests/test_phase4.py
Uses a temp instance dir and the in-process TestClient (no server needed). Prints 'PHASE 4 OK' on success.
"""
import os, shutil, sqlite3, subprocess, sys, tempfile, time

ROOT = os.getcwd()
inst = tempfile.mkdtemp(prefix="beltrul_p4_")
os.environ.update(BELT_RUL_DEV_MODE="1", BELT_RUL_INSTANCE_DIR=inst)
sys.path.insert(0, os.path.join(ROOT, "backend"))

import pandas as pd

# ── Migration: a database created BEFORE Phase 4 (no interval columns) must upgrade in place ──────────
old = sqlite3.connect(os.path.join(inst, "belt_rul.db"))
old.executescript("""CREATE TABLE inspections (id INTEGER PRIMARY KEY, belt_id INTEGER NOT NULL, hour REAL NOT NULL,
    readings_json TEXT NOT NULL, predicted_rul REAL, health_score REAL, model_id INTEGER, created_at TEXT NOT NULL,
    created_by INTEGER, UNIQUE (belt_id, hour));
    INSERT INTO inspections (belt_id, hour, readings_json, predicted_rul, created_at) VALUES (1, 100, '{}', 500, 'x');""")
old.commit(); old.close()

from fastapi.testclient import TestClient
import api, auth
from db import init_db, get_conn
from config import DATASETS_DIR

init_db()
with get_conn() as conn:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(inspections)")}
    kept = conn.execute("SELECT predicted_rul FROM inspections").fetchone()[0]
assert {"predicted_low", "predicted_high"} <= cols and kept == 500, "migration lost data or columns"
with get_conn() as conn:
    conn.execute("DELETE FROM inspections")
print("migration ok")

auth.create_user("eng", "correct-horse-battery", "engineer")
auth.create_user("plan", "tr0ub4dor-and-3-planner", "planner")
eng, plan = TestClient(api.app), TestClient(api.app)
assert eng.post("/api/auth/login", json={"username": "eng", "password": "correct-horse-battery"}).status_code == 200
assert plan.post("/api/auth/login", json={"username": "plan", "password": "tr0ub4dor-and-3-planner"}).status_code == 200


def ok(r, code=200):
    assert r.status_code == code, (r.status_code, r.text[:600])
    return r.json()


def upload(client, kind, df_or_text, dry_run, default_type=None):
    text = df_or_text if isinstance(df_or_text, str) else df_or_text.to_csv(index=False)
    data = {"dry_run": "true" if dry_run else "false"}
    if default_type:
        data["default_belt_type_id"] = str(default_type)
    return client.post(f"/api/import/{kind}", files={"file": ("x.csv", text.encode(), "text/csv")}, data=data)


# ── Train and activate a model so imported inspections get predictions ─────────────────────────────────
synth = os.path.join(inst, "synth.csv")
subprocess.run([sys.executable, os.path.join(ROOT, "training", "generate_synthetic_vbelts.py"), "--belts", "90",
                "--step", "250", "--seed", "11", "--output", synth], check=True, capture_output=True)
src = pd.read_csv(synth)
shutil.copy(synth, os.path.join(DATASETS_DIR, "synth.csv"))
mid = ok(eng.post("/api/models/train", json={"datasets": [{"filename": "synth.csv"}]}), 202)["id"]
for _ in range(180):
    m = ok(eng.get(f"/api/models/{mid}"))
    if m["status"] != "training":
        break
    time.sleep(1)
assert m["status"] == "ready", m.get("error")
ok(eng.post(f"/api/models/{mid}/activate", json={}))
print(f"model v{mid} active")

# ── Templates & permissions ────────────────────────────────────────────────────────────────────────────
tpl = plan.get("/api/import/template/inspections")
assert tpl.status_code == 200 and tpl.text.startswith("belt_tag,hour,"), tpl.text[:80]
assert plan.get("/api/import/template/belts").text.startswith("belt_tag,belt_type,")
assert plan.get("/api/import/template/nope").status_code == 404
assert plan.get("/api/accuracy").status_code == 403
assert plan.post("/api/datasets/from-fleet").status_code == 403

# ── Belts import: validation blocks the commit, existing tags are left alone ───────────────────────────
tags = sorted(src.belt_id.unique())[:50]
first = src.drop_duplicates("belt_id").set_index("belt_id").loc[tags].reset_index()
belts = pd.DataFrame({"Belt Tag": first.belt_id, "Section": first.belt_type, "Location": "Line " + first.belt_id,
                      "pitch_length_mm": first.pitch_length_mm, "driver_pulley_dia_mm": first.driver_pulley_dia_mm,
                      "driven_pulley_dia_mm": first.driven_pulley_dia_mm, "num_pulleys": first.num_pulleys,
                      "driver_rpm": first.driver_rpm, "Installed": "2025-01-31", "Load Pct": 80})          # last one: a measurement, belongs in the other file
bad = belts.copy()
bad.loc[0, "Section"] = "ZZZ-unknown"
bad.loc[1, "Belt Tag"] = bad.loc[2, "Belt Tag"]
bad.loc[3, "pitch_length_mm"] = "1,800"
bad.loc[5, "Installed"] = "2025-13-45"                       # right shape, not a real date
rep = ok(upload(plan, "belts", bad, dry_run=False))
assert rep["committed"] is False and rep["error_count"] == 4, rep["errors"]
assert any("not a valid date" in e["message"] for e in rep["errors"])
assert len(ok(plan.get("/api/belts"))) == 0, "a file with errors must import NOTHING"
assert any("Load Pct" in c["source"] and c["field"] is None for c in rep["columns"]), "inspection column must be ignored in a belts file"
rep = ok(upload(plan, "belts", belts, dry_run=True))
assert rep["committed"] is False and rep["to_create"] == 50 and not rep["errors"], rep
assert len(ok(plan.get("/api/belts"))) == 0, "dry run must not write"
rep = ok(upload(plan, "belts", belts, dry_run=False))
assert rep["committed"] and len(ok(plan.get("/api/belts"))) == 50
rep = ok(upload(plan, "belts", belts, dry_run=False))
assert rep["to_create"] == 0 and any("already registered" in w for w in rep["warnings"])
print("belts import ok")

# ── Inspections import ─────────────────────────────────────────────────────────────────────────────────
use = src[src.belt_id.isin(tags)]
# Belts that are still running / replaced early stop BEFORE their failure limits (tags[30:]); the first 30 run to failure.
cutoff = use.belt_id.map(lambda t: 1.0 if t in tags[:30] else 0.6) * use.failure_hour
use = use[use.hour <= cutoff]
insp = pd.DataFrame({"belt_tag": use.belt_id, "Operating Hours": use.hour, "Elongation": use.elongation_pct,
                     "Side wear": use.wear_pct, "crack_depth_mm": use.crack_depth_mm, "load_pct": use.load_pct,
                     "temperature_c": use.temperature_c, "slip_pct": use.slip_pct, "rul_hours": use.rul_hours,
                     "failure_hour": use.failure_hour})
bad = insp.copy().astype({"load_pct": object})
bad.iloc[0, bad.columns.get_loc("belt_tag")] = "NOT-REGISTERED"
bad.iloc[1, bad.columns.get_loc("load_pct")] = "8o"
bad.iloc[2, bad.columns.get_loc("crack_depth_mm")] = -1
bad.iloc[4] = bad.iloc[3]                                  # exact duplicate (belt, hour)
rep = ok(upload(plan, "inspections", bad, dry_run=True))
msgs = " | ".join(e["message"] for e in rep["errors"])
assert rep["error_count"] == 4 and "not registered" in msgs and "plain number" in msgs and "negative" in msgs and "twice" in msgs, msgs
ignored = set(rep["ignored_columns"])
assert {"rul_hours", "failure_hour"} <= ignored, ignored
assert {c["source"]: c["field"] for c in rep["columns"]}["Operating Hours"] == "hour"
assert {c["source"]: c["field"] for c in rep["columns"]}["Side wear"] == "wear_pct"
assert {c["source"]: c["field"] for c in ok(upload(plan, "inspections", insp.rename(columns={"temperature_c": "Temp (C)"}), dry_run=True))["columns"]}["Temp (C)"] == "temperature_c"
fahrenheit = insp.copy(); fahrenheit["temperature_c"] = fahrenheit["temperature_c"] * 9 / 5 + 32 + 60
rep = ok(upload(plan, "inspections", fahrenheit, dry_run=True))
assert any("temperature_c" in w and "outside the expected" in w for w in rep["warnings"]), rep["warnings"]
rep = ok(upload(plan, "inspections", insp, dry_run=False))
assert rep["committed"] and rep["to_create"] == len(use) and rep["belts_affected"] == 50, rep
with get_conn() as conn:
    n, with_band, with_model = conn.execute(
        "SELECT COUNT(*), SUM(predicted_low IS NOT NULL), SUM(model_id IS NOT NULL) FROM inspections").fetchone()
assert n == len(use) and with_band == n and with_model == n, (n, with_band, with_model)
rep = ok(upload(plan, "inspections", insp, dry_run=True))
assert rep["to_update"] == len(use) and rep["to_create"] == 0
print(f"inspections import ok ({n} rows, every one stored with model version + interval)")

# ── End of life: validation + backtest ─────────────────────────────────────────────────────────────────
fleet_ = {b["tag"]: b for b in ok(plan.get("/api/belts"))}
life = src.groupby("belt_id").failure_hour.first()
last_hour = use.groupby("belt_id").hour.max()
t0 = tags[0]
early = plan.put(f"/api/belts/{fleet_[t0]['id']}/end-of-life", json={"status": "failed", "hour": float(last_hour[t0]) - 500})
assert early.status_code == 400 and "cannot have failed" in early.json()["detail"], early.text
failed_tags, replaced_tags = tags[:30], tags[30:36]
for t in failed_tags:
    ok(plan.put(f"/api/belts/{fleet_[t]['id']}/end-of-life", json={"status": "failed", "hour": float(life[t]), "reason": "Rib cracking"}))
for t in replaced_tags:
    ok(plan.put(f"/api/belts/{fleet_[t]['id']}/end-of-life", json={"status": "replaced", "hour": float(last_hour[t]) + 100, "reason": "Planned"}))

acc = ok(eng.get("/api/accuracy"))
assert acc["total_failed_belts"] == 30 and acc["total_replaced_belts"] == 6, (acc["total_failed_belts"], acc["total_replaced_belts"])
m = acc["models"][0]
assert m["model_id"] == mid and m["failed_belts"] == 30 and m["low_sample"] is False
assert m["interval_coverage"] is not None and 0.0 <= m["interval_coverage"] <= 1.0 and m["mae_hours"] >= 0
assert m["buckets"] and sum(b["n"] for b in m["buckets"]) == m["points"]
assert all("error_hours" in b and "in_interval" in b for b in acc["failed_belts"])
assert all("too_pessimistic" in b for b in acc["replaced_belts"]) and not any("error_hours" in b for b in acc["replaced_belts"])
print(f"backtest ok: {m['failed_belts']} failed belts, MAE {m['mae_hours']:,.0f} h, bias {m['bias_hours']:+,.0f} h, "
      f"interval coverage {m['interval_coverage']:.0%} | replaced-early belts counted separately: {acc['total_replaced_belts']}")

detail = ok(plan.get(f"/api/belts/{fleet_[failed_tags[0]]['id']}"))
done = [i for i in detail["inspections"] if i["actual_rul"] is not None]
assert done and all(i["actual_kind"] == "failed" for i in done)
assert done[-1]["actual_rul"] == 0 and done[0]["actual_rul"] == float(life[failed_tags[0]]) - done[0]["hour"]
rd = ok(plan.get(f"/api/belts/{fleet_[replaced_tags[0]]['id']}"))
assert all(i["actual_kind"] == "replaced" and i["in_interval"] is None for i in rd["inspections"]), "replacement is only a lower bound"
live = ok(plan.get(f"/api/belts/{fleet_[tags[40]]['id']}"))
assert all(i["actual_rul"] is None for i in live["inspections"])
print("ground truth on belt pages ok")

# ── Fleet history -> training dataset -> trained model (closing the loop) ──────────────────────────────
assert plan.post("/api/datasets/from-fleet").status_code == 403
exp = ok(eng.post("/api/datasets/from-fleet"))
assert exp["failed_belts"] == 30 and exp["replaced_belts"] == 6 and exp["in_service_belts"] == 14 and exp["belts"] == 50, exp
ds = {d["filename"]: d for d in ok(eng.get("/api/datasets"))}
assert ds[exp["filename"]]["mapping_status"] == "not_needed", "exported dataset must train without a mapping step"
exported = pd.read_csv(os.path.join(DATASETS_DIR, exp["filename"]))
assert exported.groupby("belt_id").failure_hour.first().notna().sum() == 30, "failure_hour only for FAILED belts"
assert exported[exported.belt_id.isin(replaced_tags)].failure_hour.isna().all(), "replaced belts must stay censored"
mid2 = ok(eng.post("/api/models/train", json={"datasets": [{"filename": exp["filename"]}], "notes": "from fleet"}), 202)["id"]
for _ in range(180):
    m2 = ok(eng.get(f"/api/models/{mid2}"))
    if m2["status"] != "training":
        break
    time.sleep(1)
assert m2["status"] == "ready", m2.get("error")
assert m2["info"]["n_censored_belts"] >= 15, {k: m2["info"][k] for k in ("n_events", "n_censored_belts", "rul_source")}
print(f"fleet dataset trained: v{mid2}, {m2['info']['n_events']} events + {m2['info']['n_censored_belts']} censored belts, "
      f"concordance {m2['info']['metrics'].get('c_index')}")
print("PHASE 4 OK")
shutil.rmtree(inst, ignore_errors=True)
