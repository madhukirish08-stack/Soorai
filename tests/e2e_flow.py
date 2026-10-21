"""End-to-end backend check (temp instance dir, no server needed): train AFT + regression, activate, compare, predict.
Run from the repo root:  python tests/e2e_flow.py <dataset.csv>
Make a dataset:          python training/generate_synthetic_vbelts.py --censor-frac 0.3 --output /tmp/s.csv"""
import json, os, shutil, sys, tempfile, time
ROOT = os.getcwd()
inst = tempfile.mkdtemp(prefix="beltrul_")
os.environ.update(BELT_RUL_DEV_MODE="1", BELT_RUL_INSTANCE_DIR=inst)
sys.path.insert(0, os.path.join(ROOT, "backend"))
from fastapi.testclient import TestClient
import api, auth
from db import init_db

init_db()
auth.create_user("eng", "correct-horse-battery", "engineer")
c = TestClient(api.app)
r = c.post("/api/auth/login", json={"username": "eng", "password": "correct-horse-battery"})
assert r.status_code == 200, r.text
csvs = sys.argv[1:]
from config import DATASETS_DIR
for f in csvs:
    shutil.copy(f, os.path.join(DATASETS_DIR, os.path.basename(f)))
types = c.get("/api/belt-types").json()["types"]
print("belt types:", [(t["id"], t["name"]) for t in types])


def train(files, method):
    r = c.post("/api/models/train", json={"datasets": [{"filename": os.path.basename(f)} for f in files], "method": method})
    assert r.status_code == 202, r.text
    mid = r.json()["id"]
    t0 = time.time()
    while True:
        m = c.get(f"/api/models/{mid}").json()
        if m["status"] != "training":
            break
        time.sleep(1)
    print(f"v{mid} {method} -> {m['status']} in {time.time()-t0:.0f}s", m.get("error") or "")
    return mid, m


ids = {}
for method in ("aft", "regression"):
    mid, m = train(csvs[:1], method)
    assert m["status"] == "ready", m
    ids[method] = mid
    i = m["info"]
    print(json.dumps({k: i[k] for k in ("method", "aft", "n_belts", "n_events", "n_censored_belts", "n_test_belts", "n_rows")}, default=str))
    print("metrics", json.dumps(i["metrics"]))
    print("rul_source", i["datasets"][0]["rul_source"]["message"])
    top = list(i["feature_importance"].items())[:5]
    print("top features", top)

print("activate", c.post(f"/api/models/{ids['aft']}/activate", json={}).json())
cmp = c.get(f"/api/models/compare?a={ids['aft']}&b={ids['regression']}").json()
print("compare comparable:", cmp["comparable"], cmp.get("reason"))
for k, v in cmp["models"].items():
    print(" ", k, json.dumps(v.get("shared_metrics")))

# predict a new belt
tid = types[0]["id"]
b = c.post("/api/belts", json={"tag": "T-1", "belt_type_id": tid, "static": {"driver_pulley_dia_mm": 150, "driven_pulley_dia_mm": 300, "driver_rpm": 1440, "pitch_length_mm": 1800, "num_pulleys": 2}})
assert b.status_code == 200, b.text
bid = b.json()["id"]
for hr, el in ((0, 0.05), (1000, 0.4), (2000, 0.9)):
    d = c.post(f"/api/belts/{bid}/inspections", json={"hour": hr, "readings": {"elongation_pct": el, "wear_pct": el * 8, "crack_depth_mm": 0.0, "load_pct": 80}})
    assert d.status_code == 200, d.text
p = d.json()["predictions"][-1]
print("prediction:", {k: p[k] for k in ("hour", "rul_hours", "rul_low", "rul_high", "interval_level", "tier")})
print("fleet:", [(x["tag"], x["prediction"]["rul_hours"], x["prediction"]["rul_low"], x["prediction"]["rul_high"]) for x in c.get("/api/belts").json()])
# regression version -> no interval
c.post(f"/api/models/{ids['regression']}/activate", json={})
p = c.get(f"/api/belts/{bid}").json()["predictions"][-1]
print("regression prediction:", {k: p[k] for k in ("rul_hours", "rul_low", "rul_high")})
shutil.rmtree(inst, ignore_errors=True)
print("E2E OK")
