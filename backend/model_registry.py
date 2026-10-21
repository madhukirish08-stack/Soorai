"""
Model Registry
==============
Every training run creates a new, immutable model VERSION (instance/models/<id>/model.ubj
+ metadata in SQLite). Nothing is overwritten. An engineer activates a version per belt
type; predictions always use the active version for the belt's type, and older versions
stay available for rollback.

Design decisions (keep):
- XGBoost only (beat the former XGBoost + Random Forest ensemble on held-out belts).
- No fine-tuning. "Add data" = train a new version on the old datasets + the new one.
- Test belts are chosen by a HASH of the belt key, so a belt is a test belt in every
  version. That makes versions comparable on the same belts, and no belt ever appears
  in both train and test (no leakage).
- Models are stored in XGBoost's native format, never pickle (pickle executes code on load).
- Survival analysis (method "aft", the default): XGBoost's survival:aft objective. Belts with a known
  end of life are events; belts still running / replaced early are RIGHT-CENSORED (we only know they
  lasted at least their last hour). The model gives a median RUL plus an 80 % interval from the AFT
  distribution quantiles. method "regression" is the legacy point model (squared error on events only,
  no interval); versions trained before survival support have no "method" and are treated as regression.
"""

import hashlib
import json
import os
import threading
from statistics import NormalDist
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

import schema_mapper as sm
from config import DATASETS_DIR, MAPPINGS_DIR, MODELS_DIR
from db import get_conn, list_belt_types, match_belt_type, now_iso, row_to_dict

TARGET = "rul_hours"
MIN_FEATURE_COVERAGE = 0.5   # sparser columns are dropped instead of imputed
TEST_PCT = 20                # % of belts held out for testing
MIN_BELTS = 5
TYPE_PREFIX = "type_"        # one-hot belt-type features when a model covers several types
XGB_PARAMS = dict(
    n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.85,
    colsample_bytree=0.85, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)

# Native-API twin of XGB_PARAMS (xgb.train does not take the sklearn names).
_BOOSTER_PARAMS = dict(
    max_depth=XGB_PARAMS["max_depth"], learning_rate=XGB_PARAMS["learning_rate"], subsample=XGB_PARAMS["subsample"],
    colsample_bytree=XGB_PARAMS["colsample_bytree"], min_child_weight=XGB_PARAMS["min_child_weight"],
    reg_alpha=XGB_PARAMS["reg_alpha"], reg_lambda=XGB_PARAMS["reg_lambda"], seed=XGB_PARAMS["random_state"],
    nthread=-1, tree_method="hist",
)
METHODS = ("aft", "regression")
INTERVAL_LEVEL = 0.8                     # shown as an 80 % interval: [10 %, 90 %] quantiles of the AFT distribution
_Z = NormalDist().inv_cdf(1 - (1 - INTERVAL_LEVEL) / 2)
AFT_SIGMAS = (0.3, 0.5, 0.8)             # candidate scales of log-life noise; picked on a validation split
VAL_PCT = 20                             # % of TRAINING belts used to pick sigma / number of trees
MIN_RUL_H = 1.0                          # AFT works in log-space: a RUL of exactly 0 is not allowed
MIN_EVENTS = MIN_BELTS                   # belts with a known end of life needed to train

_train_lock = threading.Lock()
_booster_cache: Dict[int, xgb.Booster] = {}


class TrainingError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def dataset_mapping(filename: str, columns) -> Optional[List[Dict]]:
    """Approved mapping, identity mapping for fully canonical files, else None."""
    saved = sm.load_mapping(MAPPINGS_DIR, filename, columns)
    if saved:
        return saved["mapping"]
    if sm.is_fully_canonical(columns):
        return sm.identity_mapping(columns)
    return None


def load_dataset(spec: Dict, types: List[Dict]):
    """spec = {filename, belt_type_id?, mapping?} -> (canonical df with rul_hours, rul_info, mapping)."""
    fn = os.path.basename(spec["filename"])
    path = os.path.join(DATASETS_DIR, fn)
    if not os.path.exists(path):
        raise TrainingError(f"Dataset '{fn}' no longer exists.")
    raw = pd.read_csv(path)
    mapping = spec.get("mapping") or dataset_mapping(fn, raw.columns)
    if mapping is None:
        raise TrainingError(f"'{fn}' needs an approved column mapping before training.")
    df = sm.apply_mapping(raw, mapping)

    if "belt_type" in df.columns:
        ids = df["belt_type"].map(lambda v: match_belt_type(types, v))
        unknown = sorted(df.loc[ids.isna(), "belt_type"].astype(str).unique())
        if unknown:
            raise TrainingError(f"'{fn}': belt-type values not configured: {unknown[:10]}. "
                                "Add them as a belt type (or an alias of one) first.")
        df["belt_type_id"] = ids.astype(int)
        df = df.drop(columns=["belt_type"])
    elif spec.get("belt_type_id"):
        df["belt_type_id"] = int(spec["belt_type_id"])
    else:
        raise TrainingError(f"'{fn}' has no belt-type column - choose its belt type.")

    thresholds = {t["id"]: t["thresholds"] for t in types}
    missing = set(df["belt_type_id"].unique()) - set(thresholds)
    if missing:
        raise TrainingError(f"'{fn}': unknown belt type id(s) {sorted(missing)}.")

    # Belt IDs are only unique within one file - namespace them by dataset.
    if "belt_id" in df.columns:
        df["belt_id"] = fn + "::" + df["belt_id"].astype(str)
    row_thr = pd.DataFrame([thresholds[t] for t in df["belt_type_id"]], index=df.index)
    df, info = sm.derive_rul(df, row_thr)
    df = df[df[TARGET].notna()].copy()
    if "belt_id" not in df.columns:
        df["belt_id"] = [f"{fn}::row{i}" for i in df.index]
        info["row_keys"] = True
    return df, info, mapping


def is_test_belt(key: str) -> bool:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100 < TEST_PCT


def build_X(df: pd.DataFrame, feature_cols: List[str], medians: Dict[str, float]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c.startswith(TYPE_PREFIX):
            X[c] = (df["belt_type_id"] == int(c[len(TYPE_PREFIX):])).astype(float)
        elif c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            X[c] = np.nan
    return X.fillna({c: medians.get(c, 0.0) for c in feature_cols})[feature_cols]


def evaluate(y_true, y_pred) -> Dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    near_cut = float(np.quantile(y_true, 0.25))
    near = y_true <= near_cut
    return {
        "mae_hours": round(float(np.abs(err).mean()), 1),
        "rmse_hours": round(float(np.sqrt((err ** 2).mean())), 1),
        "r2_score": round(1 - float((err ** 2).sum()) / ss_tot, 4) if ss_tot > 0 else None,
        # Positive bias = model predicts MORE life than reality (optimistic, the dangerous side).
        "bias_hours": round(float(err.mean()), 1),
        "near_failure_mae_hours": round(float(np.abs(err[near]).mean()), 1),
        "near_failure_cutoff_hours": round(near_cut, 0),
        "n_rows": int(len(y_true)),
    }


def concordance_index(time, event, score) -> Optional[float]:
    """Harrell's C for right-censored data. score = predicted LIFE (higher = lasts longer).
    A pair is comparable when the shorter observed time is an event; censored belts only
    count as 'lasted at least this long'. 0.5 = random ordering, 1.0 = perfect."""
    t, e, s = (np.asarray(a, float) for a in (time, event, score))
    conc = total = 0.0
    for i in np.flatnonzero(e > 0):
        later = t > t[i]
        total += later.sum()
        conc += (s[later] > s[i]).sum() + 0.5 * (s[later] == s[i]).sum()
    return round(conc / total, 4) if total else None


def score_belts(df: pd.DataFrame, median, sigma: Optional[float]) -> Dict:
    """All held-out metrics for one set of test rows (df has rul_hours + rul_censored).
    MAE / RMSE / R2 / bias are over EVENT rows only - censored rows have no true RUL, only a
    lower bound. The concordance index uses every row (censored ones as lower bounds).
    c_index = across all test rows; c_index_belt = first inspection of each belt ranks the belts by
    total life (the harder, more honest question). interval_coverage = share of event rows whose true
    RUL fell inside the 80 % interval (should be ~80 %)."""
    median = np.asarray(median, float)
    ev = ~df["rul_censored"].to_numpy(bool)
    y = df[TARGET].to_numpy(float)
    m = evaluate(y[ev], median[ev])
    m["n_censored_rows"] = int((~ev).sum())
    m["c_index"] = concordance_index(y, ev, median)
    first = df.assign(_pred=median).sort_values("hour").groupby("belt_id", sort=False).head(1)
    f_ev = ~first["rul_censored"].to_numpy(bool)
    m["c_index_belt"] = concordance_index(first["hour"] + first[TARGET], f_ev, first["hour"] + first["_pred"]) \
        if len(first) > 1 else None
    if sigma is not None:
        lo, hi = median * np.exp(-_Z * sigma), median * np.exp(_Z * sigma)
        m["interval_level"] = INTERVAL_LEVEL
        m["interval_coverage"] = round(float(((y[ev] >= lo[ev]) & (y[ev] <= hi[ev])).mean()), 3)
        m["interval_mean_width_hours"] = round(float((hi[ev] - lo[ev]).mean()), 0)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def start_training(dataset_specs: List[Dict], notes: str, user_id: int, method: str = "aft") -> int:
    """Create a model version record and train it in a background thread."""
    if not dataset_specs:
        raise TrainingError("Select at least one dataset.")
    if method not in METHODS:
        raise TrainingError(f"Unknown method '{method}'.")
    if not _train_lock.acquire(blocking=False):
        raise TrainingError("Another model is being trained. Wait for it to finish.")
    try:
        with get_conn() as conn:
            model_id = conn.execute(
                "INSERT INTO models (status, notes, created_at, created_by) VALUES ('training', ?, ?, ?)",
                (notes or "", now_iso(), user_id)).lastrowid
    except Exception:
        _train_lock.release()
        raise
    threading.Thread(target=_train_worker, args=(model_id, dataset_specs, method), daemon=True).start()
    return model_id


def _train_worker(model_id: int, dataset_specs: List[Dict], method: str) -> None:
    try:
        info = _train(model_id, dataset_specs, method)
        with get_conn() as conn:
            conn.execute("UPDATE models SET status='ready', info_json=? WHERE id=?", (json.dumps(info), model_id))
    except Exception as e:  # stored on the version so the engineer sees why
        msg = str(e) if isinstance(e, (TrainingError, ValueError)) else f"Training failed: {e}"
        with get_conn() as conn:
            conn.execute("UPDATE models SET status='failed', error=? WHERE id=?", (msg, model_id))
    finally:
        _train_lock.release()


def booster_median(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """AFT: predict() returns exp(margin) = the MEDIAN life (normal noise on log-life). Regression: the point value."""
    return booster.predict(xgb.DMatrix(X))


def _aft_matrix(X: pd.DataFrame, d: pd.DataFrame) -> xgb.DMatrix:
    """Events have lower == upper == true RUL; right-censored rows have [lower bound, +inf)."""
    lower = np.maximum(d[TARGET].to_numpy(float), MIN_RUL_H)
    upper = np.where(d["rul_censored"].to_numpy(bool), np.inf, lower)
    dm = xgb.DMatrix(X)
    dm.set_float_info("label_lower_bound", lower)
    dm.set_float_info("label_upper_bound", upper)
    return dm


def _fit_aft(X: pd.DataFrame, d: pd.DataFrame):
    """Pick the AFT noise scale (sigma) and tree count on a hash-based validation split of the TRAINING
    belts (test belts are never touched), then refit on all training belts. Returns (booster, sigma, info)."""
    n_rounds = XGB_PARAMS["n_estimators"]
    base = {**_BOOSTER_PARAMS, "objective": "survival:aft", "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal"}
    keys = d["belt_id"].astype(str)
    val = keys.map(lambda k: int(hashlib.md5(("val:" + k).encode()).hexdigest()[:8], 16) % 100 < VAL_PCT).to_numpy()
    cens = d["rul_censored"].to_numpy(bool)
    tuned = val.sum() > 0 and (~val).sum() > 0 and (val & ~cens).sum() >= 3 and (~val & ~cens).sum() >= 3
    if not tuned:
        sigma, rounds = 0.5, 300
    else:
        dfit, dval = _aft_matrix(X[~val], d[~val]), _aft_matrix(X[val], d[val])
        best = None
        for sg in AFT_SIGMAS:
            res = {}
            b = xgb.train({**base, "aft_loss_distribution_scale": sg}, dfit, n_rounds, evals=[(dval, "val")],
                          early_stopping_rounds=40, evals_result=res, verbose_eval=False)
            loss = min(res["val"]["aft-nloglik"])
            if best is None or loss < best[0]:
                best = (loss, sg, b.best_iteration + 1)
        _, sigma, rounds = best
    booster = xgb.train({**base, "aft_loss_distribution_scale": sigma}, _aft_matrix(X, d), rounds)
    return booster, sigma, {"distribution": "normal", "sigma": sigma, "rounds": int(rounds),
                            "interval_level": INTERVAL_LEVEL, "sigma_tuned": bool(tuned)}


def _fit_regression(X: pd.DataFrame, d: pd.DataFrame) -> xgb.Booster:
    """Legacy point model: squared error on belts with a known end of life (censored rows are dropped)."""
    ev = ~d["rul_censored"].to_numpy(bool)
    dm = xgb.DMatrix(X[ev], label=d.loc[ev, TARGET].to_numpy(float))
    return xgb.train({**_BOOSTER_PARAMS, "objective": "reg:squarederror"}, dm, XGB_PARAMS["n_estimators"])


def _train(model_id: int, dataset_specs: List[Dict], method: str = "aft") -> Dict:
    schema = sm.load_canonical()
    with get_conn() as conn:
        types = list_belt_types(conn)

    frames, used = [], []
    for spec in dataset_specs:
        df, rul_info, mapping = load_dataset(spec, types)
        frames.append(df)
        used.append({"filename": os.path.basename(spec["filename"]), "belt_type_id": spec.get("belt_type_id"),
                     "mapping": mapping, "rul_source": rul_info, "rows": int(len(df))})
    df = pd.concat(frames, ignore_index=True)

    type_ids = sorted(int(t) for t in df["belt_type_id"].unique())
    canonical = sm.fields_by_role(schema, "feature", "degradation", "time")
    extras = sorted(c for c in df.columns if c.startswith(sm.EXTRA_PREFIX))
    base = [c for c in canonical + extras if c in df.columns and df[c].notna().mean() >= MIN_FEATURE_COVERAGE]
    if not base:
        raise TrainingError("No usable feature columns after mapping.")
    feature_cols = base + ([f"{TYPE_PREFIX}{t}" for t in type_ids] if len(type_ids) > 1 else [])
    medians = {c: float(df[c].median()) for c in base}

    keys = df["belt_id"].astype(str)
    censored = df["rul_censored"].to_numpy(bool)
    n_belts = keys.nunique()
    n_events = keys[~censored].nunique()
    if n_events < MIN_EVENTS:
        raise TrainingError(f"Only {n_events} belts with a recorded end of life (failure/replacement) - at least "
                            f"{MIN_EVENTS} are needed. Belts that are still running only tell the model they lasted "
                            "at least this long; they cannot be the only data.")
    test_mask = keys.map(is_test_belt)
    if not test_mask[~censored].any():   # tiny datasets: hold out the lowest-hash belt that has an event
        first = min(keys[~censored].unique(), key=lambda k: hashlib.md5(k.encode()).hexdigest())
        test_mask = keys == first
    if keys[~test_mask & ~censored].nunique() < 2:
        raise TrainingError("Too few belts with a recorded end of life are left for training - add more belts.")

    X = build_X(df, feature_cols, medians)
    train_df = df[~test_mask]
    if method == "aft":
        booster, sigma, aft_info = _fit_aft(X[~test_mask], train_df)
        median = booster_median(booster, X[test_mask])
    else:
        booster, sigma, aft_info = _fit_regression(X[~test_mask], train_df), None, {}
        median = np.maximum(booster_median(booster, X[test_mask]), 0.0)
    metrics = score_belts(df[test_mask], median, sigma)

    out_dir = os.path.join(MODELS_DIR, str(model_id))
    os.makedirs(out_dir, exist_ok=True)
    booster.save_model(os.path.join(out_dir, "model.ubj"))

    gain = booster.get_score(importance_type="gain")
    total_gain = sum(gain.values()) or 1.0
    importance = dict(sorted(((c, float(gain.get(c, 0.0)) / total_gain) for c in feature_cols), key=lambda kv: -kv[1]))
    estimated = any(u["rul_source"].get("estimated") for u in used)
    return {
        "datasets": used,
        "belt_type_ids": type_ids,
        "feature_cols": feature_cols,
        "medians": medians,
        "feature_stats": {c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": medians[c]} for c in base},
        "feature_importance": importance,
        "metrics": metrics,
        "rul_estimated": estimated,
        "test_split": "by_row" if any(u["rul_source"].get("row_keys") for u in used) else "by_belt",
        "n_belts": int(n_belts),
        "method": method,
        "aft": aft_info or None,
        "n_events": int(n_events),
        "n_censored_belts": int(n_belts - n_events),
        "n_test_belts": int(keys[test_mask].nunique()),
        "n_rows": int(len(df)),
        "thresholds_used": {str(t["id"]): t["thresholds"] for t in types if t["id"] in type_ids},
        "thresholds_unconfirmed": [t["name"] for t in types if t["id"] in type_ids and not t["thresholds_confirmed"]],
        "xgb_params": XGB_PARAMS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registry access
# ─────────────────────────────────────────────────────────────────────────────

def get_model(model_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT m.*, u.username AS created_by_name FROM models m "
                           "LEFT JOIN users u ON u.id = m.created_by WHERE m.id = ?", (model_id,)).fetchone()
        if row is None:
            return None
        m = row_to_dict(row)
        m["active_for"] = [r[0] for r in conn.execute(
            "SELECT belt_type_id FROM active_models WHERE model_id = ?", (model_id,)).fetchall()]
    return m


def list_models() -> List[Dict]:
    with get_conn() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM models ORDER BY id DESC").fetchall()]
    return [get_model(i) for i in ids]


def load_booster(model_id: int) -> xgb.Booster:
    if model_id not in _booster_cache:
        model = xgb.Booster()
        model.load_model(os.path.join(MODELS_DIR, str(model_id), "model.ubj"))
        _booster_cache[model_id] = model
    return _booster_cache[model_id]


def activate(model_id: int, user_id: int, belt_type_ids: Optional[List[int]] = None) -> List[int]:
    m = get_model(model_id)
    if m is None or m["status"] != "ready":
        raise TrainingError("Only a successfully trained version can be activated.")
    covered = m["info"]["belt_type_ids"]
    targets = belt_type_ids or covered
    if not set(targets) <= set(covered):
        raise TrainingError("A version can only be activated for belt types it was trained on.")
    with get_conn() as conn:
        for t in targets:
            conn.execute("INSERT INTO active_models (belt_type_id, model_id, activated_at, activated_by) "
                         "VALUES (?,?,?,?) ON CONFLICT(belt_type_id) DO UPDATE SET model_id=excluded.model_id, "
                         "activated_at=excluded.activated_at, activated_by=excluded.activated_by",
                         (t, model_id, now_iso(), user_id))
    return targets


def delete_model(model_id: int) -> None:
    m = get_model(model_id)
    if m is None:
        raise TrainingError("Model not found.")
    if m["active_for"]:
        raise TrainingError("An active version cannot be deleted. Activate another version first.")
    if m["status"] == "training":
        raise TrainingError("This version is still training.")
    with get_conn() as conn:
        conn.execute("UPDATE inspections SET model_id = NULL WHERE model_id = ?", (model_id,))
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    _booster_cache.pop(model_id, None)
    path = os.path.join(MODELS_DIR, str(model_id), "model.ubj")
    if os.path.exists(path):
        os.remove(path)
        os.rmdir(os.path.dirname(path))


def active_model_ids() -> Dict[int, int]:
    with get_conn() as conn:
        return {r[0]: r[1] for r in conn.execute("SELECT belt_type_id, model_id FROM active_models").fetchall()}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison (same test belts for both versions)
# ─────────────────────────────────────────────────────────────────────────────

def compare(a_id: int, b_id: int) -> Dict:
    ma, mb = get_model(a_id), get_model(b_id)
    if not ma or not mb or ma["status"] != "ready" or mb["status"] != "ready":
        raise TrainingError("Both versions must be trained successfully.")
    newer = ma if a_id > b_id else mb
    specs_a = {d["filename"]: d for d in ma["info"]["datasets"]}
    specs_b = {d["filename"]: d for d in mb["info"]["datasets"]}
    common = sorted(set(specs_a) & set(specs_b))
    shared_types = set(ma["info"]["belt_type_ids"]) & set(mb["info"]["belt_type_ids"])
    result = {"comparable": False, "models": {}, "datasets": common}
    for m in (ma, mb):
        result["models"][m["id"]] = {"own_metrics": m["info"]["metrics"], "flags": _flags(m)}
    if not common or not shared_types:
        result["reason"] = ("These versions share no training dataset/belt type, so there are no common "
                            "test belts. Their own metrics were measured on different belts and are not directly comparable.")
        return result

    with get_conn() as conn:
        types = list_belt_types(conn)
    newer_specs = {d["filename"]: d for d in newer["info"]["datasets"]}
    frames, estimated = [], False
    for fn in common:
        try:
            df, info, _ = load_dataset(newer_specs[fn], types)
        except TrainingError as e:
            result["reason"] = str(e)
            return result
        frames.append(df)
        estimated |= bool(info.get("estimated"))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["belt_id"].astype(str).map(is_test_belt) & df["belt_type_id"].isin(shared_types)]
    if df.empty:
        result["reason"] = "No test belts in the shared datasets."
        return result

    for m in (ma, mb):
        X = build_X(df, m["info"]["feature_cols"], m["info"]["medians"])
        median = np.maximum(booster_median(load_booster(m["id"]), X), 0.0)
        sigma = (m["info"].get("aft") or {}).get("sigma")
        result["models"][m["id"]]["shared_metrics"] = score_belts(df, median, sigma)
    result.update({
        "comparable": True,
        "n_test_belts": int(df["belt_id"].nunique()),
        "truth_estimated": estimated,
        "note": "Both versions scored on the same held-out belts. 'True' RUL uses the current belt-type "
                "limits" + (" and is ESTIMATED for some belts." if estimated else "."),
    })
    return result


def _flags(m: Dict) -> List[str]:
    info, flags = m["info"], []
    if info.get("rul_estimated"):
        flags.append("Trained on ESTIMATED RUL (no recorded failures for some belts)")
    if info.get("thresholds_unconfirmed"):
        flags.append("Unconfirmed failure limits: " + ", ".join(info["thresholds_unconfirmed"]))
    if info.get("test_split") == "by_row":
        flags.append("No belt IDs - test metrics may be optimistic")
    if info.get("n_belts", 0) < 30:
        flags.append(f"Small training set ({info.get('n_belts')} belts)")
    if (info.get("metrics") or {}).get("bias_hours", 0) > 0.1 * max(info["metrics"].get("mae_hours", 0), 1):
        flags.append("Tends to OVER-estimate remaining life (optimistic)")
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Prediction for a belt in the fleet
# ─────────────────────────────────────────────────────────────────────────────

def health_tier(health: Optional[float], failed: bool) -> str:
    """Green / amber / red from the health score (distance to the belt type's failure limits)."""
    if failed:
        return "red"
    if health is None:
        return "unknown"
    return "red" if health < 25 else "amber" if health < 50 else "green"


def predict_inspections(belt: Dict, belt_type: Dict, inspections: List[Dict], model_id: Optional[int]) -> List[Dict]:
    """inspections: [{hour, readings}] sorted by hour -> per-inspection prediction dicts."""
    if not inspections:
        return []
    schema_fields = sm.load_canonical()["fields"]
    rows = [{**belt.get("static", {}), **i["readings"], "hour": i["hour"], "belt_type_id": belt["belt_type_id"]}
            for i in inspections]
    df = sm.derive_features(pd.DataFrame(rows))

    preds, sigma = [None] * len(df), None
    if model_id is not None:
        m = get_model(model_id)
        X = build_X(df, m["info"]["feature_cols"], m["info"]["medians"])
        preds = [max(float(p), 0.0) for p in booster_median(load_booster(model_id), X)]
        sigma = (m["info"].get("aft") or {}).get("sigma")   # None for legacy point-regression versions

    out = []
    for idx, insp in enumerate(inspections):
        ratios = {}
        for col, thr in belt_type["thresholds"].items():
            val = insp["readings"].get(col)
            if val is not None and thr:
                ratios[col] = float(val) / float(thr)
        failed = any(r >= 1.0 for r in ratios.values())
        health = round(max(1 - max(ratios.values()), 0.0) * 100, 1) if ratios else None
        rul = 0.0 if failed and preds[idx] is not None else preds[idx]
        # 80 % interval = [10 %, 90 %] quantiles of the AFT distribution around the median; a belt past its
        # failure limit has no remaining life, so no interval either.
        has_band = rul is not None and sigma is not None
        low = 0.0 if has_band and failed else rul * float(np.exp(-_Z * sigma)) if has_band else None
        high = 0.0 if has_band and failed else rul * float(np.exp(_Z * sigma)) if has_band else None
        dominant = schema_fields[max(ratios, key=ratios.get)]["label"] if ratios else None
        out.append({
            "hour": float(insp["hour"]),
            "rul_hours": None if rul is None else round(rul, 0),
            "health_score": health,
            "tier": health_tier(health, failed),
            "dominant_driver": dominant,
            "limit_reached": failed,
            "degradation_ratios": {k: round(v, 3) for k, v in ratios.items()},
            "rul_low": None if low is None else round(low, 0),
            "rul_high": None if high is None else round(high, 0),
            "interval_level": INTERVAL_LEVEL if has_band else None,
            "model_id": model_id,
        })
    return out
