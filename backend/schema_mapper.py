"""
Schema Mapper
=============
Adapter layer between arbitrary customer CSVs and the canonical belt schema
(canonical_schema.json). The model only ever sees canonical column names.

Flow:
  1. propose_mapping()  - exact-name matches are resolved locally; the remaining
                          columns are sent (names, stats, 5 sample values) to Gemini
                          once, or to a fuzzy-name fallback if no API key is set.
  2. A human reviews/edits the proposal in the UI (mandatory - never auto-applied).
  3. validate_mapping() + save_mapping() persist the approved profile, keyed by
                          dataset name and column signature, so later uploads with
                          the same columns reuse it without another LLM call.
  4. apply_mapping()    - renames, unit-converts (canonical = raw*scale + offset).
  5. derive_rul()       - builds survival targets: measured RUL / failure hours / observed threshold
                          crossings are events; belts that never ended are right-censored.
"""

import os
import re
import json
import hashlib
import time
import difflib
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config  # noqa: F401  (loads .env before GEMINI_* is read)

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_schema.json")

IGNORE = "__ignore__"   # drop the column
EXTRA  = "__extra__"    # keep as an additional numeric feature with no canonical meaning
EXTRA_PREFIX = "x_"

GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_FALLBACK_MODELS", "gemini-3.7-flash,gemini-2.5-flash").split(",") if m.strip()]
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical() -> Dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def fields_by_role(schema: Dict, *roles: str) -> List[str]:
    return [k for k, v in schema["fields"].items() if v["role"] in roles]


def default_thresholds(columns, schema: Optional[Dict] = None) -> Dict[str, float]:
    """Generic limits, ONLY for sanity-checking a mapping before a belt type is chosen.
    Training and prediction always use the belt type's own limits."""
    schema = schema or load_canonical()
    return {
        c: float(schema["fields"][c]["default_failure_threshold"])
        for c in fields_by_role(schema, "degradation") if c in columns
    }


def extra_feature_name(source: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", source).strip("_").lower() or "col"
    return f"{EXTRA_PREFIX}{slug}"


def column_signature(columns) -> str:
    norm = sorted(str(c).strip().lower() for c in columns)
    return hashlib.sha1("|".join(norm).encode("utf-8")).hexdigest()[:16]


def is_fully_canonical(columns) -> bool:
    """True when every column already has a canonical name - nothing to map, so such
    files can train without a mapping review."""
    fields = load_canonical()["fields"]
    return len(columns) > 0 and all(str(c) in fields for c in columns)


# ─────────────────────────────────────────────────────────────────────────────
# Profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame, n_samples: int = 5) -> List[Dict]:
    profile = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        entry = {
            "name": str(col),
            "dtype": str(s.dtype),
            "null_pct": round(float(s.isna().mean() * 100), 1),
            "n_unique": int(non_null.nunique()),
            "samples": [str(v) for v in non_null.drop_duplicates().head(n_samples).tolist()],
        }
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.8 and num.notna().any():
            entry.update({
                "min": float(num.min()), "max": float(num.max()),
                "mean": round(float(num.mean()), 4), "median": float(num.median()),
            })
        profile.append(entry)
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Proposal (exact match → saved profile → Gemini → fuzzy fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _entry(source, target, confidence, reason, scale=1.0, offset=0.0):
    return {"source": source, "target": target, "confidence": round(float(confidence), 2),
            "scale": float(scale), "offset": float(offset), "reason": reason}


def _build_prompt(schema: Dict, profile: List[Dict], taken: List[str]) -> str:
    fields = {
        k: {kk: v.get(kk) for kk in ("role", "label", "unit", "range", "description") if v.get(kk) is not None}
        for k, v in schema["fields"].items() if k not in taken
    }
    return f"""You map columns of an industrial power-transmission belt (V-belt, wedge, cogged, Poly-V) maintenance dataset onto a canonical schema.
The result trains a Remaining-Useful-Life model, so a wrong mapping is worse than no mapping.

CANONICAL FIELDS still available (name -> definition):
{json.dumps(fields, indent=1, ensure_ascii=False)}

INCOMING COLUMNS (name, dtype, stats, sample values):
{json.dumps(profile, indent=1, ensure_ascii=False)}

RULES
- Each canonical field may be used by at most one incoming column.
- target "{EXTRA}": numeric sensor/measurement columns plausibly related to belt condition that have no canonical match.
- target "{IGNORE}": free text, row numbers, constant columns, categorical labels, or anything you cannot map with reasonable confidence.
- If the incoming unit differs from the canonical unit, give scale and offset so that canonical = raw * scale + offset
  (e.g. Fahrenheit->Celsius scale=0.5556 offset=-17.7778; fraction 0-1 -> percent scale=100; metres->mm scale=1000;
  inches->mm scale=25.4).
- Belt section / product codes (A, B, SPB, BX, Poly-F Plus, ...) map to "belt_type".
  Infer units from names AND value ranges. Otherwise scale=1, offset=0.
- A column holding remaining life in DAYS maps to rul_hours with scale=24 (state the assumption).
- confidence in [0,1]. If two canonical fields are plausible, pick the likelier one, give confidence <= 0.6 and say why.
- Return one entry for EVERY incoming column.

Respond with JSON only:
{{"mappings":[{{"source":"<incoming name>","target":"<canonical name|{EXTRA}|{IGNORE}>","confidence":0.0,"scale":1.0,"offset":0.0,"reason":"<short justification incl. unit assumption>"}}]}}"""


def call_gemini(prompt: str, api_key: str, model: str = GEMINI_MODEL, timeout: int = 60) -> Dict:
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }).encode("utf-8")
    req = urllib.request.Request(
        GEMINI_ENDPOINT.format(model=model), data=body, method="POST",
        # Key goes in a header, never the URL, so it can't leak into proxy/access logs.
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    # Gemini returns 429/503 under load ("high demand"); those are transient, so retry with backoff.
    for attempt, wait in enumerate((0, 2, 5)):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == 2:
                raise RuntimeError(f"Gemini API error {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 2:
                raise RuntimeError(f"Gemini API unreachable: {e}")
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def _fuzzy_guess(source: str, schema: Dict, taken: set) -> Tuple[str, float, str]:
    norm = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    candidates = {}
    for name, spec in schema["fields"].items():
        if name in taken:
            continue
        candidates[name] = name
        candidates[re.sub(r"[^a-z0-9]+", "_", spec["label"].lower()).strip("_")] = name
    # Strict cutoff: a loose match ("Run Hrs" -> rul_hours) is worse than none. Unmatched
    # columns are left for the engineer to map by hand.
    match = difflib.get_close_matches(norm, list(candidates), n=1, cutoff=0.8)
    if match:
        return candidates[match[0]], 0.4, f"Fuzzy name match to '{match[0]}' (no LLM available) - verify unit"
    return IGNORE, 0.0, "No confident match (no LLM available)"


def _sanitize_llm(raw: Dict, columns: List[str], schema: Dict, taken: set) -> List[Dict]:
    valid_targets = set(schema["fields"]) | {IGNORE, EXTRA}
    by_source = {}
    for m in raw.get("mappings", []):
        src = m.get("source")
        if src not in columns or src in by_source:
            continue
        tgt = m.get("target", IGNORE)
        reason = str(m.get("reason", ""))
        if tgt not in valid_targets or tgt in taken:
            reason = f"LLM suggested invalid/duplicate target '{tgt}'. " + reason
            tgt = IGNORE
        try:
            scale, offset = float(m.get("scale", 1.0)), float(m.get("offset", 0.0))
        except (TypeError, ValueError):
            scale, offset = 1.0, 0.0
        conf = min(max(float(m.get("confidence", 0.0) or 0.0), 0.0), 1.0)
        by_source[src] = _entry(src, tgt, conf, reason, scale, offset)
        if tgt not in (IGNORE, EXTRA):
            taken.add(tgt)
    for c in columns:
        if c not in by_source:
            by_source[c] = _entry(c, IGNORE, 0.0, "Not returned by the LLM")
    return [by_source[c] for c in columns]


def propose_mapping(df: pd.DataFrame, mappings_dir: str) -> Dict:
    schema = load_canonical()
    columns = [str(c) for c in df.columns]
    profile = profile_dataframe(df)
    warnings: List[str] = []

    saved = find_profile_by_signature(mappings_dir, column_signature(columns))
    if saved:
        return {"source": "saved_profile", "mapping": saved["mapping"], "profile": profile,
                "warnings": [f"Re-using the mapping approved for '{saved['dataset']}' (same columns). Review before approving."]}

    taken = set()
    resolved, unresolved = {}, []
    for c in columns:
        if c in schema["fields"] and c not in taken:
            resolved[c] = _entry(c, c, 1.0, "Exact canonical name")
            taken.add(c)
        else:
            unresolved.append(c)

    source = "exact"
    if unresolved:
        api_key = os.environ.get("GEMINI_API_KEY")
        unresolved_profile = [p for p in profile if p["name"] in unresolved]
        llm_entries = None
        if api_key:
            prompt = _build_prompt(schema, unresolved_profile, sorted(taken))
            failures = []
            # Primary model first, then fallbacks: a single overloaded model must not
            # silently degrade every proposal to name matching.
            for model in [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]:
                try:
                    raw = call_gemini(prompt, api_key, model=model)
                    llm_entries = _sanitize_llm(raw, unresolved, schema, set(taken))
                    source = "llm"
                    if failures:
                        warnings.append(f"{GEMINI_MODEL} was unavailable; proposal made by {model}.")
                    break
                except Exception as e:
                    failures.append(f"{model}: {str(e)[:120]}")
            if llm_entries is None:
                warnings.append("Gemini unavailable (" + "; ".join(failures) + "); fell back to name matching. "
                                "Review every row carefully, or retry later.")
        else:
            warnings.append("GEMINI_API_KEY is not set; using fuzzy name matching. Review every row carefully.")
        if llm_entries is None:
            source = "heuristic"
            llm_entries = []
            for c in unresolved:
                tgt, conf, reason = _fuzzy_guess(c, schema, taken)
                if tgt != IGNORE:
                    taken.add(tgt)
                llm_entries.append(_entry(c, tgt, conf, reason))
        for e in llm_entries:
            resolved[e["source"]] = e

    return {"source": source, "mapping": [resolved[c] for c in columns], "profile": profile, "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────────────
# Apply / validate / persist
# ─────────────────────────────────────────────────────────────────────────────

def apply_mapping(raw: pd.DataFrame, mapping: List[Dict]) -> pd.DataFrame:
    schema = load_canonical()
    out = pd.DataFrame(index=raw.index)
    for m in mapping:
        src, tgt = m["source"], m["target"]
        if tgt == IGNORE or src not in raw.columns:
            continue
        name = extra_feature_name(src) if tgt == EXTRA else tgt
        role = "feature" if tgt == EXTRA else schema["fields"][tgt]["role"]
        col = raw[src]
        if role in ("id", "belt_type"):
            out[name] = col.astype(str).str.strip()
        elif role == "timestamp":
            out[name] = pd.to_datetime(col, errors="coerce")
        else:
            out[name] = pd.to_numeric(col, errors="coerce") * float(m.get("scale", 1.0)) + float(m.get("offset", 0.0))

    if "timestamp" in out.columns:
        if "hour" not in out.columns:
            ts = out["timestamp"]
            origin = ts.groupby(out["belt_id"]).transform("min") if "belt_id" in out.columns else ts.min()
            out["hour"] = (ts - origin).dt.total_seconds() / 3600.0
        out = out.drop(columns=["timestamp"])
    return derive_features(out)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculated stress indicators. Used identically at training and prediction time.
    Provided values are kept; only missing ones are computed."""
    df = df.copy()

    def missing(c):
        return c not in df.columns or df[c].isna().all()

    if missing("belt_speed_mps") and {"driver_pulley_dia_mm", "driver_rpm"} <= set(df.columns):
        df["belt_speed_mps"] = np.pi * df["driver_pulley_dia_mm"] * df["driver_rpm"] / 60000.0
    if missing("pulley_ratio") and {"driver_pulley_dia_mm", "driven_pulley_dia_mm"} <= set(df.columns):
        df["pulley_ratio"] = df["driven_pulley_dia_mm"] / df["driver_pulley_dia_mm"]
    if missing("bending_freq_hz") and {"belt_speed_mps", "pitch_length_mm"} <= set(df.columns):
        pulleys = df["num_pulleys"] if "num_pulleys" in df.columns else 2
        df["bending_freq_hz"] = df["belt_speed_mps"] / (df["pitch_length_mm"] / 1000.0) * pulleys
    return df


def identity_mapping(columns) -> List[Dict]:
    schema = load_canonical()
    return [_entry(c, c if c in schema["fields"] else IGNORE, 1.0, "Exact canonical name") for c in columns]


def validate_mapping(raw: pd.DataFrame, mapping: List[Dict]) -> Tuple[List[str], List[str], Optional[Dict]]:
    schema = load_canonical()
    errors, warnings = [], []
    valid_targets = set(schema["fields"]) | {IGNORE, EXTRA}

    seen = {}
    for m in mapping:
        src, tgt = m.get("source"), m.get("target")
        if src not in raw.columns:
            errors.append(f"Unknown source column '{src}'.")
        if tgt not in valid_targets:
            errors.append(f"'{src}' maps to unknown field '{tgt}'.")
        elif tgt not in (IGNORE, EXTRA):
            if tgt in seen:
                errors.append(f"'{seen[tgt]}' and '{src}' both map to '{tgt}'.")
            seen[tgt] = src
        try:
            float(m.get("scale", 1.0)); float(m.get("offset", 0.0))
        except (TypeError, ValueError):
            errors.append(f"'{src}': scale/offset must be numbers.")
    if errors:
        return errors, warnings, None

    df = apply_mapping(raw, mapping)

    for m in mapping:
        src, tgt = m["source"], m["target"]
        if tgt == IGNORE:
            continue
        name = extra_feature_name(src) if tgt == EXTRA else tgt
        role = "feature" if tgt == EXTRA else schema["fields"][tgt]["role"]
        if role in ("id", "timestamp", "belt_type"):
            continue
        raw_non_null = raw[src].notna()
        if raw_non_null.any():
            lost = df.loc[raw_non_null, name].isna().mean()
            if lost > 0.2:
                warnings.append(f"'{src}': {lost:.0%} of values are not numeric and will be treated as missing.")
        rng = schema["fields"].get(tgt, {}).get("range")
        if rng and df[name].notna().any():
            med = float(df[name].median())
            if not (rng[0] <= med <= rng[1]):
                warnings.append(
                    f"'{src}' → {tgt}: median after conversion is {med:,.3g} {schema['fields'][tgt]['unit']}, "
                    f"outside the expected {rng[0]:,}–{rng[1]:,}. Check the unit conversion."
                )
        # Range checks miss fraction-vs-percent errors (0.005 is a "valid" %), but a
        # degradation measure that never gets near its failure threshold almost always is one.
        thr = schema["fields"].get(tgt, {}).get("default_failure_threshold")
        if thr and df[name].notna().any() and df[name].quantile(0.99) < 0.05 * thr:
            warnings.append(
                f"'{src}' → {tgt}: values never exceed 5% of the failure threshold ({thr:g} "
                f"{schema['fields'][tgt]['unit']}). If it is stored as a fraction (0–1), set scale = 100."
            )

    if "timestamp" in seen:
        ts_src = raw[seen["timestamp"]]
        bad = pd.to_datetime(ts_src, errors="coerce")[ts_src.notna()].isna().mean() if ts_src.notna().any() else 0.0
        if bad > 0.2:
            warnings.append(f"'{seen['timestamp']}': {bad:.0%} of values could not be parsed as dates.")
    if "timestamp" in seen and "hour" not in seen:
        warnings.append("Operating hours are computed from timestamps (hours since each belt's first reading). "
                        "Idle time and pre-logging run time are not accounted for - an hour-meter column is preferred.")

    features = [c for c in df.columns if c in fields_by_role(schema, "feature", "degradation", "time") or c.startswith(EXTRA_PREFIX)]
    if "belt_type" not in seen:
        warnings.append("No belt-type column: you will choose the belt type for the whole file when training.")
    if not features:
        errors.append("No usable feature columns are mapped.")

    rul_info = None
    try:
        _, rul_info = derive_rul(df)
        if rul_info["estimated"]:
            warnings.append(rul_info["message"])
    except ValueError as e:
        errors.append(str(e))
    return errors, warnings, rul_info


def _mapping_path(mappings_dir: str, dataset_name: str) -> str:
    return os.path.join(mappings_dir, f"{os.path.basename(dataset_name)}.mapping.json")


def save_mapping(mappings_dir: str, dataset_name: str, columns, mapping: List[Dict], proposal_source: str) -> Dict:
    os.makedirs(mappings_dir, exist_ok=True)
    record = {
        "dataset": os.path.basename(dataset_name),
        "signature": column_signature(columns),
        "columns": [str(c) for c in columns],
        "proposal_source": proposal_source,
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mapping": mapping,
    }
    with open(_mapping_path(mappings_dir, dataset_name), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def load_mapping(mappings_dir: str, dataset_name: str, columns) -> Optional[Dict]:
    """Approved mapping for this dataset, or None if absent or the file's columns changed since approval."""
    path = _mapping_path(mappings_dir, dataset_name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    return record if record.get("signature") == column_signature(columns) else None


def find_profile_by_signature(mappings_dir: str, signature: str) -> Optional[Dict]:
    if not os.path.isdir(mappings_dir):
        return None
    for fname in sorted(os.listdir(mappings_dir)):
        if fname.endswith(".mapping.json"):
            try:
                with open(os.path.join(mappings_dir, fname), encoding="utf-8") as f:
                    record = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("signature") == signature:
                return record
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RUL target (survival data: events + right-censored belts)
# ─────────────────────────────────────────────────────────────────────────────

MEASURE_ADVICE = ("For accurate predictions, record the actual operating hour at which each belt failed or was "
                  "replaced (see docs/DATA_REQUIREMENTS.md).")


def derive_rul(df: pd.DataFrame, row_thresholds: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:
    """Return (rows with 'rul_hours' and 'rul_censored', info) as survival-analysis targets.

    Per belt, in priority order:
      measured RUL column > recorded failure hour > observed threshold crossing  -> EVENT: rul_hours is the true
                                                                                    remaining life, rul_censored False
      none of the above (still in service, or replaced before failing)           -> RIGHT-CENSORED at its last
                                                                                    hour: rul_hours is only a LOWER
                                                                                    bound (last_hour - hour), rul_censored True
    # CUSTOM FIX (requested): belts are never extrapolated to an invented failure time any more - that
    # produced fake targets. Do not reintroduce extrapolation; censoring is the correct treatment.

    row_thresholds: failure limit per row and degradation column (each row's belt-type limits),
    aligned to df.index. None -> schema default limits (mapping sanity checks only)."""
    schema = load_canonical()
    df = df.copy()
    df["rul_censored"] = False
    has_rul = "rul_hours" in df.columns and df["rul_hours"].notna().any()
    has_failure = "failure_hour" in df.columns
    can_follow = "belt_id" in df.columns and "hour" in df.columns   # needed to see each belt's last hour

    if not can_follow:
        # No way to group rows into belts: only rows with a known target are usable (never censored).
        if has_rul and df["rul_hours"].notna().mean() >= 0.5:
            df = df[df["rul_hours"].notna()]
            return df, {"method": "measured", "estimated": False, "counts": {"measured": int(len(df))},
                        "message": "Using the measured remaining-useful-life column from the dataset."}
        if has_failure and "hour" in df.columns:
            df = df[df["failure_hour"].notna() & df["hour"].notna()]
            df["rul_hours"] = (df["failure_hour"] - df["hour"]).clip(lower=0)
            return df, {"method": "failure_hour", "estimated": False, "counts": {"measured_failure_rows": int(len(df))},
                        "message": "RUL computed from recorded failure hours."}
        raise ValueError("Cannot generate RUL: a belt/asset ID column and operating hours are needed to follow each "
                         "belt over time (or provide a remaining-useful-life / failure-hour column).")

    if "rul_hours" not in df.columns:
        df["rul_hours"] = np.nan
    deg_cols = [c for c in fields_by_role(schema, "degradation") if c in df.columns]
    if row_thresholds is None:
        defaults = default_thresholds(deg_cols, schema)
        row_thresholds = pd.DataFrame({c: defaults[c] for c in deg_cols}, index=df.index)
    thresholds = [c for c in deg_cols if c in row_thresholds.columns and row_thresholds[c].notna().any()]
    if not has_rul and not has_failure and not thresholds:
        raise ValueError("Cannot generate RUL: the dataset has no remaining-life, failure-hour, or degradation "
                         "measurement (wear / elongation / crack) column.")

    ratio = None
    if thresholds:
        ratio = pd.concat([df[c] / row_thresholds.loc[df.index, c] for c in thresholds], axis=1).max(axis=1, skipna=True)

    counts = {"measured_rul": 0, "measured_failure": 0, "threshold_observed": 0, "censored": 0, "dropped": 0}
    parts = []
    for _, g in df[df["hour"].notna()].groupby("belt_id", sort=False):
        g = g.sort_values("hour")
        if g["rul_hours"].notna().any():
            g, kind = g[g["rul_hours"].notna()].copy(), "measured_rul"
        else:
            fail_h, kind = None, None
            if has_failure and g["failure_hour"].notna().any():
                fail_h, kind = float(g["failure_hour"].dropna().iloc[0]), "measured_failure"
            elif ratio is not None:
                crossed = g.loc[ratio.loc[g.index] >= 1.0, "hour"]
                if len(crossed):
                    fail_h, kind = float(crossed.iloc[0]), "threshold_observed"
            if fail_h is not None:
                g = g[g["hour"] <= fail_h].copy()
                g["rul_hours"] = fail_h - g["hour"]
            else:
                # Censored: still running / replaced early. A row at the last hour says nothing (lower bound 0).
                kind, g = "censored", g.copy()
                g["rul_hours"] = g["hour"].max() - g["hour"]
                g["rul_censored"] = True
                g = g[g["rul_hours"] > 0]
        if g.empty:
            counts["dropped"] += 1
            continue
        counts[kind] += 1
        parts.append(g)

    if not parts:
        raise ValueError(f"Cannot generate RUL: no belt has usable inspections ({counts['dropped']} belts checked).")
    out = pd.concat(parts)
    estimated = counts["threshold_observed"] > 0
    thr_txt = "each belt type's " + ", ".join(schema["fields"][c]["label"].lower() for c in thresholds) + " limit"
    n_events = counts["measured_rul"] + counts["measured_failure"] + counts["threshold_observed"]
    message = (f"{n_events} belts with a known end of life, {counts['censored']} still-running or replaced-early belts "
               f"(right-censored: they count as 'lasted at least this long', not as failures), {counts['dropped']} skipped.")
    if estimated:
        message = (f"Failure time for {counts['threshold_observed']} belts was ESTIMATED from an assumed threshold "
                   f"({thr_txt}), not recorded. " + message + " " + MEASURE_ADVICE)
    return out, {"method": "survival", "estimated": estimated, "counts": counts, "message": message}
