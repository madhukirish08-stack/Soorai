# Project context (for an AI assistant starting a new chat)

Read this first, then the file you are about to change. Humans: see `README.md`.

## What this is
Belt RUL: remaining-useful-life prediction and health tracking for industrial power-transmission belts (JK Fenner-style V-belts), built for ONE client. FastAPI backend + vanilla-JS frontend (no build step) + SQLite + XGBoost. Two roles: **planner** (fleet, inspections, end-of-life) and **engineer** (also data upload, column mapping, training, model activation, belt types, users).

Real plant data is mostly **censored**: most belts are replaced early or still running, few run to failure. That is why the model is a survival model.

## Run it
```bash
pip install -r requirements.txt
BELT_RUL_DEV_MODE=1 BELT_RUL_INSTANCE_DIR=<temp dir> python backend/manage.py create-user eng --role engineer   # prompts for a password
BELT_RUL_DEV_MODE=1 BELT_RUL_INSTANCE_DIR=<temp dir> python backend/api.py                                    # http://127.0.0.1:8001
python training/generate_synthetic_vbelts.py --belts 300 --censor-frac 0.3 --output synth.csv
python tests/e2e_flow.py synth.csv        # the regression check; must print "E2E OK"
```
Dev mode enables synthetic data generation, `/docs`, and non-Secure cookies. Never in production. All runtime state (datasets, mappings, models, SQLite db) lives in `instance/` (git-ignored); client data must never be committed.

## Layout and responsibilities
| File | Role |
|---|---|
| `backend/api.py` | All REST routes + static frontend mount. Pydantic bodies, role checks via `auth.current_user` / `auth.require_engineer`. `TrainingError` becomes HTTP 400. |
| `backend/model_registry.py` | Training, versioning, activation, comparison, prediction. The core. |
| `backend/schema_mapper.py` | CSV column mapping (exact match, saved profile, Gemini, fuzzy fallback), unit conversion, `derive_features`, `derive_rul` (survival targets). |
| `backend/canonical_schema.json` | The model's field names, roles (`feature`, `degradation`, `time`, `target`, `failure_time`, ...), units, ranges. |
| `backend/db.py` | SQLite schema + seed belt types. Tables: users, sessions, belt_types, belts, inspections, models, active_models. |
| `backend/auth.py`, `config.py`, `manage.py` | Scrypt passwords + cookie sessions; env/.env settings and paths; admin CLI. |
| `frontend/app.js` | Single-file SPA: hash routing, one `render*` function per view (fleet, belt, types, data, models, users), a `state` object, event delegation via `data-action`. |
| `training/generate_synthetic_vbelts.py` | Physics-inspired demo data, NOT calibrated to real belts. `--censor-frac` truncates belts' logs. |
| `backend/fleet.py` | Phase 4: `store_inspection` (the ONE writer of an inspection + its stored prediction), bulk CSV import (belts/inspections, validate-first, all-or-nothing), `accuracy_report` (backtest), `actual_remaining`, `export_fleet_dataset`. |
| `tests/e2e_flow.py` | In-process FastAPI TestClient flow on a temp instance dir. |
| `tests/test_phase4.py` | Phase 4 regression: import validation/commit, backtest, ground truth on belt pages, fleet->dataset->train, DB migration. |

## Data flow
1. Upload CSV -> `schema/propose` -> engineer approves mapping (saved as `instance/mappings/<file>.mapping.json`, keyed by column signature). Fully canonical files skip the review.
2. `load_dataset` (registry): apply mapping -> match belt type -> namespace `belt_id` per file -> `derive_rul` using each row's belt-type failure limits.
3. `_train` builds features, hash-splits belts into train/test, fits, saves `instance/models/<id>/model.ubj`, stores metrics and metadata in `models.info_json`.
4. Engineer activates a version per belt type (`active_models`). `predict_inspections` always uses the ACTIVE model of the belt's type, live at request time. Inspections also store the prediction made at the time (audit trail).

## Modelling: key facts and invariants
- **Targets** (`derive_rul`): per belt, priority = measured `rul_hours` > `failure_hour` > observed threshold crossing (event, flagged `estimated`) > otherwise **right-censored** at the belt's last hour. Output columns `rul_hours` (true RUL, or a LOWER BOUND when censored) and `rul_censored`. Never extrapolate degradation to invent a failure time (removed on purpose). A censored belt's last row (lower bound 0) is dropped.
- **Methods**: `aft` (default; `survival:aft`, normal distribution) or `regression` (legacy `reg:squarederror` on events only, no interval). Models saved before survival support have no `info["method"]`; treat as regression. Both load with `xgb.Booster`.
- **AFT details**: `predict()` returns exp(margin) = the MEDIAN RUL. Interval = median * exp(+-z * sigma), z for 80 % (`INTERVAL_LEVEL`). Sigma (`AFT_SIGMAS`) and tree count are picked on a hash-based validation split of TRAINING belts, then refit on all training belts. Sigma is stored in `info["aft"]["sigma"]`. Labels are clipped to >= `MIN_RUL_H` (log space).
- **Test belts** = `md5(belt_id) % 100 < TEST_PCT` (`is_test_belt`). Same belts in every version, which makes `compare()` fair and prevents leakage. Do not change the hash or the belt-id namespacing without understanding that old versions' test sets would shift. If the test set has no events, the lowest-hash event belt is used.
- **Metrics** (`score_belts`): MAE/RMSE/R2/bias/near-failure on EVENT rows only; `c_index` (all test rows, censored as lower bounds), `c_index_belt` (first inspection per belt), `interval_coverage` (target about 0.8), `interval_mean_width_hours`. `compare()` reuses `score_belts` on the shared test belts.
- **Features**: canonical `feature`/`degradation`/`time` roles + `x_*` extras + one-hot `type_<id>` when several belt types. Columns under 50 % coverage dropped; missing values filled with training medians (`build_X`). `derive_features` computes belt speed, bending frequency, speed ratio identically at train and predict time.
- **Failure limits belong to the belt type** (DB), not the schema. Seeded limits are placeholders until an engineer confirms them. Health score = distance to the nearest limit; a belt past a limit gets RUL 0 and no interval.
- Models are stored as XGBoost `.ubj`, never pickle.
- Training runs in a background thread under `_train_lock`; one training at a time; single worker process only.

## Prediction payload (per inspection)
`rul_hours` (median), `rul_low`, `rul_high`, `interval_level` (null for regression versions), `health_score`, `tier`, `dominant_driver`, `limit_reached`, `degradation_ratios`, `model_id`. The old `confidence_pct` heuristic is gone; the UI shows "RUL low-high h" instead.

## Known behaviours and gotchas
- `failure_hour` in the canonical schema means "failed or replaced"; if a client fills it for early-replaced belts they are treated as failures. Document this to clients (README / DATA_REQUIREMENTS).
- On the synthetic set (30 % censored), AFT has higher MAE than regression (about 2,830 vs 1,910 h) but gives intervals (coverage about 87 %), concordance about 0.89. Synthetic censoring is random; real censoring is probably informative, which AFT does not model.
- The interval width is proportional to the median (single sigma): it shrinks near failure.
- `frontend/app.js` is a large single file; match its style (template strings, `esc()` on user text, `api()` helper, `guarded()` for errors).
- Windows dev machine: use forward slashes in bash; very long heredocs in the shell tool can break, so write scripts to files.

## Conventions
- Comment any custom fix with the reason and a warning not to revert it.
- Prefer removing code that causes unwanted behaviour over patching around it.
- Run `node --check frontend/app.js`, `python tests/e2e_flow.py <csv>` and `python tests/test_phase4.py` before declaring done.

## Phase 4 invariants (fleet loop)
- Every inspection row stores the prediction made AT THAT TIME: `predicted_rul` (median), `predicted_low/high` (80 % interval), `model_id`. Written only through `fleet.store_inspection`, by both manual entry and bulk import. The backtest uses these stored values, never a re-prediction. DB columns added after v1 go in `db.MIGRATIONS` (idempotent `ALTER TABLE` in `init_db`).
- **Bulk import is all-or-nothing**: any row error blocks the commit; `dry_run` (default true in the API) writes nothing; validation and commit share one transaction. Numbers must be plain (`1,800` is rejected, never guessed). Unrecognised columns are ignored but reported loudly. Inspection files may not carry belt-level (static) columns and vice versa. Header aliases live in `fleet.py` (`TAG_ALIASES`, `HOUR_ALIASES`, `FIELD_ALIASES`); aliases never imply a unit conversion.
- **Accuracy counts only belts with status `failed`.** `replaced` belts are right-censored: the true RUL is only a lower bound, so they are reported separately (`too_pessimistic` = the model's interval ended before the proven survival time). Under `MIN_FAILED_FOR_CONFIDENCE` (10) failed belts the UI flags figures as anecdotal. Bias = predicted - actual (positive = optimistic).
- `end_of_life` refuses an end hour earlier than the belt's last inspection.
- `export_fleet_dataset` writes a fully canonical CSV (no mapping step needed). `failure_hour` is filled ONLY for failed belts; replaced and in-service belts stay blank so they are censored. `x_*` extra readings are not exported. Replaced belts whose last reading is past a failure limit are still treated by `derive_rul` as (estimated) failures - correct, and flagged in `rul_source`.

## Roadmap (from the project owner)
Phases 1-5 done (auth, persistence, belt types with own limits, model registry, bulk import + real-failure backtest + fleet->dataset, survival analysis). Possible next: calibrate the interval per belt/horizon using the real-failure backtest (conformal-style), treat early replacement as informative censoring, per-belt-type accuracy breakdown, scheduled inspection reminders.
