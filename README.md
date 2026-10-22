# Belt RUL — Belt Health & Remaining Life

> AI assistants: read `projectcontext.md` first.

Remaining-useful-life (RUL) prediction and health tracking for industrial **power-transmission belts**: classical wrapped, wedge, raw-edge cogged, Poly-V and others. Plant data is mapped to a common schema with AI help, and an XGBoost model is trained per deployment and versioned. Every prediction is made against the failure limits of the belt's own type.

---

## Who uses it

| Role | Does | Where |
|---|---|---|
| **Planner** | Registers belts, records inspections, sees RUL and health, records failures and replacements | **Fleet** → belt page |
| **Engineer** | Everything a planner can do, plus upload data, review column mappings, train, compare and activate model versions, manage belt types and users | All tabs |

### Typical flow
1. **Engineer:** confirms the failure limits of each **belt type**. The seeded limits are placeholders and are flagged until someone confirms them.
2. **Engineer:** uploads plant data (any CSV layout) under **Data & Training**. Gemini proposes how the columns map to the model's fields, including unit conversions. The engineer reviews the proposal, sees any warnings, and approves it.
3. **Engineer:** trains a new model version, then compares it with earlier versions on the **same held-out belts** and **activates** it for its belt types. Older versions stay available for one-click rollback.
4. **Planner:** opens **Fleet**, which lists the most urgent belts first. They record an inspection (or **import many from a CSV**) and get RUL, health and the measurement closest to its limit.
5. **Planner:** when a belt fails or is replaced, records the hour and reason. This is the ground truth that improves the next model version.
6. **Engineer:** checks **Models → Real-world accuracy**: for every belt that actually failed, what each model version predicted at each inspection vs. what happened. When enough belts have ended, they click **Create from fleet records** (Data & Training) to turn the fleet's own history into a training dataset and train the next version on it.

### Bulk import (Fleet → Import from CSV)
Planners and engineers can load belts and inspections from a CSV. Download the template from the import panel. The file is **checked first**: every column is shown with the field and unit it was read as, unrecognised columns are called out, and each bad row is listed with the reason. **Nothing is saved until the file has no errors and you confirm**; re-importing a corrected file is safe (an inspection is identified by belt + hour and is replaced, not duplicated). Each imported inspection gets a stored prediction (model version and interval), exactly like one entered by hand.

### How the accuracy figures are judged
- Only belts recorded as **failed** count. A belt that was **replaced** only proves it lasted *at least* that long, so it is listed separately and never scored as a miss.
- The prediction judged is the one **stored at inspection time**, never a re-prediction with today's model.
- Figures based on fewer than 10 failed belts are flagged as anecdotal.

---

## Architecture

```
backend/
  api.py                 FastAPI app: REST API + serves the frontend (same origin)
  auth.py                Users, scrypt passwords, cookie sessions, roles
  db.py                  SQLite schema: users, belt types, belts, inspections, model versions
  model_registry.py      Train / version / activate / compare / predict (XGBoost, native format)
  schema_mapper.py       Column mapping (Gemini + fallback), validation, feature derivation, RUL derivation
  canonical_schema.json  Field names, units, plausible ranges of the model schema
  config.py              Settings and .env loading
  manage.py              Admin CLI (create the first user, reset passwords)
frontend/                index.html, app.js, style.css (no build step)
training/
  generate_synthetic_vbelts.py   Demo data generator (dev mode only); --censor-frac simulates still-running belts
tests/
  e2e_flow.py                    Backend end-to-end check: train, activate, compare, predict
docs/DATA_REQUIREMENTS.md        What the plant should record (share with the client)
instance/                        Runtime state, git-ignored: datasets/, mappings/, models/, belt_rul.db
```

### Modelling decisions
- **XGBoost only.** On held-out belts it beat the former XGBoost + Random Forest ensemble.
- **Belts held out for testing.** Test belts are picked by a hash of the belt ID, so a test belt is never in training. This also lets two versions be compared on the same belts.
- **No fine-tuning.** To add data, train a new version on the old datasets plus the new one (**Retrain + data**).
- **One model for several belt types**, with belt type as an input. Each type keeps its own failure limits.
- **Calculated stress features:** belt speed, bending frequency and speed ratio, computed from the drive geometry.
- **Survival analysis.** Models use XGBoost's `survival:aft` objective. Belts with a recorded failure/replacement hour (or a measured RUL column) are events; belts still running or replaced before failure are *right-censored* at their last hour — never extrapolated. Predictions are a median RUL plus an 80 % interval. A legacy point-regression method (`method: "regression"` on train) remains available. If a belt's failure is inferred from an assumed limit being crossed, that is flagged as *estimated*.

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
cp .env.example .env            # then set GEMINI_API_KEY
python backend/manage.py create-user <your-name> --role engineer
python backend/api.py
```

The app is then available at http://127.0.0.1:8001. Engineers add further users under **Users**.

### Local development
Set `BELT_RUL_DEV_MODE=1` in `.env`. This enables:
- the synthetic V-belt generator (**Data & Training**);
- `/docs`;
- cookies over plain HTTP.

**Never enable dev mode on the hosted server.**

### Hosting (production)
- Run behind a reverse proxy with **HTTPS** (nginx, Caddy or IIS). Session cookies are `Secure`, so login only works over HTTPS.
- The app listens on `127.0.0.1:8001` by default (`BELT_RUL_HOST` / `BELT_RUL_PORT`). Expose only the proxy.
- Set `BELT_RUL_INSTANCE_DIR` to a persistent, backed-up folder. It holds the client's data, the models and the database.
- Run a single worker process. Training runs in a background thread, and SQLite plus the model cache are per-process.
- Keep `.env` readable only by the service account.

---

### Checking your changes
```bash
python training/generate_synthetic_vbelts.py --belts 300 --censor-frac 0.3 --output synth.csv
python tests/e2e_flow.py synth.csv      # uses a throw-away instance folder; prints held-out metrics
python tests/test_phase4.py             # bulk import, real-failure backtest, fleet -> dataset, DB migration
```
Held-out metrics reported per model version (Models → Details): MAE, RMSE, R², bias, near-failure MAE, **concordance index** (does the model rank belts by remaining life correctly; 0.5 = random) and **80 % interval coverage** (should be about 80 %). Error figures use belts with a known end of life only.

### What to record for good predictions
See `docs/DATA_REQUIREMENTS.md`. In short: the operating hour at which each belt failed or was replaced. Belts that are still running or were replaced before failing are fine as well: they are right-censored, not wasted. If your CSV has a failure-hour column, leave it **blank** for belts that did not fail; do not fill it with the replacement hour unless the belt had effectively failed.

---

## Data privacy
During column mapping, Google Gemini receives the column names, basic statistics and up to 5 sample values per column. Full datasets never leave the server, and training and prediction run locally. Without `GEMINI_API_KEY`, the app falls back to name matching, and the engineer maps the columns by hand.

## Known limitations
- **Placeholder failure limits.** The seeded belt-type limits are not JK Fenner figures. Confirm them before relying on health scores.
- **Synthetic demo data.** The synthetic data is not calibrated against real belt tests. Its accuracy figures say nothing about real-world performance.
- **Interval calibration.** The 80 % interval uses one noise scale (sigma, chosen on a validation split) for all belts, so its width is proportional to the median RUL. Check the held-out interval coverage under Models → Details.
