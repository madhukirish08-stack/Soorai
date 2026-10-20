# Belt RUL Predictor — v2 Intelligence System

AI-powered Remaining Useful Life (RUL) prediction and degradation forecasting system for industrial conveyor belts. Built on an XGBoost regressor, trained on physics-informed synthetic degradation telemetry or on your plant's own data.

---

## 🚀 Key Features

- **Live Browser Dashboard**: Sleek, responsive web interface (`http://localhost:8001/ui`).
- **Dynamic Progressive Confidence**: Confidence starts at baseline (~40%) on initial inspection and progressively grows toward model accuracy (test $R^2$ on belts held out from training, $\approx 0.89$ on the synthetic data) as more sequential inspection points are recorded.
- **Works With Any Dataset**: Upload any belt CSV. Gemini proposes how its columns (names, units) map to the model schema, an engineer reviews and approves, and the approved profile is reused for later uploads with the same columns.
- **Automatic RUL Target**: If the data has no remaining-life column, RUL is computed from recorded failure hours, or estimated from degradation trends (clearly flagged — see [docs/DATA_REQUIREMENTS.md](docs/DATA_REQUIREMENTS.md)).
- **Physical Clamping & Health Scoring**: Automatic degradation ratio calculations across wear (%), elongation (%), and crack length (mm) with dominant failure driver identification.
- **FastAPI Backend**: Clean REST endpoints for metadata, dataset listing, on-demand training, and multi-point RUL inference.
- **Interactive Visualization**: Real-time degradation gauges, RUL trend graphs (via Chart.js), and complete inspection history tables.

---

## 🛠️ Architecture

```
v2/
├── backend/
│   ├── api.py                 # FastAPI REST backend & static file server
│   ├── train_model.py         # XGBoost RUL trainer
│   ├── schema_mapper.py       # Column mapping (Gemini proposal), validation, RUL derivation
│   └── canonical_schema.json  # The field names/units/ranges the model is trained on
├── frontend/
│   ├── index.html         # Modern dark-theme web dashboard
│   ├── script.js          # Interactive UI state & chart logic
│   └── style.css          # Curated responsive design system
├── training/
│   └── training_data_5_oct.py   # Physics-informed synthetic dataset generator
├── data/
│   └── belt_rul_training_data.csv  # Active training dataset
├── mappings/                # Human-approved column mappings (one per dataset)
├── docs/
│   └── DATA_REQUIREMENTS.md # What the plant should record (share with the company)
└── models_v2/
    └── model_metadata.json  # Feature importance, training metrics & statistics
```

---

## 📦 Getting Started

### 1. Requirements
Ensure Python 3.9+ is installed:
```bash
pip install fastapi uvicorn xgboost scikit-learn pandas numpy joblib
```

### 2. (Optional) Enable AI column mapping
Set a Gemini API key so uploaded datasets get an AI-proposed column mapping. Without it, the app falls back to name matching and you map columns by hand.
```powershell
$env:GEMINI_API_KEY = "<your key>"
$env:GEMINI_MODEL   = "gemini-3.8-flash"   # optional, this is the default
```

### 3. Run the Web Application
```bash
python backend/api.py
```
Open your browser at:
👉 **[http://localhost:8001/ui](http://localhost:8001/ui)** (or simply `http://localhost:8001`)

Interactive API documentation is available at:
👉 **[http://localhost:8001/docs](http://localhost:8001/docs)**

---

## ⚙️ Model Training

You can train or retrain the model at any time:
1. **Via UI**: Select the dataset in Step 1. If its columns don't already use the model's names, click **"Review Column Mapping"**, check each row (target field, unit conversion `model = raw × scale + offset`), and **Approve**. Then click **"Train Model on Selected Dataset"**.
2. **Via CLI**:
   ```bash
   python backend/train_model.py --csv data/belt_rul_training_data.csv --output models_v2
   # non-canonical data: pass the mapping approved in the UI
   python backend/train_model.py --csv data/plant.csv --output models_v2 --mapping mappings/plant.csv.mapping.json
   ```

The model trains on whichever features the mapped data contains. When the data has no measured RUL or failure hours, the trained model is flagged **Estimated RUL**. Ask the plant to record failures ([docs/DATA_REQUIREMENTS.md](docs/DATA_REQUIREMENTS.md)).

---

