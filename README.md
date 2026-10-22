# Belt RUL Predictor — v2 Intelligence System

AI-powered Remaining Useful Life (RUL) prediction and degradation forecasting system for industrial conveyor belts. Built on an XGBoost regressor, trained on physics-informed synthetic degradation telemetry or on your plant's own data.

---

## 🚀 Key Features

- **Live Browser Dashboard**: Sleek, responsive web interface (`http://localhost:8001/ui`).
- **Belt Category & Profile Taxonomy**: Built-in support for **Flat Conveyor Belts**, **V-Belts (Classical/Wedge)**, **Ribbed / Multi-V Belts (Poly-V)**, and **Timing / Synchronous Belts**, plus dynamic registration for custom belt geometries.
- **Model Zoo & Checkpoint Registry**: Versioned model registry (`models_v2/models_registry.json`) tracking hyperparameters, metrics ($R^2$, MAE, RMSE), and champion status across belt profiles.
- **Dataset Decision Gate**: Choose between **Zero-Training Benchmarking** on the active champion model or **Training a Candidate Model** from scratch with algorithm choice (XGBoost, Random Forest, Gradient Boosting).
- **Intelligent Model Comparison Studio**: Side-by-side metric comparison, automated performance verdicts (*Good / Champion*, *Marginal*, *Degraded*), algorithmic trade-offs (advantages & disadvantages), and one-click promotion to champion.
- **Dynamic Progressive Confidence**: Confidence starts at baseline (~40%) on initial inspection and progressively grows toward model accuracy as more sequential inspection points are recorded.
- **Works With Any Dataset**: Upload any belt CSV. Column mappings are reviewed and validated before training.
- **FastAPI Backend**: Clean REST endpoints for metadata, belt profiles, model registry, benchmarking, and multi-point RUL inference.

---

## 🛠️ Architecture

```
v2/
├── backend/
│   ├── api.py                 # FastAPI REST backend & static file server
│   ├── model_registry.py      # Versioned model registry, comparison engine & benchmarks
│   ├── belt_types.json        # Standard & custom belt geometry profiles
│   ├── train_model.py         # Multi-algorithm RUL trainer (XGBoost, Random Forest, Gradient Boosting)
│   ├── schema_mapper.py       # Column mapping (Gemini proposal), validation, RUL derivation
│   └── canonical_schema.json  # The field names/units/ranges the model is trained on
├── frontend/
│   ├── index.html             # Web dashboard & Model Comparison Studio
│   ├── script.js              # Interactive UI state, decision gate & comparison logic
│   └── style.css              # Curated responsive design system & modal styles
├── training/
│   └── training_data_5_oct.py # Physics-informed synthetic dataset generator
├── data/
│   └── belt_rul_training_data.csv  # Active training dataset
├── mappings/                  # Human-approved column mappings (one per dataset)
├── docs/
│   └── DATA_REQUIREMENTS.md   # Telemetry specification for industrial plant data
└── models_v2/
    ├── models_registry.json   # Catalog of all trained models & active champion
    ├── rul_model.pkl          # Active production champion model
    ├── model_metadata.json    # Active champion metadata & statistics
    └── runs/                  # Versioned checkpoints per run (artifacts + metadata)
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

