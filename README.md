# Belt RUL Predictor — v2 Intelligence System

AI-powered Remaining Useful Life (RUL) prediction and degradation forecasting system for industrial conveyor belts. Built with an ensemble of XGBoost and Random Forest models trained on physics-informed synthetic degradation telemetry.

---

## 🚀 Key Features

- **Live Browser Dashboard**: Sleek, responsive web interface (`http://localhost:8001/ui`).
- **Dynamic Progressive Confidence**: Confidence starts at baseline (~40%) on initial inspection and progressively grows toward model accuracy ($R^2 \approx 99.1\%$) as more sequential inspection points are recorded.
- **Physical Clamping & Health Scoring**: Automatic degradation ratio calculations across wear (%), elongation (%), and crack length (mm) with dominant failure driver identification.
- **FastAPI Backend**: Clean REST endpoints for metadata, dataset listing, on-demand training, and multi-point RUL inference.
- **Interactive Visualization**: Real-time degradation gauges, RUL trend graphs (via Chart.js), and complete inspection history tables.

---

## 🛠️ Architecture

```
v2/
├── backend/
│   ├── api.py            # FastAPI REST backend & static file server
│   └── train_model.py     # XGBoost + Random Forest ensemble trainer
├── frontend/
│   ├── index.html         # Modern dark-theme web dashboard
│   ├── script.js          # Interactive UI state & chart logic
│   └── style.css          # Curated responsive design system
├── training/
│   ├── training_data_5_oct.py   # Physics-informed synthetic dataset generator
│   └── life_utils.py     # Utility degradation equations
├── data/
│   └── belt_rul_training_data.csv  # Active training dataset
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

### 2. Run the Web Application
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
1. **Via UI**: Select the dataset in Step 1 of the dashboard and click **"Train Model on Selected Dataset"**.
2. **Via CLI**:
   ```bash
   python backend/train_model.py --csv data/belt_rul_training_data.csv --output models_v2
   ```

---

