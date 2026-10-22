"""
Comprehensive Test Suite for Model Studio & Belt Type Intelligence
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add backend to path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_DIR)

import model_registry as mr
from api import app

client = TestClient(app)

def test_belt_types_loading():
    """Verify standard industrial and mechanical drive belt types are loaded."""
    types_data = mr.load_belt_types()
    assert "types" in types_data
    types = types_data["types"]
    assert "flat_conveyor" in types
    assert "v_belt" in types
    assert "ribbed_multi_v" in types
    assert "timing_toothed" in types

    v_belt = types["v_belt"]
    assert "Trapezoidal" in v_belt["geometry"]
    assert v_belt["nominal_hardness"] > 0
    assert len(v_belt["typical_failure_modes"]) > 0


def test_add_custom_belt_type():
    """Verify adding a custom user-defined belt type."""
    custom = {
        "id": "test_hexagonal_double_v",
        "name": "Hexagonal Double-V Belt",
        "geometry": "Double-V Hexagonal",
        "icon": "🔷",
        "nominal_hardness": 72.0,
        "nominal_thickness": 14.0,
        "nominal_tensile": 900.0,
        "typical_failure_modes": ["Dual Sidewall Fatigue", "Flex Glazing"],
        "description": "Dual-sided V-drive for serpentine counter-rotating shafts."
    }
    added = mr.add_custom_belt_type(custom)
    assert added["id"] == "test_hexagonal_double_v"
    assert added["is_custom"] is True

    # Confirm persistence
    types = mr.load_belt_types()["types"]
    assert "test_hexagonal_double_v" in types


def test_model_registry_initialization():
    """Verify model registry has an active champion."""
    reg = mr.get_model_registry()
    assert "models" in reg
    assert len(reg["models"]) > 0
    assert reg["active_champion_id"] is not None


def test_algorithm_tradeoffs():
    """Verify pros and cons exist for all supported algorithms."""
    for algo in ["xgboost", "random_forest", "gradient_boosting"]:
        profile = mr.get_algorithm_tradeoffs(algo)
        assert "advantages" in profile
        assert "disadvantages" in profile
        assert len(profile["advantages"]) > 0
        assert len(profile["disadvantages"]) > 0


def test_model_comparison():
    """Verify intelligent comparison computes deltas and verdicts."""
    candidate = {
        "model_id": "test_candidate_rf",
        "algorithm": "random_forest",
        "belt_type": "flat_conveyor",
        "dataset": "belt_rul_training_data.csv",
        "training_rows": 10000,
        "metrics": {
            "random_forest": {
                "r2_score": 0.965,
                "mae_hours": 320.0,
                "rmse_hours": 450.0
            }
        }
    }
    comp = mr.compare_models(candidate)
    assert "verdict" in comp
    assert "deltas" in comp or comp["baseline"] is None
    assert "recommendation" in comp
    assert len(comp["recommendation"]) > 0


def test_zero_training_benchmark():
    """Verify benchmark on canonical dataset."""
    csv_path = os.path.join(BACKEND_DIR, "..", "data", "belt_rul_training_data.csv")
    if os.path.exists(csv_path):
        bench = mr.benchmark_dataset_on_model(csv_path)
        assert bench["rows_evaluated"] > 0
        assert "r2_score" in bench["metrics"]
        assert len(bench["samples"]) > 0


def test_api_endpoints():
    """Test FastAPI endpoints via TestClient."""
    # 1. GET /belt-types
    r = client.get("/belt-types")
    assert r.status_code == 200
    assert "types" in r.json()

    # 2. GET /models
    r = client.get("/models")
    assert r.status_code == 200
    assert "models" in r.json()

    # 3. GET /metadata
    r = client.get("/metadata")
    assert r.status_code == 200

    # 4. POST /models/compare
    r = client.post("/models/compare", json={
        "candidate_meta": {
            "model_id": "api_test_cand",
            "algorithm": "xgboost",
            "metrics": {"xgboost": {"r2_score": 0.95, "mae_hours": 350.0, "rmse_hours": 480.0}}
        }
    })
    assert r.status_code == 200
    assert "comparison" in r.json()


if __name__ == "__main__":
    test_belt_types_loading()
    test_add_custom_belt_type()
    test_model_registry_initialization()
    test_algorithm_tradeoffs()
    test_model_comparison()
    test_zero_training_benchmark()
    test_api_endpoints()
    print("\n[OK] All Model Studio & Belt Type tests passed successfully!")
