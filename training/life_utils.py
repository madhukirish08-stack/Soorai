"""
Life Model Prediction Helper & Wrapper
======================================
Provides clean expected life, remaining life, and survival probability calculations
using the trained survival AFT booster and gradient boosting life regressor.
"""

import numpy as np
import xgboost as xgb

class UnifiedLifeModel:
    def __init__(self, aft_booster, life_regressor, feature_names=None, c_index=0.0, mae=0.0, rmse=0.0, r2=0.0):
        self.aft_booster = aft_booster
        self.life_regressor = life_regressor
        self.feature_names = feature_names or []
        self.c_index = c_index
        self.mae = mae
        self.rmse = rmse
        self.r2 = r2

    def predict_expected_life(self, X_encoded):
        preds = self.life_regressor.predict(X_encoded)
        return np.maximum(preds, 500.0)

    def predict_remaining_life(self, X_encoded, runtime_hours):
        expected = self.predict_expected_life(X_encoded)
        rul = expected - runtime_hours
        return np.maximum(rul, 0.0)

    def predict_survival_probability(self, X_encoded, current_runtime=None):
        dmat = xgb.DMatrix(X_encoded)
        expected_aft = self.aft_booster.predict(dmat)

        if current_runtime is None:
            survival_prob = 1.0 / (1.0 + np.exp((expected_aft - 20000.0) / 10000.0))
        else:
            scale = 0.50
            z = (np.log(np.maximum(current_runtime, 10.0)) - np.log(np.maximum(expected_aft, 10.0))) / scale
            survival_prob = 1.0 / (1.0 + np.exp(z))

        return np.clip(survival_prob, 0.01, 0.99)
