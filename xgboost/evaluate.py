"""
evaluate.py — Evaluate the trained model on the test set.

Run after train.py:
  python3 evaluate.py

Prints: accuracy, precision, recall, confusion matrix
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

FEATURE_COLS = ["mean", "p50", "p75", "p90", "p99", "p99_9", "p99_99", "max", "stddev"]

# ---------------------------------------------------------------------------
# 1. Load the test features
#    Read data/test_features.csv into a DataFrame.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Split into X_test and y_test (same column split as train.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Load the saved model
#    model = xgb.Booster()
#    model.load_model("model.json")
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Run inference
#    XGBoost returns probabilities, not class labels.
#    model.predict() on a DMatrix gives a float array of P(label=1).
#    Convert to binary predictions with a threshold — 0.5 is the standard
#    starting point, but you can tune this.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Print metrics
#    Use the sklearn functions imported above.
#    Confusion matrix layout:
#
#                 Predicted 0   Predicted 1
#      Actual 0  [ TN           FP ]
#      Actual 1  [ FN           TP ]
#
#    Think about what FP and FN mean here:
#      FP — flagged as poor performance when it was actually fine
#      FN — missed a real period of poor performance
#    Which is more costly in a real system?
# ---------------------------------------------------------------------------
