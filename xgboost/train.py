"""
train.py — Train an XGBoost binary classifier on the extracted features.

Run after extract_features.py:
  python3 train.py

Output: model.json (saved XGBoost model)
"""

import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# 1. Load the training features
#    Read data/train_features.csv into a DataFrame.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Split into X (features) and y (labels)
#    Feature columns: mean, p50, p75, p90, p99, p99_9, p99_99, max, stddev
#    Label column: label
#    Drop window_start_ms — it's not a feature.
# ---------------------------------------------------------------------------

FEATURE_COLS = ["mean", "p50", "p75", "p90", "p99", "p99_9", "p99_99", "max", "stddev"]


# ---------------------------------------------------------------------------
# 3. Construct a DMatrix
#    XGBoost's native data structure. Pass X and label=y.
#    Docs: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Define training parameters
#    Good starting point:
#      "objective":   "binary:logistic"
#      "eval_metric": "logloss"
#      "max_depth":   4
#      "eta":         0.1   (learning rate)
#    Feel free to experiment — these are just defaults.
# ---------------------------------------------------------------------------

params = {
    # TODO
}

# ---------------------------------------------------------------------------
# 5. Train the model
#    Use xgb.train(params, dtrain, num_boost_round=...).
#    Try num_boost_round=100 to start.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Save the model
#    model.save_model("model.json")
# ---------------------------------------------------------------------------


print("Done. Model saved to model.json")
