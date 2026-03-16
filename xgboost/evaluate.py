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
#
#    The test set is a held-out slice of the synthetic data that the model has
#    never seen during training. Evaluating on it gives you an unbiased
#    estimate of how well the model will perform on new, real-world data.
#    If you only measured accuracy on the training set, a model that memorised
#    the training data would look perfect — even if it generalised terribly.
#
#    Use pd.read_csv("data/test_features.csv").
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Split into X_test and y_test
#
#    Apply exactly the same column split you used in train.py: select
#    FEATURE_COLS for X_test and "label" for y_test. The column names must
#    match what the model was trained on, or XGBoost will raise an error.
#    Do not include window_start_ms.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Load the saved model
#
#    XGBoost's Booster is the base model class. You create an empty one and
#    then populate it from disk using load_model. This is the complement of
#    save_model in train.py — the model file encodes the full tree structure,
#    so loading it reconstructs the exact same model state without retraining.
#
#      model = xgb.Booster()
#      model.load_model("model.json")
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Run inference
#
#    XGBoost's predict() method returns a numpy array of floats, not integers.
#    Each value is the model's estimated probability that the window belongs to
#    class 1 (poor performance) — i.e., P(label=1 | features). To turn these
#    probabilities into binary yes/no predictions, you apply a threshold.
#
#    The standard threshold is 0.5: predict 1 if the probability is >= 0.5,
#    otherwise predict 0. But this is a tunable decision. In a real monitoring
#    system you might lower the threshold to catch more true positives at the
#    cost of more false alarms, or raise it if false alarms are expensive.
#
#    Steps:
#      1. Wrap X_test in a DMatrix: dtest = xgb.DMatrix(X_test)
#      2. Get probabilities: probs = model.predict(dtest)
#      3. Threshold: preds = (probs >= 0.5).astype(int)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Print metrics
#
#    Four metrics give you a complete picture of classifier performance:
#
#    Accuracy — the fraction of windows predicted correctly overall. Easy to
#    understand, but can be misleading if classes are imbalanced.
#
#    Precision — of all the windows you flagged as "poor performance", what
#    fraction actually were? High precision means few false alarms.
#
#    Recall — of all the windows that were truly "poor performance", what
#    fraction did you catch? High recall means few missed incidents.
#
#    There is a tension between precision and recall: lowering the threshold
#    increases recall (you catch more real incidents) but decreases precision
#    (you also raise more false alarms). Which matters more depends on context.
#    In a latency monitoring system, missing a real incident (false negative)
#    is usually more costly than a spurious alert (false positive) — but think
#    about what the tradeoff looks like for your use case.
#
#    Confusion matrix layout (for reference):
#
#                 Predicted 0   Predicted 1
#      Actual 0  [    TN            FP    ]
#      Actual 1  [    FN            TP    ]
#
#    Use the sklearn functions already imported at the top of this file:
#    accuracy_score, precision_score, recall_score, confusion_matrix.
#    Print each metric with a label so the output is human-readable.
# ---------------------------------------------------------------------------
