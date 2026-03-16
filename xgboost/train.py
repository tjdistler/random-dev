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
#
#    The feature extraction step already did the hard work of converting raw
#    per-request latency events into a fixed-width table of summary statistics.
#    Each row represents one 30-second window. All you need to do here is read
#    that table into a pandas DataFrame so you can work with it.
#
#    Use pd.read_csv("data/train_features.csv").
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Split into X (features) and y (labels)
#
#    Machine learning models need two things: the input features (X) and the
#    ground-truth target (y). The feature matrix X is what the model learns
#    patterns from; y is what it's trying to predict.
#
#    The CSV has 11 columns. Nine of them are the statistical features you want
#    to train on (listed in FEATURE_COLS below). One is the label (0 = nominal,
#    1 = poor performance). One is window_start_ms, which is just a timestamp
#    — metadata that would leak temporal information if used as a feature, so
#    drop it.
#
#    Build X by selecting only FEATURE_COLS from the DataFrame, and y by
#    selecting the "label" column.
# ---------------------------------------------------------------------------

FEATURE_COLS = ["mean", "p50", "p75", "p90", "p99", "p99_9", "p99_99", "max", "stddev"]


# ---------------------------------------------------------------------------
# 3. Construct a DMatrix
#
#    XGBoost doesn't operate on raw pandas DataFrames — it uses its own
#    internal data structure called a DMatrix. A DMatrix bundles the feature
#    matrix and the labels together in a memory-efficient format that XGBoost's
#    C++ core can process directly.
#
#    Create one with: dtrain = xgb.DMatrix(X, label=y)
#
#    Passing label=y at construction time is important: it's what tells the
#    training loop what it's optimising against. If you forget it, XGBoost will
#    train without a target and produce nonsense.
#
#    Docs: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Define training parameters
#
#    XGBoost is controlled by a plain Python dict of hyperparameters. The most
#    important ones for a binary classification problem:
#
#    "objective": "binary:logistic"
#        Tells XGBoost this is a binary classification task. The model will
#        output probabilities (floats between 0 and 1) rather than raw scores.
#
#    "eval_metric": "logloss"
#        The loss function used to evaluate each boosting round internally.
#        Log-loss penalises confident wrong predictions heavily, which
#        encourages well-calibrated probability outputs.
#
#    "max_depth": 4
#        Controls how deep each individual decision tree can grow. Deeper trees
#        can capture more complex patterns but are also more prone to
#        overfitting. Start at 4; try 3 if the model overfits, 6 if it
#        underfits.
#
#    "eta": 0.1
#        The learning rate — how much weight each new tree gets when it's added
#        to the ensemble. Smaller values mean more trees are needed to reach a
#        good solution, but the final model tends to generalise better. 0.1 is
#        a safe default.
#
#    These are just suggested starting values. After getting a baseline working,
#    experiment: What happens if you double max_depth? Halve eta?
# ---------------------------------------------------------------------------

params = {
    # TODO
}

# ---------------------------------------------------------------------------
# 5. Train the model
#
#    XGBoost builds an ensemble of decision trees one at a time, where each new
#    tree corrects the residual errors of the previous ones. This is called
#    gradient boosting. The num_boost_round argument controls how many trees
#    are built in total — it's the most direct lever on model capacity.
#
#    Use: model = xgb.train(params, dtrain, num_boost_round=100)
#
#    100 rounds is a reasonable starting point for a small dataset like this
#    (~39 training windows). Too few rounds and the model won't converge; too
#    many and it may overfit. Once you have it working, try varying this value
#    and observing the effect on training vs. test accuracy in validate.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Save the model
#
#    Saving the model to disk decouples training from evaluation: you can train
#    once and run evaluate.py many times, or share the model file with someone
#    else without them needing to retrain. XGBoost's JSON format is human-
#    readable — open model.json in a text editor after saving and you'll see
#    the actual tree structure encoded inside it.
#
#    Use: model.save_model("model.json")
# ---------------------------------------------------------------------------


print("Done. Model saved to model.json")
