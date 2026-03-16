"""
validate.py — Submission validator.

Run this after train.py and evaluate.py to check your work:
  python3 validate.py

Each test prints PASS or FAIL with an explanation.
All tests must pass for a complete solution.
"""

import subprocess
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

FEATURE_COLS = ["mean", "p50", "p75", "p90", "p99", "p99_9", "p99_99", "max", "stddev"]

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []


def check(name, passed, hint=""):
    status = PASS if passed else FAIL
    print(f"  [{status}] {name}")
    if not passed and hint:
        print(f"         hint: {hint}")
    results.append(passed)


# ---------------------------------------------------------------------------
# Part 1: train.py — does it produce a valid, well-trained model?
# ---------------------------------------------------------------------------

print("\n── Part 1: train.py ──────────────────────────────────────────────────\n")

# 1a. model.json exists
model_exists = os.path.exists("model.json")
check(
    "model.json was created",
    model_exists,
    "Run train.py first. It should save the model with model.save_model('model.json').",
)

model = None
if model_exists:
    # 1b. model loads without error
    try:
        model = xgb.Booster()
        model.load_model("model.json")
        check("model.json loads as a valid XGBoost Booster", True)
    except Exception as e:
        check("model.json loads as a valid XGBoost Booster", False, str(e))
        model = None

if model is not None:
    train_df = pd.read_csv("data/train_features.csv")
    test_df  = pd.read_csv("data/test_features.csv")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["label"]

    dtrain = xgb.DMatrix(X_train)
    dtest  = xgb.DMatrix(X_test)

    train_probs = model.predict(dtrain)
    test_probs  = model.predict(dtest)
    train_preds = (train_probs >= 0.5).astype(int)
    test_preds  = (test_probs  >= 0.5).astype(int)

    # 1c. Model uses the right features
    expected_features = set(FEATURE_COLS)
    try:
        model_features = set(model.feature_names or [])
        features_match = model_features == expected_features
    except Exception:
        features_match = False
    check(
        "Model was trained on the correct 9 features",
        features_match,
        f"Expected feature names: {sorted(expected_features)}. "
        "Make sure you pass FEATURE_COLS as X, not the full DataFrame.",
    )

    # 1d. Not predicting a single class (degenerate model)
    unique_train = set(train_preds)
    unique_test  = set(test_preds)
    not_degenerate = len(unique_train) > 1 or len(unique_test) > 1
    check(
        "Model predicts both classes (not degenerate)",
        not_degenerate,
        "Your model is predicting the same label for every window. "
        "Check your params — try lowering eta or increasing num_boost_round.",
    )

    # 1e. Training accuracy (model should fit training data well)
    train_acc = accuracy_score(y_train, train_preds)
    check(
        f"Training accuracy >= 85%  (yours: {train_acc:.0%})",
        train_acc >= 0.85,
        "The model isn't fitting the training data well. "
        "Try more boosting rounds or a deeper max_depth.",
    )

    # 1f. Test accuracy
    test_acc = accuracy_score(y_test, test_preds)
    check(
        f"Test accuracy >= 70%  (yours: {test_acc:.0%})",
        test_acc >= 0.70,
        "The model isn't generalising to the test set. "
        "This may be overfitting — try reducing max_depth or adding min_child_weight.",
    )

    # 1g. Test recall — catching actual poor-performance windows matters
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    check(
        f"Test recall >= 0.60  (yours: {test_recall:.2f})",
        test_recall >= 0.60,
        "The model is missing too many true poor-performance windows (false negatives). "
        "Try lowering the decision threshold below 0.5 in evaluate.py.",
    )

    # 1h. Test precision — not too many false alarms
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    check(
        f"Test precision >= 0.60  (yours: {test_prec:.2f})",
        test_prec >= 0.60,
        "The model is raising too many false alarms (false positives). "
        "It may be overfitting spikes in the nominal windows.",
    )

# ---------------------------------------------------------------------------
# Part 2: evaluate.py — does it run and produce visible output?
# ---------------------------------------------------------------------------

print("\n── Part 2: evaluate.py ───────────────────────────────────────────────\n")

evaluate_exists = os.path.exists("evaluate.py")
check(
    "evaluate.py exists",
    evaluate_exists,
    "Create evaluate.py in this directory.",
)

if evaluate_exists:
    result = subprocess.run(
        [sys.executable, "evaluate.py"],
        capture_output=True,
        text=True,
    )

    # 2a. Runs without error
    check(
        "evaluate.py runs without error",
        result.returncode == 0,
        result.stderr.strip() or "Check the traceback above.",
    )

    if result.returncode == 0:
        output = result.stdout.lower()

        # 2b. Prints something numeric (a metric)
        has_numbers = any(c.isdigit() for c in result.stdout)
        check(
            "evaluate.py prints numeric output (metrics)",
            has_numbers,
            "Make sure you print accuracy, precision, recall, and the confusion matrix.",
        )

        # 2c. Mentions key metric names
        for keyword in ["accuracy", "precision", "recall", "confusion"]:
            check(
                f"evaluate.py output mentions '{keyword}'",
                keyword in output,
                f"Print a line that includes the word '{keyword}'.",
            )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n── Summary ───────────────────────────────────────────────────────────\n")
passed = sum(results)
total  = len(results)
print(f"  {passed}/{total} checks passed\n")

if passed == total:
    print("  All checks passed. Nice work.")
else:
    print("  Some checks failed — see the hints above.")

print()
sys.exit(0 if passed == total else 1)
