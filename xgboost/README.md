# XGBoost Time-Series Performance Detection

A hands-on learning exercise: train an XGBoost binary classifier to detect windows of poor latency performance in time-series data.

The scaffolding (synthetic data generation, feature extraction, validation) is provided. **You write the training and evaluation code.**

---

## File Layout

```
xgboost/
├── README.md
├── REQUIREMENTS.md            # Full spec: data format, label rules, feature definitions
├── generate_data.py           # PROVIDED — generates synthetic latency CSVs
├── extract_features.py        # PROVIDED — sliding-window feature extraction
├── validate.py                # PROVIDED — runs after you're done; checks your work
├── train.py                   # YOU WRITE — load features, train XGBoost, save model
├── evaluate.py                # YOU WRITE — load model, run inference, print metrics
└── data/                      # created when you run the scripts (git-ignored)
    ├── train.csv
    ├── test.csv
    ├── train_features.csv
    └── test_features.csv
```

---

## Environment Setup

Requires Python 3.

```bash
pip install numpy pandas xgboost scikit-learn
```

macOS also requires:

```bash
brew install libomp
```

---

## How to Use This Project

### Step 1 — Generate synthetic data

```bash
python3 generate_data.py
```

Creates `data/train.csv` (~10 min, ~120k rows) and `data/test.csv` (~3 min, ~36k rows). Each file has two columns: `timestamp_ms` and `latency_ms`.

The data models two nominal latency regimes (fast ~0.3ms, slow ~5ms) with random "degradation episodes" that inject bursts of spikes above 10ms.

### Step 2 — Extract features

```bash
python3 extract_features.py
```

Applies a 30-second sliding window (15s step) over the raw data and computes 9 statistical features per window. Labels each window 0 (nominal) or 1 (poor). Writes `data/train_features.csv` and `data/test_features.csv`.

Feature columns: `mean, p50, p75, p90, p99, p99_9, p99_99, max, stddev`
Label column: `label`
(Ignore `window_start_ms` when training — it's metadata, not a feature.)

### Step 3 — Write `train.py`

Open `train.py` and follow the numbered comments. You need to:
- Load `data/train_features.csv`
- Build an `xgb.DMatrix` from the 9 feature columns
- Configure params and call `xgb.train()`
- Save the model to `model.json`

### Step 4 — Write `evaluate.py`

Open `evaluate.py` and follow the numbered comments. You need to:
- Load `data/test_features.csv` and `model.json`
- Run `model.predict()` to get probabilities, then threshold at 0.5
- Print accuracy, precision, recall, and a confusion matrix

### Step 5 — Validate your solution

```bash
python3 validate.py
```

Runs 15 checks across both files — model correctness, metric thresholds, and output format. Each failed check includes a hint. Aim for all 15 passing.

---

## What the Label Rule Means

A 30-second window is labelled **poor (1)** if any 5-second sub-window within it contains 3 or more latency spikes above 10ms. Otherwise it's **nominal (0)**.

The training set has roughly a 50/50 class split, so the model has balanced examples to learn from.

---

## Reference

See `REQUIREMENTS.md` for the complete specification: data generation parameters, exact label rules, feature definitions, and validator check descriptions.
