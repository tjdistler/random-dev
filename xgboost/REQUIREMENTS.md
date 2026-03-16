# XGBoost Time-Series Performance Detection

## Project Goal

Build a binary classifier using XGBoost to detect "poor performance" windows in latency time-series data. This is a learning exercise — scaffolding and synthetic data are provided; you write the training and evaluation code, then validate your solution.

---

## Input Data Format

Raw time-series data representing per-request latency values:

| Column | Type | Description |
|---|---|---|
| `timestamp_ms` | float | Unix timestamp in milliseconds |
| `latency_ms` | float | Per-request latency in milliseconds |

- Event-driven Poisson arrivals (~200 req/s), not fixed rate
- Dense enough to observe sub-millisecond variance

---

## Synthetic Data Generation (`generate_data.py`) — PROVIDED

Generates two datasets under `data/`.

**Nominal latency** randomly alternates between two baseline regimes:
- **Fast regime**: ~0.3ms mean, ~0.05ms stddev
- **Slow regime**: ~5ms mean, ~1ms stddev
- Random transitions (exponentially distributed, ~20s mean duration)

**Degradation episodes** are injected as bursts rather than uniform random spikes:
- ~0.8 episodes per minute, each lasting ~8 seconds on average
- During an episode, 10% of requests become spikes (latency > 10ms, mean ~25ms)
- Outside episodes, spikes are effectively zero
- This produces a realistic ~50/50 class split in the feature windows

**Outputs:**
- `data/train.csv` — ~10 minutes (~120,000 rows)
- `data/test.csv` — ~3 minutes (~36,000 rows)

---

## Feature Extraction (`extract_features.py`) — PROVIDED

Converts raw time-series into a labeled feature matrix using a sliding window.

**Windowing parameters:**
- Window size: 30 seconds
- Step size: 15 seconds (50% overlap)

**Features per window (9 total):**

| Column | Description |
|---|---|
| `mean` | Mean latency |
| `p50` | 50th percentile |
| `p75` | 75th percentile |
| `p90` | 90th percentile |
| `p99` | 99th percentile |
| `p99_9` | 99.9th percentile |
| `p99_99` | 99.99th percentile |
| `max` | Maximum latency |
| `stddev` | Standard deviation |

**Additional columns (do not use as features):**
- `label` — the target variable
- `window_start_ms` — timestamp of the window start

**Label rule:**
- `label = 1` (poor) if any 5-second sub-window within the 30s window contains **3 or more spikes > 10ms**
- `label = 0` (nominal) otherwise

**Outputs:**
- `data/train_features.csv` — ~39 windows (~50/50 class split)
- `data/test_features.csv` — ~11 windows

---

## What You Implement

### `train.py`

1. Load `data/train_features.csv`
2. Split into `X` (the 9 feature columns) and `y` (`label`); drop `window_start_ms`
3. Construct an `xgb.DMatrix`
4. Define training parameters and call `xgb.train()`
5. Save the model with `model.save_model("model.json")`

Suggested starting parameters: `objective='binary:logistic'`, `eval_metric='logloss'`, `max_depth=4`, `eta=0.1`, `num_boost_round=100`.

### `evaluate.py`

1. Load `data/test_features.csv`
2. Load `model.json` into an `xgb.Booster`
3. Run `model.predict()` — this returns probabilities, not class labels
4. Threshold at 0.5 to get binary predictions
5. Print: accuracy, precision, recall, confusion matrix

Think about what false positives and false negatives mean in a real monitoring system — which is more costly?

---

## Validation

Run `validate.py` after completing both files:

```bash
python3 validate.py
```

It checks 15 things across two parts:

**Part 1 — train.py (8 checks):**
- `model.json` exists
- `model.json` loads as a valid `xgb.Booster`
- Model was trained on the correct 9 features
- Model predicts both classes (not degenerate)
- Training accuracy ≥ 85%
- Test accuracy ≥ 70%
- Test recall ≥ 0.60
- Test precision ≥ 0.60

**Part 2 — evaluate.py (7 checks):**
- `evaluate.py` exists
- Runs without error
- Prints numeric output
- Output mentions: `accuracy`, `precision`, `recall`, `confusion`

---

## Project Structure

```
xgboost/
├── REQUIREMENTS.md
├── README.md
├── generate_data.py       # PROVIDED — synthetic time-series generation
├── extract_features.py    # PROVIDED — sliding-window feature extraction
├── validate.py            # PROVIDED — submission validator (run last)
├── train.py               # YOU WRITE — XGBoost model training
├── evaluate.py            # YOU WRITE — model evaluation
└── data/                  # created by generate_data.py / extract_features.py
    ├── train.csv
    ├── test.csv
    ├── train_features.csv
    └── test_features.csv
```

---

## Tech Stack

- Python 3
- `numpy`
- `pandas`
- `xgboost`
- `scikit-learn`

```bash
pip install numpy pandas xgboost scikit-learn
```

macOS users also need:
```bash
brew install libomp
```
