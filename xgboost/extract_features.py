"""
extract_features.py — Sliding-window feature extraction from raw latency data.

Reads:   data/train.csv, data/test.csv
Writes:  data/train_features.csv, data/test_features.csv

Windowing:
  - Window size : 30 seconds
  - Step size   : 15 seconds (50% overlap)

Features per window (9):
  mean, p50, p75, p90, p99, p99_9, p99_99, max, stddev

Label rule:
  label = 1 (poor) if ANY 5-second sub-window within the 30s window
          contains 3 or more spikes with latency_ms > 10
  label = 0 (nominal) otherwise
"""

import numpy as np
import pandas as pd

WINDOW_S    = 30.0   # window size in seconds
STEP_S      = 15.0   # step size in seconds
SUBWINDOW_S = 5.0    # sub-window size for labelling
SPIKE_MS    = 10.0   # latency threshold for a spike
SPIKE_COUNT = 3      # minimum spikes in a sub-window to trigger label=1


def extract_features(window: pd.Series) -> dict:
    """
    Given a pandas Series of latency values for a single window,
    return a dict of the 9 features.
    """
    return {
        "mean"   : window.mean(),
        "p50"    : window.quantile(0.50),
        "p75"    : window.quantile(0.75),
        "p90"    : window.quantile(0.90),
        "p99"    : window.quantile(0.99),
        "p99_9"  : window.quantile(0.999),
        "p99_99" : window.quantile(0.9999),
        "max"    : window.max(),
        "stddev" : window.std(),
    }


def label_window(window_df: pd.DataFrame) -> int:
    """
    Given a DataFrame slice (columns: timestamp_ms, latency_ms) for a 30s window,
    return 1 if any 5-second sub-window contains SPIKE_COUNT or more spikes,
    otherwise 0.

    A spike is any row where latency_ms > SPIKE_MS.
    """
    t_start = window_df["timestamp_ms"].iloc[0]
    t_end   = window_df["timestamp_ms"].iloc[-1]

    sub_start = t_start
    while sub_start < t_end:
        sub_end  = sub_start + SUBWINDOW_S * 1000  # convert s → ms
        sub_mask = (window_df["timestamp_ms"] >= sub_start) & \
                   (window_df["timestamp_ms"] <  sub_end)
        sub_df   = window_df[sub_mask]

        spike_count = (sub_df["latency_ms"] > SPIKE_MS).sum()
        if spike_count >= SPIKE_COUNT:
            return 1

        sub_start = sub_end

    return 0


def process_file(input_path: str, output_path: str) -> None:
    print(f"Processing {input_path}...")
    df = pd.read_csv(input_path)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    t_min = df["timestamp_ms"].iloc[0]
    t_max = df["timestamp_ms"].iloc[-1]

    rows = []
    window_start = t_min

    while window_start + WINDOW_S * 1000 <= t_max:
        window_end = window_start + WINDOW_S * 1000

        mask       = (df["timestamp_ms"] >= window_start) & \
                     (df["timestamp_ms"] <  window_end)
        window_df  = df[mask]

        if len(window_df) < 2:
            window_start += STEP_S * 1000
            continue

        features = extract_features(window_df["latency_ms"])
        label    = label_window(window_df)

        rows.append({**features, "label": label, "window_start_ms": window_start})
        window_start += STEP_S * 1000

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    pos = (out_df["label"] == 1).sum()
    neg = (out_df["label"] == 0).sum()
    print(f"  {len(out_df)} windows → {output_path}  (label=1: {pos}, label=0: {neg})")


def main():
    process_file("data/train.csv", "data/train_features.csv")
    process_file("data/test.csv",  "data/test_features.csv")


if __name__ == "__main__":
    main()
