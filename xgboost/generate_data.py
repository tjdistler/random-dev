"""
generate_data.py — Synthetic latency time-series generator.

Produces two CSV files:
  data/train.csv  (~10 minutes of data)
  data/test.csv   (~3 minutes of data)

Each CSV has columns: timestamp_ms, latency_ms

Latency behaviour:
  - Alternates between two nominal regimes (fast ~0.3ms, slow ~5ms)
  - "Degradation episodes" injected randomly: during an episode the spike
    rate is high (many requests > 10ms), outside episodes it is near zero.
    This ensures a realistic mix of labelled-1 and labelled-0 windows.
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Nominal regime parameters
FAST_MEAN_MS = 0.3
FAST_STD_MS  = 0.05
SLOW_MEAN_MS = 5.0
SLOW_STD_MS  = 1.0

# Regime transitions (geometric distribution of duration in seconds)
REGIME_MEAN_DURATION_S = 20.0

# Degradation episode parameters
EPISODE_RATE_PER_MIN  = 0.8     # expected episodes per minute
EPISODE_MEAN_DURATION = 8.0     # seconds; long enough to affect several sub-windows
EPISODE_SPIKE_PROB    = 0.10    # 10% of requests during an episode are spikes

# Spike shape
SPIKE_MEAN_MS = 25.0
SPIKE_STD_MS  = 10.0
SPIKE_MIN_MS  = 10.001          # spikes must exceed 10 ms

# Approximate request rate (requests per second)
REQUESTS_PER_SEC = 200


def _regime_sequence(duration_seconds: float):
    """Yield (t_start_s, t_end_s, regime_id) segments covering [0, duration_seconds]."""
    t = 0.0
    regime = 0  # 0=fast, 1=slow
    while t < duration_seconds:
        dur = RNG.exponential(REGIME_MEAN_DURATION_S)
        yield (t, min(t + dur, duration_seconds), regime)
        t += dur
        regime = 1 - regime


def _episode_mask(timestamps_ms: np.ndarray, duration_seconds: float) -> np.ndarray:
    """Boolean array: True where a request falls inside a degradation episode."""
    mask = np.zeros(len(timestamps_ms), dtype=bool)
    t = 0.0
    episode_interval_s = 60.0 / EPISODE_RATE_PER_MIN  # mean seconds between episode starts
    while t < duration_seconds:
        # Wait for next episode start
        t += RNG.exponential(episode_interval_s)
        if t >= duration_seconds:
            break
        episode_end = t + RNG.exponential(EPISODE_MEAN_DURATION)
        ep_mask = (timestamps_ms >= t * 1000) & (timestamps_ms < episode_end * 1000)
        mask |= ep_mask
        t = episode_end
    return mask


def generate_series(duration_seconds: float, start_ts_ms: float = 0.0) -> pd.DataFrame:
    """Generate a single latency time-series of the given duration."""
    n_requests = int(duration_seconds * REQUESTS_PER_SEC)

    # Poisson arrivals (exponential inter-arrivals)
    inter_arrival_ms = RNG.exponential(1000.0 / REQUESTS_PER_SEC, size=n_requests)
    timestamps_ms = start_ts_ms + np.cumsum(inter_arrival_ms)

    # Regime-based nominal latencies
    regime_per_request = np.zeros(n_requests, dtype=int)
    for (t_start, t_end, regime) in _regime_sequence(duration_seconds):
        seg_mask = (timestamps_ms >= start_ts_ms + t_start * 1000) & \
                   (timestamps_ms <  start_ts_ms + t_end   * 1000)
        regime_per_request[seg_mask] = regime

    latencies = np.empty(n_requests)
    fast_mask = regime_per_request == 0
    slow_mask = regime_per_request == 1
    latencies[fast_mask] = RNG.normal(FAST_MEAN_MS, FAST_STD_MS, fast_mask.sum()).clip(min=0.01)
    latencies[slow_mask] = RNG.normal(SLOW_MEAN_MS, SLOW_STD_MS, slow_mask.sum()).clip(min=0.01)

    # Degradation episodes — bursty spike injection
    in_episode = _episode_mask(timestamps_ms - start_ts_ms, duration_seconds)
    spike_coin  = RNG.random(n_requests)
    spike_mask  = in_episode & (spike_coin < EPISODE_SPIKE_PROB)
    n_spikes    = spike_mask.sum()
    latencies[spike_mask] = RNG.normal(SPIKE_MEAN_MS, SPIKE_STD_MS, n_spikes).clip(min=SPIKE_MIN_MS)

    return pd.DataFrame({"timestamp_ms": timestamps_ms, "latency_ms": latencies})


def main():
    os.makedirs("data", exist_ok=True)

    print("Generating train.csv (10 minutes)...")
    train_df = generate_series(duration_seconds=600.0, start_ts_ms=0.0)
    train_df.to_csv("data/train.csv", index=False)
    print(f"  {len(train_df):,} rows → data/train.csv")

    test_start_ms = train_df["timestamp_ms"].iloc[-1] + 1000.0

    print("Generating test.csv (3 minutes)...")
    test_df = generate_series(duration_seconds=180.0, start_ts_ms=test_start_ms)
    test_df.to_csv("data/test.csv", index=False)
    print(f"  {len(test_df):,} rows → data/test.csv")

    spike_count = (train_df["latency_ms"] > 10).sum()
    print(f"\nSanity check — train spikes (>10ms): {spike_count} "
          f"({100 * spike_count / len(train_df):.2f}%)")


if __name__ == "__main__":
    main()
