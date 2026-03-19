#!/usr/bin/env python3
"""
will_algorithm.py

A simple example algorithm for `evaluator.py`.

Model type
-----------
Empirical calibration lookup:
  - Bin `game_elapsed_seconds` into `num_time_bins` buckets across regulation (0..2400s).
  - Bucket `kalshi_probability` into integer percent points (1..99).
  - For each (time_bin, prob_int) cell, estimate win rate from training data.

This is not meant to be a state-of-the-art model; it’s a lightweight baseline
so you can verify the evaluator pipeline end-to-end.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# --- Hyperparameters for bucketing / smoothing ---
NUM_TIME_BINS = 40  # match your heatmaps
ALPHA_SMOOTH = 1.0  # Laplace-like smoothing strength


# Learned state after fit()
_cell_win_rate: Dict[Tuple[int, int], float] = {}
_global_win_rate: float = 0.5


def _time_bin_index(game_elapsed_seconds: float) -> int:
    # 0..2400 regulation; anything outside gets clamped into the nearest bucket.
    # time bin size equals 2400 / 40 = 60 seconds.
    bin_size = 2400.0 / NUM_TIME_BINS
    idx = int(game_elapsed_seconds // bin_size)
    return int(max(0, min(NUM_TIME_BINS - 1, idx)))


def _prob_int(kalshi_probability: float) -> int:
    # Convert probability (0..1) to integer percent (1..99).
    p_int = int(round(kalshi_probability * 100.0))
    return int(max(1, min(99, p_int)))


def fit(train_df: pd.DataFrame) -> None:
    global _cell_win_rate, _global_win_rate

    if train_df.empty:
        _cell_win_rate = {}
        _global_win_rate = 0.5
        return

    # Expected columns: game_elapsed_seconds, kalshi_prob, team_won
    if "kalshi_prob" not in train_df.columns:
        if "win_prob_pct" in train_df.columns:
            train_df = train_df.copy()
            train_df["kalshi_prob"] = train_df["win_prob_pct"].astype(float) / 100.0
        else:
            raise ValueError("train_df must contain 'kalshi_prob' or 'win_prob_pct'")

    df = train_df.copy()
    df["time_bin"] = df["game_elapsed_seconds"].astype(float).map(_time_bin_index)
    df["prob_int"] = df["kalshi_prob"].astype(float).map(_prob_int)

    _global_win_rate = float(df["team_won"].astype(float).mean())

    grouped = (
        df.groupby(["time_bin", "prob_int"], observed=False)
        .agg(wins=("team_won", "sum"), n=("team_won", "count"))
        .reset_index()
    )

    # Posterior mean with smoothing toward global win rate.
    # mean = (wins + alpha*global) / (n + alpha)
    cell: Dict[Tuple[int, int], float] = {}
    for _, r in grouped.iterrows():
        tb = int(r["time_bin"])
        pi = int(r["prob_int"])
        wins = float(r["wins"])
        n = float(r["n"])
        cell[(tb, pi)] = float((wins + ALPHA_SMOOTH * _global_win_rate) / (n + ALPHA_SMOOTH))

    _cell_win_rate = cell


def predict_probability(game_elapsed_seconds: float, kalshi_probability: float) -> float:
    tb = _time_bin_index(float(game_elapsed_seconds))
    pi = _prob_int(float(kalshi_probability))
    val = _cell_win_rate.get((tb, pi))
    if val is None:
        # If we’ve never seen this region in training, fall back to Kalshi’s quote.
        # (You can change this to _global_win_rate if you prefer.)
        return float(kalshi_probability)
    return float(val)
