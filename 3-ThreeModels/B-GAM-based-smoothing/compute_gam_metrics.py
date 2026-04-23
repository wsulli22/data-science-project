#!/usr/bin/env python3
"""
Compute scalar GAM evaluation metrics aligned with the report / rubric:

  1. MACE-week — mean over calendar weeks of mean per-game absolute calibration
     error (percentage points), matching the construction behind Figure 2
     (accuracy_across_time.py).

  2. RMSE-smooth — sqrt of weighted mean squared error between GAM-smoothed
     and raw empirical win-rate cells (same pipeline as gam_true_win_heatmap.py).

  3. MASE — weighted mean absolute smoothed signed edge at observed cells
     (same pipeline as gam_edge_heatmap.py; weights = cell counts).
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from pygam import LinearGAM, s, te

# Re-use GAM structure from heatmap scripts
from gam_true_win_heatmap import (
    N_SPLINES_PROB as N_SPLINES_PROB_WIN,
    N_SPLINES_TIME as N_SPLINES_TIME_WIN,
    NUM_TIME_BINS_DEFAULT,
    _prepare_minute_grouped_empirical_win_pct,
)
from gam_edge_heatmap import (
    N_SPLINES_PROB as N_SPLINES_PROB_EDGE,
    N_SPLINES_TIME as N_SPLINES_TIME_EDGE,
    _prepare_minute_grouped_signed_edge,
)


def compute_mace_week(data_dir: str) -> float:
    """
    Per-game error_pp = |100 * empirical_win_rate - mean(round(win_prob_pct))|
    (team_1 YES-side), then per week mean(error_pp), then mean across weeks
    with at least one game (same week bucketing as accuracy_across_time.py).
    """
    pattern = os.path.join(data_dir, "week_*_games.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)

    wide = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    need = {"kalshi_event", "realworld_timestamp", "team_1", "team_1_win_prob_pct", "winning_team"}
    missing = need - set(wide.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = wide[list(need)].copy()
    df = df.rename(columns={"team_1_win_prob_pct": "win_prob_pct"})
    df["team_won"] = (df["team_1"] == df["winning_team"]).astype(int)
    df = df.dropna(subset=["kalshi_event", "realworld_timestamp", "win_prob_pct", "team_won"])
    df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"], errors="coerce")
    df = df.dropna(subset=["realworld_timestamp"])

    if df["win_prob_pct"].max(skipna=True) <= 1.0 + 1e-6:
        df = df.copy()
        df["win_prob_pct"] = df["win_prob_pct"] * 100.0

    df["prob_int"] = df["win_prob_pct"].round(0).astype(int)
    df = df[df["prob_int"].between(1, 99)]

    per_game = (
        df.groupby("kalshi_event", observed=False)
        .agg(
            game_start_ts=("realworld_timestamp", "min"),
            empirical_win_rate=("team_won", "mean"),
            kalshi_avg_prob_pct=("prob_int", "mean"),
        )
        .reset_index()
    )
    per_game["error_pp"] = (
        per_game["empirical_win_rate"] * 100.0 - per_game["kalshi_avg_prob_pct"]
    ).abs()
    per_game["week_start"] = per_game["game_start_ts"].dt.to_period("W-MON").dt.start_time

    weekly = (
        per_game.groupby("week_start", observed=False)
        .agg(mean_abs_error_pp=("error_pp", "mean"), n_games=("kalshi_event", "count"))
        .reset_index()
    )
    weekly = weekly[weekly["n_games"] >= 1]
    return float(weekly["mean_abs_error_pp"].mean())


def _fit_win_gam_and_rmse(data_dir: str, num_time_bins: int) -> float:
    (
        _df,
        win_raw,
        count_matrix,
        time_labels,
        probs,
        _time_edges,
        _num_time_bins,
        time_bin_to_frac,
        bin_centres_frac,
    ) = _prepare_minute_grouped_empirical_win_pct(data_dir, num_time_bins)

    x_list, y_list, w_list = [], [], []
    for p in probs:
        for tb in time_labels:
            n = int(count_matrix.loc[p, tb])
            if n < 1:
                continue
            v = win_raw.loc[p, tb]
            if pd.isna(v):
                continue
            x_list.append([time_bin_to_frac[tb], p / 100.0])
            y_list.append(float(v))
            w_list.append(float(n))

    x_train = np.asarray(x_list, dtype=float)
    y_train = np.asarray(y_list, dtype=float)
    w_train = np.asarray(w_list, dtype=float)

    gam = LinearGAM(
        s(0, n_splines=N_SPLINES_TIME_WIN, spline_order=3)
        + s(1, n_splines=N_SPLINES_PROB_WIN, spline_order=3)
        + te(0, 1, n_splines=[8, 8])
    )
    gam.gridsearch(x_train, y_train, weights=w_train, lam=np.logspace(-3, 3, 11), progress=False)

    y_pred = gam.predict(x_train)
    resid = y_pred - y_train
    rmse = float(np.sqrt(np.sum(w_train * resid**2) / np.sum(w_train)))
    return rmse


def _fit_edge_gam_and_mase(data_dir: str, num_time_bins: int) -> float:
    (
        _df,
        signed_raw,
        count_matrix,
        time_labels,
        probs,
        _time_edges,
        _num_time_bins,
        time_bin_to_frac,
        _bin_centres_frac,
    ) = _prepare_minute_grouped_signed_edge(data_dir, num_time_bins)

    x_list, y_list, w_list = [], [], []
    for p in probs:
        for tb in time_labels:
            n = int(count_matrix.loc[p, tb])
            if n < 1:
                continue
            v = signed_raw.loc[p, tb]
            if pd.isna(v):
                continue
            x_list.append([time_bin_to_frac[tb], p / 100.0])
            y_list.append(float(v))
            w_list.append(float(n))

    x_train = np.asarray(x_list, dtype=float)
    y_train = np.asarray(y_list, dtype=float)
    w_train = np.asarray(w_list, dtype=float)

    gam = LinearGAM(
        s(0, n_splines=N_SPLINES_TIME_EDGE, spline_order=3)
        + s(1, n_splines=N_SPLINES_PROB_EDGE, spline_order=3)
        + te(0, 1, n_splines=[8, 8])
    )
    gam.gridsearch(x_train, y_train, weights=w_train, lam=np.logspace(-3, 3, 11), progress=False)

    y_pred = gam.predict(x_train)
    mase = float(np.sum(w_train * np.abs(y_pred)) / np.sum(w_train))
    return mase


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, "../../0-Data"))

    print("Data directory:", data_dir)
    mace_week = compute_mace_week(data_dir)
    print(f"MACE-week (mean of weekly mean per-game |quote − outcome|, pp): {mace_week:.4f}")

    rmse_smooth = _fit_win_gam_and_rmse(data_dir, NUM_TIME_BINS_DEFAULT)
    print(f"RMSE-smooth (weighted RMSE raw vs GAM win %%, pp): {rmse_smooth:.4f}")

    mase = _fit_edge_gam_and_mase(data_dir, NUM_TIME_BINS_DEFAULT)
    print(f"MASE (weighted mean |GAM smoothed signed edge|, pp): {mase:.4f}")

    out_path = os.path.join(script_dir, "gam_metrics.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "GAM evaluation metrics (see compute_gam_metrics.py)\n"
            f"MACE-week: {mace_week:.6f}\n"
            f"RMSE-smooth: {rmse_smooth:.6f}\n"
            f"MASE: {mase:.6f}\n"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
