#!/usr/bin/env python3
"""
model.py

Probability calibration model for Kalshi college basketball markets.

Trains a Generalized Additive Model (GAM) that learns a smooth mapping from
(game_elapsed_seconds, Kalshi_win_probability) → empirical win probability.

Pipeline
--------
1. Load merged game data
2. Normalize features, build game-level train/test split (80/20)
3. Fit a LogisticGAM with:
      - spline on game time
      - spline on Kalshi probability
      - tensor interaction between the two
4. Evaluate against baseline (raw Kalshi prob) via:
      - Brier score
      - Log loss
      - Calibration curves
5. Produce two key visualizations:
      a. Calibrated probability surface
      b. Miscalibration map (model − Kalshi)

Usage
-----
    python model.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import seaborn as sns
from pygam import LogisticGAM, s, te

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE = "GeneratedDataFiles/all_games_merged_clean_GOOD.csv"
OUTPUT_DIR = "GeneratedDataFiles"
TEST_FRACTION = 0.20       # hold-out fraction (game-level split)
RANDOM_SEED = 42
N_SPLINES_TIME = 20        # number of spline basis functions for game time
N_SPLINES_PROB = 20        # number of spline basis functions for Kalshi prob
GRID_RESOLUTION = 100      # resolution for probability surface plots


# ── 1. Load & Prepare Data ───────────────────────────────────────────────────
def load_data(path):
    """Load the merged CSV and derive normalised features."""
    print(f"\n{'='*70}")
    print("  PROBABILITY CALIBRATION MODEL")
    print(f"{'='*70}")

    df = pd.read_csv(path)
    print(f"\n  Loaded {len(df):,} observations from {path}")
    print(f"  Unique games (team-sides): {df['kalshi_event'].nunique():,}")

    # Unique game identifiers (kalshi_event + team) — used for game-level split
    df["game_id"] = df["kalshi_event"] + "_" + df["team"]

    # Feature: Kalshi probability as fraction [0, 1]
    df["kalshi_prob"] = df["win_prob_pct"] / 100.0

    # Feature: game time as fraction [0, 1]  (0 = tip-off, 1 = end of regulation)
    df["time_frac"] = df["game_elapsed_seconds"] / 2400.0

    return df


# ── 2. Train / Test Split (by game) ──────────────────────────────────────────
def split_by_game(df, test_frac=TEST_FRACTION, seed=RANDOM_SEED):
    """Split into train/test sets at the game level so no data leaks."""
    rng = np.random.RandomState(seed)
    game_ids = df["game_id"].unique()
    rng.shuffle(game_ids)

    n_test = int(len(game_ids) * test_frac)
    test_ids = set(game_ids[:n_test])
    train_ids = set(game_ids[n_test:])

    train = df[df["game_id"].isin(train_ids)].copy()
    test  = df[df["game_id"].isin(test_ids)].copy()

    print(f"\n  Train / Test split (by game, {100*(1-test_frac):.0f}/{100*test_frac:.0f}):")
    print(f"    Train: {len(train):>9,} obs  ({len(train_ids):,} game-sides)")
    print(f"    Test:  {len(test):>9,} obs  ({len(test_ids):,} game-sides)")

    return train, test


# ── 3. Fit the GAM ───────────────────────────────────────────────────────────
def fit_gam(train):
    """
    Fit a LogisticGAM with:
        s(time_frac)  +  s(kalshi_prob)  +  te(time_frac, kalshi_prob)

    The tensor product `te` captures the interaction so that Kalshi's
    miscalibration can vary across game time.
    """
    X_train = train[["time_frac", "kalshi_prob"]].values
    y_train = train["team_won"].values

    print("\n  Fitting LogisticGAM ...")
    print(f"    Splines for time:        {N_SPLINES_TIME}")
    print(f"    Splines for probability: {N_SPLINES_PROB}")

    gam = LogisticGAM(
        s(0, n_splines=N_SPLINES_TIME,  spline_order=3) +
        s(1, n_splines=N_SPLINES_PROB,  spline_order=3) +
        te(0, 1, n_splines=[8, 8])
    )
    gam.gridsearch(
        X_train, y_train,
        lam=np.logspace(-3, 3, 11),
        progress=False,
    )
    print(f"    Best λ (smoothing): {gam.lam}")
    print(f"    Pseudo-R²:          {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")

    return gam


# ── 4. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(gam, test):
    """
    Compare the GAM against the raw Kalshi baseline on held-out data.

    Metrics:
        - Brier score  (lower is better)
        - Log loss     (lower is better)
        - Calibration curves
    """
    X_test = test[["time_frac", "kalshi_prob"]].values
    y_test = test["team_won"].values

    # --- predictions ---
    p_kalshi = test["kalshi_prob"].values                      # baseline
    p_model  = gam.predict_proba(X_test)                       # calibrated

    # Clip to avoid log(0)
    eps = 1e-8
    p_kalshi_clip = np.clip(p_kalshi, eps, 1 - eps)
    p_model_clip  = np.clip(p_model,  eps, 1 - eps)

    # --- scalar metrics ---
    brier_baseline = brier_score_loss(y_test, p_kalshi)
    brier_model    = brier_score_loss(y_test, p_model_clip)
    ll_baseline    = log_loss(y_test, p_kalshi_clip)
    ll_model       = log_loss(y_test, p_model_clip)

    print(f"\n  {'─'*50}")
    print(f"  Evaluation on held-out test set ({len(y_test):,} obs)")
    print(f"  {'─'*50}")
    print(f"  {'Metric':<20} {'Baseline (Kalshi)':>18} {'GAM Model':>12} {'Δ':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Brier score':<20} {brier_baseline:>18.6f} {brier_model:>12.6f} {brier_model - brier_baseline:>+10.6f}")
    print(f"  {'Log loss':<20} {ll_baseline:>18.6f} {ll_model:>12.6f} {ll_model - ll_baseline:>+10.6f}")
    pct_brier = (brier_baseline - brier_model) / brier_baseline * 100
    pct_ll    = (ll_baseline - ll_model)       / ll_baseline    * 100
    print(f"\n  Brier improvement:  {pct_brier:+.2f}%")
    print(f"  Log-loss improvement: {pct_ll:+.2f}%")

    metrics = dict(
        brier_baseline=brier_baseline, brier_model=brier_model,
        ll_baseline=ll_baseline, ll_model=ll_model,
        y_test=y_test, p_kalshi=p_kalshi, p_model=p_model_clip,
    )
    return metrics


# ── 5. Calibration Curve Plot ─────────────────────────────────────────────────
def plot_calibration_curves(metrics, out_dir=OUTPUT_DIR):
    """Reliability diagram comparing Kalshi baseline to GAM."""
    y = metrics["y_test"]
    p_b = metrics["p_kalshi"]
    p_m = metrics["p_model"]

    n_bins = 20
    frac_b, mean_b = calibration_curve(y, p_b, n_bins=n_bins, strategy="uniform")
    frac_m, mean_m = calibration_curve(y, p_m, n_bins=n_bins, strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: calibration curves ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_b, frac_b, "o-", color="#E74C3C", lw=2, ms=6, label=f"Kalshi baseline  (Brier={metrics['brier_baseline']:.4f})")
    ax.plot(mean_m, frac_m, "s-", color="#2E86C1", lw=2, ms=6, label=f"GAM calibrated   (Brier={metrics['brier_model']:.4f})")
    ax.set_xlabel("Predicted probability", fontsize=13)
    ax.set_ylabel("Observed win rate", fontsize=13)
    ax.set_title("Calibration Curves (Reliability Diagram)", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # --- Right: histogram of predicted probabilities ---
    ax2 = axes[1]
    ax2.hist(p_b, bins=50, alpha=0.5, color="#E74C3C", label="Kalshi baseline", density=True)
    ax2.hist(p_m, bins=50, alpha=0.5, color="#2E86C1", label="GAM calibrated", density=True)
    ax2.set_xlabel("Predicted probability", fontsize=13)
    ax2.set_ylabel("Density", fontsize=13)
    ax2.set_title("Distribution of Predicted Probabilities", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_curves.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved calibration curves  → {path}")


# ── 6. Calibrated Probability Surface ─────────────────────────────────────────
def plot_probability_surface(gam, out_dir=OUTPUT_DIR, resolution=GRID_RESOLUTION):
    """
    3-panel figure:
        Left:   Raw Kalshi probability (identity surface for reference)
        Centre: GAM-calibrated probability surface
        Right:  Difference (calibrated − Kalshi)
    """
    # Build evaluation grid
    t = np.linspace(0, 1, resolution)
    p = np.linspace(0.01, 0.99, resolution)
    T, P = np.meshgrid(t, p)
    grid = np.column_stack([T.ravel(), P.ravel()])

    # GAM predictions on the grid
    Z_model = gam.predict_proba(grid).reshape(resolution, resolution)
    Z_kalshi = P  # identity
    Z_diff = Z_model - Z_kalshi

    # Convert axes to human-readable units
    time_labels = np.linspace(0, 40, 9)   # minutes
    prob_labels = np.linspace(0, 100, 11) # %

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # --- Panel 1: Kalshi raw (reference) ---
    im0 = axes[0].imshow(
        Z_kalshi, origin="lower", aspect="auto",
        extent=[0, 40, 1, 99], cmap="RdYlGn", vmin=0, vmax=1,
    )
    axes[0].set_title("Raw Kalshi Probability\n(identity reference)", fontsize=14)
    axes[0].set_xlabel("Game Time (minutes)", fontsize=12)
    axes[0].set_ylabel("Kalshi Win Probability (%)", fontsize=12)
    plt.colorbar(im0, ax=axes[0], label="Win Probability", shrink=0.8)

    # --- Panel 2: GAM calibrated surface ---
    im1 = axes[1].imshow(
        Z_model, origin="lower", aspect="auto",
        extent=[0, 40, 1, 99], cmap="RdYlGn", vmin=0, vmax=1,
    )
    axes[1].set_title("GAM-Calibrated Probability Surface", fontsize=14)
    axes[1].set_xlabel("Game Time (minutes)", fontsize=12)
    axes[1].set_ylabel("Kalshi Win Probability (%)", fontsize=12)
    plt.colorbar(im1, ax=axes[1], label="Calibrated Win Probability", shrink=0.8)

    # --- Panel 3: Miscalibration (difference) ---
    max_abs = max(0.15, np.abs(Z_diff).max())  # symmetric color scale
    im2 = axes[2].imshow(
        Z_diff, origin="lower", aspect="auto",
        extent=[0, 40, 1, 99], cmap="RdBu_r", vmin=-max_abs, vmax=max_abs,
    )
    axes[2].set_title("Calibration Correction\n(GAM − Kalshi)", fontsize=14)
    axes[2].set_xlabel("Game Time (minutes)", fontsize=12)
    axes[2].set_ylabel("Kalshi Win Probability (%)", fontsize=12)
    cb = plt.colorbar(im2, ax=axes[2], label="Correction (+ = Kalshi understates)", shrink=0.8)

    plt.suptitle(
        "Probability Calibration Surface — Kalshi College Basketball Markets",
        fontsize=16, y=1.02, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_surface.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved probability surface → {path}")

    return Z_model, Z_kalshi, Z_diff


# ── 6b. 3D Surface Plot ──────────────────────────────────────────────────────────
def plot_3d_surface(gam, out_dir=OUTPUT_DIR, resolution=GRID_RESOLUTION):
    """
    Create a 3D surface plot of the calibrated probability surface.
    
    X-axis: Game time (minutes)
    Y-axis: Kalshi win probability (%)
    Z-axis: GAM-calibrated win probability
    """
    # Build evaluation grid
    t = np.linspace(0, 1, resolution)
    p = np.linspace(0.01, 0.99, resolution)
    T, P = np.meshgrid(t, p)
    grid = np.column_stack([T.ravel(), P.ravel()])

    # GAM predictions on the grid
    Z_model = gam.predict_proba(grid).reshape(resolution, resolution)

    # Convert to human-readable units for axes
    T_minutes = T * 40  # 0-40 minutes
    P_percent = P * 100  # 1-99%

    # Scale z-axis by multiplier to make the curve more visible
    Z_SCALE = 10  # Multiplier for z-axis height (increased from 5)
    Z_model_scaled = Z_model * Z_SCALE

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot with color mapping
    # Scale z-axis for visibility, but color based on original probability values (0-1)
    # Normalize Z_model to 0-1 range for colormap
    norm = plt.Normalize(vmin=0, vmax=1)
    colors = plt.cm.RdYlGn(norm(Z_model))
    
    surf = ax.plot_surface(
        T_minutes, P_percent, Z_model_scaled,
        facecolors=colors,  # Color based on original 0-1 scale
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    # Set labels and title
    ax.set_xlabel('Game Time (minutes)', fontsize=13, labelpad=10)
    ax.set_ylabel('Kalshi Win Probability (%)', fontsize=13, labelpad=10)
    ax.set_zlabel(f'Calibrated Win Probability (×{Z_SCALE})', fontsize=13, labelpad=10)
    ax.set_title(
        '3D Calibrated Probability Surface — GAM Model\n'
        'Smooth mapping from (Game Time, Kalshi Prob) → Empirical Win Probability',
        fontsize=14, fontweight='bold', pad=20
    )

    # Set axis limits
    ax.set_xlim(0, 40)
    ax.set_ylim(1, 99)
    ax.set_zlim(0, Z_SCALE)

    # Add colorbar (using original scale 0-1, not scaled)
    # Create a mappable for the colorbar using the original Z_model values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1, label='Win Probability')

    # Adjust viewing angle for better visualization
    # Rotated 45 degrees to the right from previous position (260 + 45 = 305)
    ax.view_init(elev=35, azim=305)

    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_surface_3d.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3D surface plot    → {path}")


# ── 7. Detailed Miscalibration Heat Map ───────────────────────────────────────
def plot_miscalibration_map(gam, df_test, out_dir=OUTPUT_DIR):
    """
    Empirical miscalibration map:
        Bins the test data by (time, Kalshi prob) and shows
        (empirical win rate − Kalshi prob) in each cell.

    Red  = Kalshi overstates true win probability (too high)
    Blue = Kalshi understates true win probability (too low)

    Overlays the GAM's correction contours for comparison.
    """
    # Bin settings
    n_time_bins = 16
    n_prob_bins = 20
    min_obs = 30

    time_edges = np.linspace(0, 2400, n_time_bins + 1)
    prob_edges = np.linspace(0, 100, n_prob_bins + 1)

    df = df_test.copy()
    df["time_bin"] = pd.cut(df["game_elapsed_seconds"], bins=time_edges, right=False, include_lowest=True)
    df["prob_bin"] = pd.cut(df["win_prob_pct"], bins=prob_edges, right=False, include_lowest=True)

    grouped = df.groupby(["prob_bin", "time_bin"], observed=False).agg(
        empirical_wr=("team_won", "mean"),
        kalshi_mean=("win_prob_pct", "mean"),
        n=("team_won", "count"),
    ).reset_index()

    # Miscalibration = empirical win rate − (kalshi mean / 100)
    grouped["miscal"] = grouped["empirical_wr"] - (grouped["kalshi_mean"] / 100.0)
    grouped.loc[grouped["n"] < min_obs, "miscal"] = np.nan

    # Pivot
    time_labels = [f"{int(lo/60)}-{int(hi/60)}m" for lo, hi in zip(time_edges[:-1], time_edges[1:])]
    prob_labels = [f"{int(lo)}-{int(hi)}%" for lo, hi in zip(prob_edges[:-1], prob_edges[1:])]

    miscal_matrix = np.full((n_prob_bins, n_time_bins), np.nan)
    count_matrix  = np.full((n_prob_bins, n_time_bins), 0.0)

    for _, row in grouped.iterrows():
        pi = prob_edges.searchsorted(row["kalshi_mean"], side="right") - 1 if not pd.isna(row["kalshi_mean"]) else -1
        # Use the bin indices from the categorical
        prob_idx = grouped["prob_bin"].cat.categories.get_loc(row["prob_bin"]) if row["prob_bin"] in grouped["prob_bin"].cat.categories else -1
        time_idx = grouped["time_bin"].cat.categories.get_loc(row["time_bin"]) if row["time_bin"] in grouped["time_bin"].cat.categories else -1
        if 0 <= prob_idx < n_prob_bins and 0 <= time_idx < n_time_bins:
            miscal_matrix[prob_idx, time_idx] = row["miscal"]
            count_matrix[prob_idx, time_idx]  = row["n"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(18, 10))

    max_abs = 0.20
    mask = np.isnan(miscal_matrix)

    im = ax.imshow(
        miscal_matrix, origin="lower", aspect="auto",
        extent=[0, n_time_bins, 0, n_prob_bins],
        cmap="RdBu_r", vmin=-max_abs, vmax=max_abs,
        interpolation="nearest",
    )

    # Overlay the GAM correction contours
    t_grid = np.linspace(0, 1, n_time_bins)
    p_grid = np.linspace(0.01, 0.99, n_prob_bins)
    TG, PG = np.meshgrid(t_grid, p_grid)
    grid_pts = np.column_stack([TG.ravel(), PG.ravel()])
    Z_gam = gam.predict_proba(grid_pts).reshape(n_prob_bins, n_time_bins) - PG

    contour_levels = np.array([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
    cs = ax.contour(
        np.arange(n_time_bins) + 0.5,
        np.arange(n_prob_bins) + 0.5,
        Z_gam,
        levels=contour_levels,
        colors="k", linewidths=1.0, linestyles="--", alpha=0.7,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%+.2f")

    # Annotate cells with enough data
    for pi in range(n_prob_bins):
        for ti in range(n_time_bins):
            if not np.isnan(miscal_matrix[pi, ti]) and count_matrix[pi, ti] >= min_obs:
                val = miscal_matrix[pi, ti]
                color = "white" if abs(val) > 0.12 else "black"
                ax.text(
                    ti + 0.5, pi + 0.5,
                    f"{val:+.0%}",
                    ha="center", va="center", fontsize=6.5,
                    fontweight="bold", color=color,
                )

    # Grey out sparse cells
    for pi in range(n_prob_bins):
        for ti in range(n_time_bins):
            if np.isnan(miscal_matrix[pi, ti]):
                ax.add_patch(plt.Rectangle(
                    (ti, pi), 1, 1, fill=True,
                    facecolor="#d9d9d9", edgecolor="white", linewidth=0.3,
                ))

    # Axes
    ax.set_xticks(np.arange(n_time_bins) + 0.5)
    ax.set_xticklabels(time_labels, fontsize=9)
    ax.set_yticks(np.arange(n_prob_bins) + 0.5)
    ax.set_yticklabels(prob_labels, fontsize=9)
    ax.set_xlabel("Game Time", fontsize=13, labelpad=10)
    ax.set_ylabel("Kalshi Win Probability", fontsize=13, labelpad=10)

    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Empirical Win Rate − Kalshi Probability", fontsize=11)

    ax.set_title(
        "Miscalibration Map — Where Kalshi Over/Understates Win Probability\n"
        "(Red = Kalshi too high · Blue = Kalshi too low · Dashed contours = GAM correction)",
        fontsize=14, pad=14,
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "miscalibration_map.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved miscalibration map  → {path}")


# ── 8. Time-Sliced Calibration Detail ────────────────────────────────────────
def plot_time_sliced_calibration(gam, test, out_dir=OUTPUT_DIR):
    """
    Show calibration curves at different game time slices
    to reveal how miscalibration evolves during a game.
    """
    time_slices = [
        ("0-5 min (early)",    0,   300),
        ("5-15 min (1st half)", 300, 900),
        ("15-20 min (halftime area)", 900, 1200),
        ("20-30 min (2nd half early)", 1200, 1800),
        ("30-40 min (late game)",  1800, 2400),
    ]

    fig, axes = plt.subplots(1, len(time_slices), figsize=(28, 6), sharey=True)

    for ax, (label, t_lo, t_hi) in zip(axes, time_slices):
        mask = (test["game_elapsed_seconds"] >= t_lo) & (test["game_elapsed_seconds"] < t_hi)
        sub = test[mask]
        if len(sub) < 100:
            ax.set_title(f"{label}\n(n={len(sub)}, too few)", fontsize=10)
            continue

        y = sub["team_won"].values
        p_kalshi = sub["kalshi_prob"].values
        X_sub = sub[["time_frac", "kalshi_prob"]].values
        p_model = np.clip(gam.predict_proba(X_sub), 1e-8, 1 - 1e-8)

        n_bins = 15
        frac_b, mean_b = calibration_curve(y, p_kalshi, n_bins=n_bins, strategy="uniform")
        frac_m, mean_m = calibration_curve(y, p_model,  n_bins=n_bins, strategy="uniform")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.plot(mean_b, frac_b, "o-", color="#E74C3C", lw=1.5, ms=4, label="Kalshi")
        ax.plot(mean_m, frac_m, "s-", color="#2E86C1", lw=1.5, ms=4, label="GAM")
        ax.set_title(f"{label}\n(n={len(sub):,})", fontsize=10)
        ax.set_xlabel("Predicted prob", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Observed win rate", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Calibration by Game Phase — Kalshi vs. GAM Model",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_by_game_phase.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved time-sliced curves  → {path}")


# ── 8b. Calibration Heatmap (discrete boxes, like regular heatmap) ────────────
def plot_full_resolution_heatmap(gam, out_dir=OUTPUT_DIR):
    """
    Generate a calibration heatmap of the GAM-calibrated probability surface
    using discrete boxes (matching the style of the regular empirical heatmap).

    Time buckets: 2-minute bins (±1 min game-clock accuracy → 20 bins).
    Probability:  1 row per integer percentage (1 %–99 %).

    Grid: 99 prob rows  ×  20 time bins.
    Also exports the underlying matrix as a CSV.
    """
    print("\n  Building calibration heatmap (discrete boxes) …")

    # ── configuration ────────────────────────────────────────────────────
    NUM_TIME_BINS = 20          # 2-minute bins (40 min / 20 = 2 min each)
    BIN_SECONDS   = 2400 / NUM_TIME_BINS  # 120 s per bin
    MIN_OBS       = 0           # GAM predictions always exist, no masking needed

    # ── time bin edges and labels ────────────────────────────────────────
    TIME_EDGES = np.linspace(0, 2400, NUM_TIME_BINS + 1)
    TIME_LABELS = [f"{int(lo/60)}-{int(hi/60)} min"
                   for lo, hi in zip(TIME_EDGES[:-1], TIME_EDGES[1:])]

    # ── probability rows ─────────────────────────────────────────────────
    probs = np.arange(1, 100)   # 1 % … 99 %

    # ── evaluate GAM at the centre of each (time_bin, prob) cell ─────────
    bin_centres_sec = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2.0
    bin_centres_frac = bin_centres_sec / 2400.0
    prob_frac = probs / 100.0

    T, P = np.meshgrid(bin_centres_frac, prob_frac)
    grid = np.column_stack([T.ravel(), P.ravel()])
    Z = gam.predict_proba(grid).reshape(len(probs), NUM_TIME_BINS)

    # ── build pandas matrices for seaborn ────────────────────────────────
    win_rate_matrix = pd.DataFrame(Z, index=probs, columns=TIME_LABELS)

    # ── save raw matrix as CSV ───────────────────────────────────────────
    csv_path = os.path.join(out_dir, "calibration_heatmap_data.csv")
    df_out = win_rate_matrix.copy()
    df_out.index.name = "kalshi_prob_pct"
    df_out.columns.name = "time_bin"
    df_out.to_csv(csv_path)
    print(f"    CSV ({len(probs)} × {NUM_TIME_BINS}) → {csv_path}")

    # ── annotation matrix (show "XX%" in each cell) ──────────────────────
    annot_matrix = win_rate_matrix.copy().astype(object)
    for r in win_rate_matrix.index:
        for c in win_rate_matrix.columns:
            val = win_rate_matrix.loc[r, c]
            if pd.isna(val):
                annot_matrix.loc[r, c] = ""
            else:
                annot_matrix.loc[r, c] = f"{val*100:.0f}%"

    # ── flip so high probability is on top ───────────────────────────────
    data_plot  = win_rate_matrix.iloc[::-1]
    annot_plot = annot_matrix.iloc[::-1]

    # ── plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 28))

    heatmap = sns.heatmap(
        data_plot,
        annot=annot_plot,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "GAM-Calibrated Win Probability", "shrink": 0.6, "pad": 0.02},
        ax=ax,
        xticklabels=True,
        annot_kws={"fontsize": 5.5, "fontweight": "bold", "color": "white"},
    )
    
    # Explicitly set all annotation text to white
    for text in heatmap.texts:
        text.set_color("white")
    
    # Set colorbar tick labels and label to white
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(colors="white")
    cbar.set_label("GAM-Calibrated Win Probability", color="white")

    # ── y-axis: show every 5th % ─────────────────────────────────────────
    flipped_index = list(data_plot.index)
    ytick_positions = []
    ytick_labels = []
    for i, prob in enumerate(flipped_index):
        if prob % 5 == 0:
            ytick_positions.append(i + 0.5)   # centre of cell
            ytick_labels.append(f"{prob}%")

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=10, color="white")

    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12, color="white")
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12, color="white")
    ax.set_title(
        "GAM-Calibrated Probability Heatmap — Discrete Boxes\n"
        f"(99 prob rows × {NUM_TIME_BINS} time bins · {int(BIN_SECONDS/60)}-min buckets · ±1 min clock accuracy)",
        fontsize=15,
        pad=16,
        color="white",
    )

    ax.tick_params(axis="x", labelsize=11, rotation=0, colors="white")
    ax.tick_params(axis="y", labelsize=10, colors="white")

    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Heatmap image           → {path}")


# ── 8c. Zoomed High-Probability Calibration Heatmap ──────────────────────────
def plot_zoomed_high_prob_heatmap(gam, out_dir=OUTPUT_DIR):
    """
    Generate a zoomed-in, elongated heatmap focused on the 80-100% probability
    range, showing every second of game time (0-2400s) and every percentage
    point from 80% to 100%.

    Grid: 21 rows (80-100%) × 2,401 columns (every second).
    """
    print("\n  Building zoomed high-probability calibration heatmap (80-100%) …")

    # ── evaluation grid ──────────────────────────────────────────────────
    seconds = np.arange(0, 2401)              # 0 … 2400 s
    probs   = np.arange(80, 101)              # 80 % … 100 %

    time_frac = seconds / 2400.0              # normalise to [0, 1]
    prob_frac = probs   / 100.0               # normalise to [0, 1]

    T, P = np.meshgrid(time_frac, prob_frac)
    grid = np.column_stack([T.ravel(), P.ravel()])

    Z = gam.predict_proba(grid).reshape(len(probs), len(seconds))

    # ── save raw matrix as CSV ───────────────────────────────────────────
    csv_path = os.path.join(out_dir, "calibration_heatmap_80_100_data.csv")
    df_out = pd.DataFrame(Z, index=probs, columns=seconds)
    df_out.index.name = "kalshi_prob_pct"
    df_out.columns.name = "game_elapsed_seconds"
    df_out.to_csv(csv_path)
    print(f"    CSV ({len(probs)} × {len(seconds)}) → {csv_path}")

    # ── plot (elongated figure) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(28, 10))  # wider, shorter for elongation

    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[0, 40, 80, 100],              # x in minutes, y in %
        cmap="RdYlGn", vmin=0, vmax=1,
        interpolation="nearest",
    )

    # ── contour lines at finer increments for high probabilities ────────
    T_min = T * 40                            # time in minutes for contour
    P_pct = P * 100                           # prob in % for contour
    # Contours from 0.8 to 1.0 in 0.02 increments
    contour_levels = np.arange(0.80, 1.01, 0.02)
    cs = ax.contour(
        T_min, P_pct, Z,
        levels=contour_levels,
        colors="black", linewidths=0.7, alpha=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    # ── x-axis: every 5 minutes with minor ticks at 1 minute ────────────
    major_minutes = np.arange(0, 41, 5)
    minor_minutes = np.arange(0, 41, 1)
    ax.set_xticks(major_minutes)
    ax.set_xticklabels([f"{int(m)}" for m in major_minutes], fontsize=12)
    ax.set_xticks(minor_minutes, minor=True)
    ax.set_xlabel("Game Time (minutes)", fontsize=15, labelpad=12)

    # ── y-axis: every 1 % with minor ticks ───────────────────────────────
    ax.set_yticks(np.arange(80, 101, 1))
    ax.set_yticklabels([f"{p}%" for p in range(80, 101, 1)], fontsize=11)
    ax.set_yticks(np.arange(80, 101, 0.5), minor=True)
    ax.set_ylabel("Kalshi Win Probability (%)", fontsize=15, labelpad=12)
    ax.set_ylim(80, 100)

    ax.tick_params(which="minor", length=2, color="#999999")

    # ── colour bar ───────────────────────────────────────────────────────
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("GAM-Calibrated Win Probability", fontsize=13)

    # ── title ────────────────────────────────────────────────────────────
    ax.set_title(
        "GAM-Calibrated Probability Heatmap — High Probability Range (80-100%)\n"
        "(21 probability rows  ×  2,401 time columns · Every second × Every percentage point)",
        fontsize=17, fontweight="bold", pad=16,
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_heatmap_80_100.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"    Zoomed heatmap image    → {path}")


# ── 9. Summary Statistics ─────────────────────────────────────────────────────
def print_summary(gam, metrics, Z_diff):
    """Print a concise summary of findings."""
    print(f"\n{'='*70}")
    print("  SUMMARY OF FINDINGS")
    print(f"{'='*70}")
    print(f"""
  Model:  LogisticGAM with spline basis functions
          s(game_time) + s(kalshi_prob) + te(game_time, kalshi_prob)

  Baseline (raw Kalshi):
      Brier score:  {metrics['brier_baseline']:.6f}
      Log loss:     {metrics['ll_baseline']:.6f}

  GAM Calibrated:
      Brier score:  {metrics['brier_model']:.6f}   (Δ = {metrics['brier_model'] - metrics['brier_baseline']:+.6f})
      Log loss:     {metrics['ll_model']:.6f}   (Δ = {metrics['ll_model'] - metrics['ll_baseline']:+.6f})

  Miscalibration surface:
      Max overstatement (Kalshi too high):  {Z_diff.min():+.3f}  ({Z_diff.min()*100:+.1f} pp)
      Max understatement (Kalshi too low):  {Z_diff.max():+.3f}  ({Z_diff.max()*100:+.1f} pp)
      Mean absolute correction:             {np.nanmean(np.abs(Z_diff)):.3f}  ({np.nanmean(np.abs(Z_diff))*100:.1f} pp)
""")

    # Identify biggest miscalibration regions
    resolution = Z_diff.shape[0]
    t_vals = np.linspace(0, 40, resolution)
    p_vals = np.linspace(1, 99, resolution)

    # Flatten and find extremes
    flat_idx = np.unravel_index(np.argmin(Z_diff), Z_diff.shape)
    print(f"  Worst overstatement region:  ~{p_vals[flat_idx[0]]:.0f}% Kalshi prob at ~{t_vals[flat_idx[1]]:.0f} min")
    flat_idx = np.unravel_index(np.argmax(Z_diff), Z_diff.shape)
    print(f"  Worst understatement region: ~{p_vals[flat_idx[0]]:.0f}% Kalshi prob at ~{t_vals[flat_idx[1]]:.0f} min")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_model(input_file=INPUT_FILE):
    """Run the full calibration pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    df = load_data(input_file)

    # 2. Split
    train, test = split_by_game(df)
    
    # Save test game_ids for backtesting (to avoid data leakage)
    test_game_ids = test["game_id"].unique()
    test_ids_path = os.path.join(OUTPUT_DIR, "test_game_ids.txt")
    with open(test_ids_path, "w") as f:
        for gid in sorted(test_game_ids):
            f.write(f"{gid}\n")
    print(f"\n  Saved test set game IDs → {test_ids_path} ({len(test_game_ids):,} games)")

    # 3. Fit GAM
    gam = fit_gam(train)

    # 4. Evaluate
    metrics = evaluate(gam, test)

    # 5. Calibration curves
    plot_calibration_curves(metrics)

    # 6. Probability surface (+ miscalibration panel)
    Z_model, Z_kalshi, Z_diff = plot_probability_surface(gam)

    # 6b. 3D surface plot
    plot_3d_surface(gam)

    # 7. Detailed miscalibration map
    plot_miscalibration_map(gam, test)

    # 8. Time-sliced calibration
    plot_time_sliced_calibration(gam, test)

    # 8b. Full-resolution calibration heatmap (every second × every %)
    plot_full_resolution_heatmap(gam)

    # 8c. Zoomed high-probability heatmap (80-100%)
    plot_zoomed_high_prob_heatmap(gam)

    # 9. Summary
    print_summary(gam, metrics, Z_diff)

    print(f"  All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}\n")

    return gam, metrics


if __name__ == "__main__":
    run_model()
