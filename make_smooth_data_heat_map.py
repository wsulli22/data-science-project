#!/usr/bin/env python3
"""
make_smooth_data_heat_map.py

Reads all_games_merged_clean_GOOD.csv and produces a smoothed calibration
heat map using a GAM (Generalized Additive Model) to smooth the data.

The smoothing is done using a LogisticGAM with splines on game time and
Kalshi probability, similar to the approach in model.py.

Outputs:
    - smoothed_heatmap.png: Visual heatmap of smoothed probabilities
    - smoothed_heatmap_data.csv: CSV file with the smoothed probability matrix
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LogisticGAM, s, te

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE = "GeneratedDataFiles/all_games_merged_clean_GOOD.csv"
OUTPUT_DIR = "GeneratedDataFiles"  # For CSV data files
VISUALIZATION_DIR = "GeneratedVisualizations"  # For PNG visualization files
NUM_TIME_BINS = 20          # Number of time bins (2-minute bins for 40 min game)
N_SPLINES_TIME = 20         # Number of spline basis functions for game time
N_SPLINES_PROB = 20         # Number of spline basis functions for Kalshi prob
MIN_OBS = 0                 # GAM predictions always exist, no masking needed


def load_and_prepare_data(path):
    """Load the merged CSV and prepare features for GAM."""
    print(f"\n{'='*70}")
    print("  SMOOTHED DATA HEATMAP GENERATOR")
    print(f"{'='*70}")
    
    df = pd.read_csv(path)
    print(f"\n  Loaded {len(df):,} observations from {path}")
    print(f"  Unique games: {df['kalshi_event'].nunique():,}")
    
    # Create unique game identifier for potential game-level splitting
    df["game_id"] = df["kalshi_event"] + "_" + df["team"]
    
    # Feature: Kalshi probability as fraction [0, 1]
    df["kalshi_prob"] = df["win_prob_pct"] / 100.0
    
    # Feature: game time as fraction [0, 1] (0 = tip-off, 1 = end of regulation)
    df["time_frac"] = df["game_elapsed_seconds"] / 2400.0
    
    # Filter out invalid data
    df = df.dropna(subset=["win_prob_pct", "game_elapsed_seconds", "team_won"])
    df = df[df["win_prob_pct"].between(1, 99)]  # Keep probabilities in 1-99% range
    df = df[df["game_elapsed_seconds"].between(0, 2400)]  # Keep within regulation time
    
    print(f"  After filtering: {len(df):,} observations")
    
    return df


def fit_gam_model(df):
    """
    Fit a LogisticGAM with:
        s(time_frac)  +  s(kalshi_prob)  +  te(time_frac, kalshi_prob)
    
    The tensor product `te` captures the interaction so that calibration
    can vary across game time.
    """
    X = df[["time_frac", "kalshi_prob"]].values
    y = df["team_won"].values
    
    print("\n  Fitting LogisticGAM for smoothing...")
    print(f"    Splines for time:        {N_SPLINES_TIME}")
    print(f"    Splines for probability: {N_SPLINES_PROB}")
    
    gam = LogisticGAM(
        s(0, n_splines=N_SPLINES_TIME, spline_order=3) +
        s(1, n_splines=N_SPLINES_PROB, spline_order=3) +
        te(0, 1, n_splines=[8, 8])
    )
    gam.gridsearch(
        X, y,
        lam=np.logspace(-3, 3, 11),
        progress=False,
    )
    print(f"    Best λ (smoothing): {gam.lam}")
    print(f"    Pseudo-R²:          {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")
    
    return gam


def plot_heatmap_from_matrix(win_rate_matrix, out_dir=OUTPUT_DIR, vis_dir=VISUALIZATION_DIR, save_csv=False):
    """
    Generate a heatmap visualization from a pre-computed matrix.
    
    Args:
        win_rate_matrix: DataFrame with probabilities as index and time bins as columns
        out_dir: Output directory for CSV data files
        vis_dir: Output directory for PNG visualization files
        save_csv: Whether to save the matrix as CSV (if False, assumes it already exists)
    """
    print("\n  Building smoothed calibration heatmap...")
    
    # ── time bin edges and labels ────────────────────────────────────────
    BIN_SECONDS = 2400 / NUM_TIME_BINS  # 120 s per bin (2 minutes)
    
    # ── save raw matrix as CSV (if requested) ───────────────────────────
    if save_csv:
        csv_path = os.path.join(out_dir, "smoothed_heatmap_data.csv")
        df_out = win_rate_matrix.copy()
        df_out.index.name = "kalshi_prob_pct"
        df_out.columns.name = "time_bin"
        df_out.to_csv(csv_path)
        print(f"    CSV ({len(win_rate_matrix)} × {len(win_rate_matrix.columns)}) → {csv_path}")
    
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
    data_plot = win_rate_matrix.iloc[::-1]
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
        cbar_kws={"label": "GAM-Smoothed Win Probability", "shrink": 0.6, "pad": 0.02},
        ax=ax,
        xticklabels=True,
        annot_kws={"fontsize": 5.5, "fontweight": "bold", "color": "white"},
    )
    
    # Explicitly set all annotation text to white
    for text in heatmap.texts:
        text.set_color("white")
    
    # ── y-axis: show every 5th % ─────────────────────────────────────────
    flipped_index = list(data_plot.index)
    ytick_positions = []
    ytick_labels = []
    for i, prob in enumerate(flipped_index):
        if prob % 5 == 0:
            ytick_positions.append(i + 0.5)   # centre of cell
            ytick_labels.append(f"{prob}%")
    
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    
    # ── x-axis: show every other label (half as many) ────────────────────
    x_tick_positions = ax.get_xticks()
    x_tick_labels = ax.get_xticklabels()
    # Keep every other label (indices 0, 2, 4, ...)
    x_tick_positions_filtered = [x_tick_positions[i] for i in range(0, len(x_tick_positions), 2)]
    x_tick_labels_filtered = [x_tick_labels[i].get_text() for i in range(0, len(x_tick_labels), 2)]
    ax.set_xticks(x_tick_positions_filtered)
    ax.set_xticklabels(x_tick_labels_filtered, fontsize=11, rotation=0)
    
    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12)
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12)
    ax.set_title(
        "GAM-Smoothed Calibration Heatmap — Kalshi Win Probability vs. Empirical Win Rate\n"
        f"(99 prob rows × {NUM_TIME_BINS} time bins · {int(BIN_SECONDS/60)}-min buckets · smoothed via GAM)",
        fontsize=15,
        pad=16,
    )
    
    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=10)
    
    plt.tight_layout()
    os.makedirs(vis_dir, exist_ok=True)
    png_path = os.path.join(vis_dir, "smoothed_heatmap.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Heatmap image           → {png_path}")
    
    return win_rate_matrix


def generate_smoothed_heatmap(gam, out_dir=OUTPUT_DIR, vis_dir=VISUALIZATION_DIR):
    """
    Generate a smoothed calibration heatmap using the GAM model.
    
    Time buckets: 2-minute bins (20 bins for 40-minute game).
    Probability:  1 row per integer percentage (1%–99%).
    
    Grid: 99 prob rows × 20 time bins.
    Also exports the underlying matrix as a CSV.
    """
    # ── time bin edges and labels ────────────────────────────────────────
    TIME_EDGES = np.linspace(0, 2400, NUM_TIME_BINS + 1)
    TIME_LABELS = [f"{int(lo/60)}-{int(hi/60)} min"
                   for lo, hi in zip(TIME_EDGES[:-1], TIME_EDGES[1:])]
    
    # ── probability rows ─────────────────────────────────────────────────
    probs = np.arange(1, 100)   # 1% … 99%
    
    # ── evaluate GAM at the centre of each (time_bin, prob) cell ─────────
    bin_centres_sec = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2.0
    bin_centres_frac = bin_centres_sec / 2400.0
    prob_frac = probs / 100.0
    
    T, P = np.meshgrid(bin_centres_frac, prob_frac)
    grid = np.column_stack([T.ravel(), P.ravel()])
    Z = gam.predict_proba(grid).reshape(len(probs), NUM_TIME_BINS)
    
    # ── build pandas matrices for seaborn ────────────────────────────────
    win_rate_matrix = pd.DataFrame(Z, index=probs, columns=TIME_LABELS)
    
    # Generate the plot and save CSV
    return plot_heatmap_from_matrix(win_rate_matrix, out_dir, vis_dir, save_csv=True)


def load_smoothed_data_from_csv(csv_path):
    """Load smoothed heatmap data from CSV file."""
    print(f"\n  Loading smoothed data from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"    Loaded matrix: {len(df)} rows × {len(df.columns)} columns")
    return df


def main():
    """Main function to generate smoothed heatmap."""
    parser = argparse.ArgumentParser(description="Generate smoothed calibration heatmap")
    parser.add_argument("--JustGraph", action="store_true", 
                       help="Skip model fitting and just generate graph from existing CSV")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    csv_path = os.path.join(OUTPUT_DIR, "smoothed_heatmap_data.csv")
    
    # If --JustGraph flag is set, skip model fitting
    if args.JustGraph:
        print(f"\n{'='*70}")
        print("  SMOOTHED DATA HEATMAP GENERATOR")
        print(f"{'='*70}")
        
        if not os.path.exists(csv_path):
            print(f"\n  ERROR: {csv_path} not found!")
            print("  Cannot generate graph without smoothed data CSV.")
            print("  Run without --JustGraph flag to generate the data first.")
            sys.exit(1)
        
        print("\n  --JustGraph flag set. Skipping GAM fitting...")
        
        # Load existing smoothed data and regenerate image
        smoothed_matrix = load_smoothed_data_from_csv(csv_path)
        plot_heatmap_from_matrix(smoothed_matrix, vis_dir=VISUALIZATION_DIR, save_csv=False)
    else:
        # Check if smoothed data already exists
        if os.path.exists(csv_path):
            print(f"\n{'='*70}")
            print("  SMOOTHED DATA HEATMAP GENERATOR")
            print(f"{'='*70}")
            print("\n  Found existing smoothed data CSV. Skipping GAM fitting...")
            
            # Load existing smoothed data and regenerate image
            smoothed_matrix = load_smoothed_data_from_csv(csv_path)
            plot_heatmap_from_matrix(smoothed_matrix, vis_dir=VISUALIZATION_DIR, save_csv=False)
        else:
            # 1. Load and prepare data
            df = load_and_prepare_data(INPUT_FILE)
            
            # 2. Fit GAM model for smoothing
            gam = fit_gam_model(df)
            
            # 3. Generate smoothed heatmap
            smoothed_matrix = generate_smoothed_heatmap(gam)
    
    print(f"\n  All outputs saved:")
    print(f"    - {VISUALIZATION_DIR}/smoothed_heatmap.png")
    print(f"    - {OUTPUT_DIR}/smoothed_heatmap_data.csv")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
