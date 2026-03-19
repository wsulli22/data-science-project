#!/usr/bin/env python3
"""
make_smooth_data_heat_map.py

Reads all_games_merged_clean_GOOD.csv and produces a smoothed calibration
heat map using a GAM (Generalized Additive Model) to smooth the data.

The smoothing is done using a LogisticGAM with splines on game time and
Kalshi probability, similar to the approach in model.py.

Outputs:
    - smoothed_heatmap.png: Visual heatmap of smoothed probabilities
"""

import os
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
INPUT_FILE = "../GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv"
VISUALIZATION_DIR = "GeneratedDataAndVisualizations"  # PNG output
NUM_TIME_BINS = 40          # Number of time bins (1-minute bins for 40 min game)
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


def calculate_distribution_percentage_matrix(df_raw):
    """
    Calculate percentage matrix showing what % of observations in each time column
    fall into each probability cell, calculated separately for above and below 50%.
    
    For each time column:
    - Cells >= 50%: percentage of observations with prob >= 50% that fall into that cell
    - Cells < 50%: percentage of observations with prob < 50% that fall into that cell
    
    This ensures percentages above 50% sum to 100% and below 50% sum to 100%.
    
    Returns:
        DataFrame with same shape as win_rate_matrix, containing percentages
    """
    # ── time bin edges and labels ────────────────────────────────────────
    TIME_EDGES = np.linspace(0, 2400, NUM_TIME_BINS + 1)
    TIME_LABELS = [f"{int(lo/60)}-{int(hi/60)} min"
                   for lo, hi in zip(TIME_EDGES[:-1], TIME_EDGES[1:])]
    
    # ── probability rows ─────────────────────────────────────────────────
    probs = np.arange(1, 100)   # 1% … 99%
    
    # Prepare data
    df_counts = df_raw.copy()
    df_counts["prob_int"] = df_counts["win_prob_pct"].round(0).astype(int)
    df_counts = df_counts[df_counts["prob_int"].between(1, 99)]
    
    df_counts["time_bin"] = pd.cut(
        df_counts["game_elapsed_seconds"],
        bins=TIME_EDGES,
        labels=TIME_LABELS,
        right=False,
        include_lowest=True,
    )
    
    # Initialize percentage matrix
    pct_matrix = pd.DataFrame(0.0, index=probs, columns=TIME_LABELS)
    
    # For each time column, calculate percentages separately for above/below 50%
    for time_bin in TIME_LABELS:
        time_data = df_counts[df_counts["time_bin"] == time_bin]
        if len(time_data) == 0:
            continue
        
        # Split data into above and below 50%
        time_data_above_50 = time_data[time_data["prob_int"] >= 50]
        time_data_below_50 = time_data[time_data["prob_int"] < 50]
        
        total_above_50 = len(time_data_above_50)
        total_below_50 = len(time_data_below_50)
        
        # For each probability level
        for prob in probs:
            if prob >= 50:
                # For >=50%: count observations with exactly this probability
                count = len(time_data_above_50[time_data_above_50["prob_int"] == prob])
                if total_above_50 > 0:
                    pct_matrix.loc[prob, time_bin] = (count / total_above_50) * 100
            else:
                # For <50%: count observations with exactly this probability
                count = len(time_data_below_50[time_data_below_50["prob_int"] == prob])
                if total_below_50 > 0:
                    pct_matrix.loc[prob, time_bin] = (count / total_below_50) * 100
    
    return pct_matrix


def plot_heatmap_from_matrix(win_rate_matrix, count_matrix=None, df_raw=None, vis_dir=VISUALIZATION_DIR):
    """
    Generate a heatmap visualization from a pre-computed matrix.
    
    Args:
        win_rate_matrix: DataFrame with probabilities as index and time bins as columns
        count_matrix: Optional DataFrame with observation counts (same shape as win_rate_matrix)
        df_raw: Optional raw dataframe for calculating distribution percentages (annotations)
        vis_dir: Output directory for PNG visualization files
    """
    print("\n  Building smoothed calibration heatmap...")
    
    # ── time bin edges and labels ────────────────────────────────────────
    BIN_SECONDS = 2400 / NUM_TIME_BINS  # seconds per time bin
    
    # ── Calculate distribution percentage matrix if raw data is available (for annotations) ──────────────────────
    dist_pct_matrix = None
    if df_raw is not None:
        print("    Calculating distribution percentages...")
        dist_pct_matrix = calculate_distribution_percentage_matrix(df_raw)
    
    # ── annotation matrix (show "XX% (YY%)" in each cell if distribution percentages available, else "XX%") ──────────────────────
    
    annot_matrix = win_rate_matrix.copy().astype(object)
    for r in win_rate_matrix.index:
        for c in win_rate_matrix.columns:
            val = win_rate_matrix.loc[r, c]
            if pd.isna(val):
                annot_matrix.loc[r, c] = ""
            else:
                if dist_pct_matrix is not None and r in dist_pct_matrix.index and c in dist_pct_matrix.columns:
                    dist_pct = dist_pct_matrix.loc[r, c]
                    if pd.isna(dist_pct) or dist_pct == 0:
                        annot_matrix.loc[r, c] = f"{val*100:.2f}%"
                    else:
                        # Show distribution percentage (rounded)
                        pct = round(dist_pct)
                        annot_matrix.loc[r, c] = f"{val*100:.2f}% ({pct}%)"
                else:
                    annot_matrix.loc[r, c] = f"{val*100:.2f}%"
    
    # ── flip so high probability is on top ───────────────────────────────
    data_plot = win_rate_matrix.iloc[::-1]
    annot_plot = annot_matrix.iloc[::-1]
    
    # ── plot ─────────────────────────────────────────────────────────────
    # Keep the heatmap's aspect consistent when NUM_TIME_BINS changes:
    # if we double time-bin resolution, the plot should be about twice as wide.
    fig_width = 24 * (NUM_TIME_BINS / 20)
    fig, ax = plt.subplots(figsize=(fig_width, 28))
    
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
    
    # ── x-axis: show a proportional subset of labels for readability ───
    x_tick_positions = ax.get_xticks()
    x_tick_labels = ax.get_xticklabels()
    # For 20 bins -> show ~10 labels (step=2); for 40 bins -> step=4.
    label_step = max(1, int(round(NUM_TIME_BINS / 10)))
    x_tick_positions_filtered = [
        x_tick_positions[i] for i in range(0, len(x_tick_positions), label_step)
    ]
    x_tick_labels_filtered = [
        x_tick_labels[i].get_text() for i in range(0, len(x_tick_labels), label_step)
    ]
    ax.set_xticks(x_tick_positions_filtered)
    ax.set_xticklabels(x_tick_labels_filtered, fontsize=11, rotation=0)
    
    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12)
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12)
    title_text = "GAM-Smoothed Calibration Heatmap — Kalshi Win Probability vs. Empirical Win Rate\n"
    title_text += f"(99 prob rows × {NUM_TIME_BINS} time bins · {int(BIN_SECONDS/60)}-min buckets · smoothed via GAM)"
    if dist_pct_matrix is not None:
        title_text += "\nNote: Numbers in parentheses indicate % of observations in that time column that fall into each cell (calculated separately for ≥50% and <50%, each summing to 100%)"
    ax.set_title(
        title_text,
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


def generate_smoothed_heatmap(gam, df, vis_dir=VISUALIZATION_DIR):
    """
    Generate a smoothed calibration heatmap using the GAM model.
    
    Time buckets: 1-minute bins (40 bins for 40-minute game).
    Probability:  1 row per integer percentage (1%–99%).
    
    Grid: 99 prob rows × 40 time bins.
    
    Args:
        gam: Fitted GAM model
        df: Raw dataframe with observations (for calculating counts)
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
    
    # ── calculate raw observation counts for each cell ───────────────────
    count_matrix = calculate_count_matrix_from_raw_data(df)
    
    return plot_heatmap_from_matrix(win_rate_matrix, count_matrix, df_raw=df, vis_dir=vis_dir)


def generate_smoothed_heatmap_from_file(input_file=INPUT_FILE, vis_dir=VISUALIZATION_DIR):
    """
    One-line function to generate smoothed heatmap from input file.
    
    Args:
        input_file: Path to the input CSV file
        vis_dir: Output directory for PNG visualization files
    
    Returns:
        The smoothed win rate matrix, or None if no data
    """
    df = load_and_prepare_data(input_file)
    if df is None or len(df) == 0:
        print("\n  No observations after loading/filtering — skipping smoothed heatmap.")
        print("  Fix the merged data (see NaN diagnostics above) and re-run.")
        return None
    gam = fit_gam_model(df)
    return generate_smoothed_heatmap(gam, df, vis_dir)


def calculate_count_matrix_from_raw_data(df_raw):
    """
    Calculate raw observation count matrix from raw data.
    Uses the same binning as the smoothed heatmap.
    """
    # ── time bin edges and labels ────────────────────────────────────────
    TIME_EDGES = np.linspace(0, 2400, NUM_TIME_BINS + 1)
    TIME_LABELS = [f"{int(lo/60)}-{int(hi/60)} min"
                   for lo, hi in zip(TIME_EDGES[:-1], TIME_EDGES[1:])]
    
    # ── probability rows ─────────────────────────────────────────────────
    probs = np.arange(1, 100)   # 1% … 99%
    
    # Round win_prob_pct to nearest integer and bin time
    df_counts = df_raw.copy()
    df_counts["prob_int"] = df_counts["win_prob_pct"].round(0).astype(int)
    df_counts = df_counts[df_counts["prob_int"].between(1, 99)]
    
    df_counts["time_bin"] = pd.cut(
        df_counts["game_elapsed_seconds"],
        bins=TIME_EDGES,
        labels=TIME_LABELS,
        right=False,
        include_lowest=True,
    )
    
    # Count observations per cell
    count_grouped = (
        df_counts.groupby(["prob_int", "time_bin"], observed=False)
        .size()
        .reset_index(name="count")
    )
    
    # Create count matrix, handling case where pivot might have missing values
    try:
        count_matrix = count_grouped.pivot(
            index="prob_int", columns="time_bin", values="count"
        )
        # Align count_matrix with win_rate_matrix (fill missing with 0)
        count_matrix = count_matrix.reindex(index=probs, columns=TIME_LABELS, fill_value=0)
        # Convert to int to avoid float issues
        count_matrix = count_matrix.fillna(0).astype(int)
        return count_matrix
    except Exception as e:
        print(f"    Warning: Could not create count matrix: {e}")
        return None


def main():
    """Main function to generate smoothed heatmap (fits GAM and writes PNG)."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    df = load_and_prepare_data(INPUT_FILE)
    gam = fit_gam_model(df)
    generate_smoothed_heatmap(gam, df)

    print(f"\n  Output saved:")
    print(f"    - {VISUALIZATION_DIR}/smoothed_heatmap.png")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
