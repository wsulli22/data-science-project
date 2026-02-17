#!/usr/bin/env python3
"""
strategy.py

Betting strategy builder for Kalshi college basketball markets.

Uses three inputs:
  1. GAM-calibrated "true" probabilities  (from model.py)
  2. Frequency of each (time, prob) cell  (from raw game data)
  3. Kalshi fee structure

Produces:
  - Edge matrix:              calibrated_prob − kalshi_prob
  - EV-per-trade matrix:      expected value of $1 bet in each cell
  - Frequency-weighted EV:    expected profit per game from each cell
  - Kelly fraction:           optimal bet sizing
  - Strategy heatmaps & summary statistics

Core idea
---------
Kalshi contracts cost `p` cents and pay $1 if the event happens.
If our model says the TRUE probability is `p_true`:
  • Buy YES when p_true > p  (Kalshi underprices the win)
  • Buy NO  when p_true < p  (Kalshi overprices the win)
  • Expected value = p_true − p  (for YES)  or  p − p_true  (for NO)
    … minus Kalshi's fee on winning trades.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_CSV       = "GeneratedDataFiles/all_games_merged_clean_GOOD.csv"
OUTPUT_DIR      = "GeneratedDataFiles"
NUM_TIME_BINS   = 20          # 2-min bins  (matches calibration heatmap)
KALSHI_FEE_RATE = 0.07        # 7 ¢ per $1 profit on winning trades
MIN_FREQ_GAMES  = 0.05        # minimum avg occurrences per game to trade a cell
MIN_EDGE        = 0.00        # minimum |edge| to consider (before fees)
MIN_OBS_CELL    = 30          # minimum observations in a cell for reliable freq

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Build the frequency matrix from raw game data
# ══════════════════════════════════════════════════════════════════════════════
def build_frequency_matrix(csv_path=INPUT_CSV, num_time_bins=NUM_TIME_BINS):
    """
    Count how many observations fall into each (time_bin, prob_int) cell
    in the historical data, then normalise by number of games.
    
    Returns
    -------
    freq_per_game : pd.DataFrame   (index=prob 1..99, columns=time bin labels)
        Average number of times per game we observe each cell.
    raw_counts    : pd.DataFrame
        Raw observation counts.
    n_games       : int
    """
    df = pd.read_csv(csv_path)
    n_games = df["kalshi_event"].nunique()
    print(f"  Loaded {len(df):,} rows from {n_games:,} games")

    # ── time bins ────────────────────────────────────────────────────────
    time_edges  = np.linspace(0, 2400, num_time_bins + 1)
    time_labels = [f"{int(lo/60)}-{int(hi/60)} min"
                   for lo, hi in zip(time_edges[:-1], time_edges[1:])]

    df["time_bin"] = pd.cut(
        df["game_elapsed_seconds"],
        bins=time_edges, labels=time_labels,
        right=False, include_lowest=True,
    )

    # ── probability rows ────────────────────────────────────────────────
    df = df.dropna(subset=["win_prob_pct"])
    df["prob_int"] = df["win_prob_pct"].round(0).astype(int)
    df = df[df["prob_int"].between(1, 99)]

    # ── count ────────────────────────────────────────────────────────────
    counts = (
        df.groupby(["prob_int", "time_bin"], observed=False)
        .size()
        .reset_index(name="n")
    )
    raw_counts = counts.pivot(index="prob_int", columns="time_bin", values="n").fillna(0)

    # Normalise: average occurrences per game
    freq_per_game = raw_counts / n_games

    return freq_per_game, raw_counts, n_games


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Load the GAM-calibrated probability matrix
# ══════════════════════════════════════════════════════════════════════════════
def load_calibrated_matrix(csv_path=None):
    """
    Load the calibration heatmap CSV produced by model.py.
    Returns a DataFrame indexed by prob (1..99), columns = time bin labels.
    """
    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, "calibration_heatmap_data.csv")
    cal = pd.read_csv(csv_path, index_col=0)
    cal.index = cal.index.astype(int)
    cal.index.name = "prob_int"
    print(f"  Loaded calibration matrix: {cal.shape[0]} probs × {cal.shape[1]} time bins")
    return cal


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Compute edge, EV, and Kelly for every cell
# ══════════════════════════════════════════════════════════════════════════════
def compute_strategy_matrices(cal_matrix, freq_matrix, raw_counts,
                              fee_rate=KALSHI_FEE_RATE,
                              min_obs=MIN_OBS_CELL):
    """
    For every (prob, time_bin) cell compute:
      • edge         = calibrated_prob − kalshi_prob
      • direction    = "YES" or "NO"
      • ev_per_trade = expected value of a $1 notional trade (after fees)
      • freq_wt_ev   = ev_per_trade × freq_per_game  (expected $/game from cell)
      • kelly_frac   = Kelly-optimal fraction of bankroll to wager

    Returns dict of DataFrames, all shaped (99 probs × 20 time bins).
    """
    probs = cal_matrix.index.values                    # 1 … 99
    cols  = cal_matrix.columns

    # Kalshi quoted probability (the "price") for each row
    p_kalshi = probs / 100.0                           # shape (99,)

    # GAM calibrated probability matrix
    p_true = cal_matrix.values                         # shape (99, 20)

    # ── edge ─────────────────────────────────────────────────────────────
    edge = p_true - p_kalshi[:, None]                  # positive → buy YES

    # ── EV per $1 notional ───────────────────────────────────────────────
    # YES bet: cost = p_kalshi, win payout = 1
    #   EV_yes = p_true × (1-p_kalshi)×(1-fee) − (1-p_true) × p_kalshi
    ev_yes = (p_true * (1 - p_kalshi[:, None]) * (1 - fee_rate)
              - (1 - p_true) * p_kalshi[:, None])

    # NO bet:  cost = 1−p_kalshi, win payout = 1
    #   EV_no = (1-p_true) × p_kalshi×(1-fee) − p_true × (1-p_kalshi)
    ev_no  = ((1 - p_true) * p_kalshi[:, None] * (1 - fee_rate)
              - p_true * (1 - p_kalshi[:, None]))

    # Best direction: pick whichever bet has higher EV (or 0 if both negative)
    best_ev   = np.maximum(ev_yes, ev_no)
    best_ev   = np.maximum(best_ev, 0)                 # don't trade negative EV
    direction = np.where(ev_yes >= ev_no, "YES", "NO")
    direction = np.where(best_ev <= 0, "—", direction)

    # ── mask cells with insufficient data ────────────────────────────────
    obs_mask = raw_counts.reindex(index=probs, columns=cols).fillna(0).values < min_obs
    best_ev[obs_mask]   = 0
    direction[obs_mask]  = "—"

    # ── Kelly fraction (half-Kelly for safety) ──────────────────────────
    # For YES: b = (1-p_kalshi)/p_kalshi (net odds), f* = (b×p_true − (1-p_true)) / b
    # For NO:  b = p_kalshi/(1-p_kalshi), f* = (b×(1-p_true) − p_true) / b
    # After fees, the net odds are reduced by (1 − fee_rate)
    b_yes = (1 - p_kalshi[:, None]) * (1 - fee_rate) / p_kalshi[:, None]
    kelly_yes = np.where(
        b_yes > 0,
        (b_yes * p_true - (1 - p_true)) / b_yes,
        0,
    )
    b_no  = p_kalshi[:, None] * (1 - fee_rate) / (1 - p_kalshi[:, None])
    kelly_no = np.where(
        b_no > 0,
        (b_no * (1 - p_true) - p_true) / b_no,
        0,
    )
    kelly_full = np.where(ev_yes >= ev_no, kelly_yes, kelly_no)
    kelly_full = np.clip(kelly_full, 0, 1)
    kelly_half = kelly_full / 2                        # half-Kelly
    kelly_half[obs_mask] = 0

    # ── frequency-weighted EV ───────────────────────────────────────────
    freq_vals = freq_matrix.reindex(index=probs, columns=cols).fillna(0).values

    # REALISTIC: you enter at most 1 trade per cell per game.
    # Opportunity rate = P(cell visited at least once) ≈ 1 − exp(−freq)
    opportunity_rate = 1 - np.exp(-freq_vals)
    freq_wt_ev = best_ev * opportunity_rate            # $/game from this cell (1 trade max)

    # ── package into DataFrames ──────────────────────────────────────────
    def to_df(arr):
        return pd.DataFrame(arr, index=probs, columns=cols)

    return {
        "edge":            to_df(edge),
        "direction":       pd.DataFrame(direction, index=probs, columns=cols),
        "ev_per_trade":    to_df(best_ev),
        "freq_per_game":   to_df(freq_vals),
        "opportunity_rate": to_df(opportunity_rate),
        "freq_wt_ev":      to_df(freq_wt_ev),
        "kelly_half":      to_df(kelly_half),
        "raw_counts":      to_df(raw_counts.reindex(index=probs, columns=cols).fillna(0).values),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap(data, title, cbar_label, fname, cmap="RdYlGn", center=None,
             vmin=None, vmax=None, fmt_pct=True, figsize=(16, 28),
             mask=None, annot_thresh=None):
    """Helper: produce a heatmap matching the project style."""
    data_plot  = data.iloc[::-1]
    if mask is not None:
        mask_plot = mask.iloc[::-1]
    else:
        mask_plot = None

    # Build annotation matrix
    annot = data_plot.copy().astype(object)
    for r in data_plot.index:
        for c in data_plot.columns:
            v = data_plot.loc[r, c]
            if mask_plot is not None and mask_plot.loc[r, c]:
                annot.loc[r, c] = ""
            elif pd.isna(v) or v == 0:
                annot.loc[r, c] = ""
            elif fmt_pct:
                annot.loc[r, c] = f"{v*100:.1f}%"
            else:
                annot.loc[r, c] = f"{v:.3f}"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data_plot,
        mask=mask_plot,
        annot=annot, fmt="",
        cmap=cmap, center=center, vmin=vmin, vmax=vmax,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.6, "pad": 0.02},
        ax=ax,
        xticklabels=True,
        annot_kws={"fontsize": 5, "fontweight": "bold"},
    )
    ax.set_facecolor("#d9d9d9")

    # y-ticks every 5 %
    flipped = list(data_plot.index)
    ypos, ylab = [], []
    for i, prob in enumerate(flipped):
        if prob % 5 == 0:
            ypos.append(i + 0.5)
            ylab.append(f"{prob}%")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=10)

    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12)
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12)
    ax.set_title(title, fontsize=15, pad=16)
    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


def plot_strategy_heatmaps(strat, n_games):
    """Generate all strategy visualisation heatmaps."""
    no_trade = (strat["ev_per_trade"] <= 0)

    # 1. Edge heatmap (calibrated − Kalshi)
    print("\n  Plotting edge heatmap …")
    _heatmap(
        strat["edge"],
        "Calibration Edge — (GAM True Prob − Kalshi Quoted Prob)\n"
        "Green = Kalshi underprices (buy YES) · Red = Kalshi overprices (buy NO)",
        "Edge (true − quoted)", "strategy_edge.png",
        cmap="RdBu", center=0, vmin=-0.15, vmax=0.15,
    )

    # 2. EV per $1 trade
    print("  Plotting EV-per-trade heatmap …")
    _heatmap(
        strat["ev_per_trade"],
        f"Expected Value per $1 Trade (after {KALSHI_FEE_RATE*100:.0f}% Kalshi fee)\n"
        "Only cells with sufficient data shown · Grey = no trade",
        "EV per $1 trade", "strategy_ev_per_trade.png",
        cmap="Greens", vmin=0, vmax=0.10, mask=no_trade,
    )

    # 3. Frequency per game
    print("  Plotting frequency heatmap …")
    _heatmap(
        strat["freq_per_game"],
        f"Observation Frequency per Game ({n_games:,} games)\n"
        "How many times per game each (time, prob) cell is observed",
        "Avg observations / game", "strategy_frequency.png",
        cmap="YlOrRd", vmin=0, vmax=1.0, fmt_pct=False,
    )

    # 4. ★ Frequency-weighted EV — THE KEY CHART
    print("  Plotting frequency-weighted EV heatmap …")
    _heatmap(
        strat["freq_wt_ev"],
        "★ Frequency-Weighted EV per Game (¢ per $1 bankroll unit)\n"
        f"EV × frequency · {KALSHI_FEE_RATE*100:.0f}% fee · Only profitable cells",
        "Expected ¢ / game", "strategy_freq_weighted_ev.png",
        cmap="YlGn", vmin=0, vmax=0.02, mask=no_trade, fmt_pct=False,
    )

    # 5. Kelly fraction
    print("  Plotting Kelly fraction heatmap …")
    _heatmap(
        strat["kelly_half"],
        "Half-Kelly Bet Sizing (fraction of bankroll per trade)\n"
        "Conservative sizing · Grey = do not trade",
        "Half-Kelly fraction", "strategy_kelly.png",
        cmap="YlGn", vmin=0, vmax=0.10, mask=no_trade,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Strategy summary table & stats
# ══════════════════════════════════════════════════════════════════════════════
def print_strategy_summary(strat, n_games, fee_rate=KALSHI_FEE_RATE):
    """Print comprehensive strategy statistics."""
    ev   = strat["ev_per_trade"].values
    freq = strat["freq_per_game"].values
    fwev = strat["freq_wt_ev"].values
    edge = strat["edge"].values
    kelly = strat["kelly_half"].values
    direction = strat["direction"].values
    raw  = strat["raw_counts"].values

    opp  = strat["opportunity_rate"].values

    tradeable  = ev > 0
    n_cells    = ev.size
    n_trade    = tradeable.sum()
    n_yes      = ((direction == "YES") & tradeable).sum()
    n_no       = ((direction == "NO") & tradeable).sum()

    total_freq_wt_ev = fwev.sum()
    # Weighted average EV (weighted by opportunity rate, not raw freq)
    total_opp = opp[tradeable].sum()
    avg_ev    = np.average(ev[tradeable], weights=opp[tradeable]) if total_opp > 0 else 0

    # Number of distinct trades per game (max 1 per cell, weighted by opportunity)
    trades_per_game = opp[tradeable].sum()

    print(f"\n{'='*70}")
    print("  TRADING STRATEGY SUMMARY")
    print(f"{'='*70}")
    print(f"""
  Data
  ────
    Games in dataset:          {n_games:,}
    Total grid cells:          {n_cells:,}  (99 prob × 20 time bins)
    Tradeable cells (EV > 0):  {n_trade:,}  ({n_trade/n_cells*100:.1f}%)
      → Buy YES signals:       {n_yes:,}
      → Buy NO signals:        {n_no:,}

  Kalshi Fee
  ──────────
    Fee on winning trades:     {fee_rate*100:.0f}%
    (Applied to profit only, not on losses)

  Expected Value  (realistic: max 1 trade per cell per game)
  ──────────────
    Avg EV per $1 trade (opp-weighted):  {avg_ev*100:.2f}¢
    Distinct trades per game (expected):  {trades_per_game:.1f}
    Total EV per game (all cells):       {total_freq_wt_ev*100:.2f}¢ per $1 per trade
""")

    # ── Top 20 cells by frequency-weighted EV ────────────────────────────
    probs = strat["freq_wt_ev"].index
    cols  = strat["freq_wt_ev"].columns
    records = []
    for p in probs:
        for c in cols:
            fwev_val = strat["freq_wt_ev"].loc[p, c]
            if fwev_val > 0:
                records.append({
                    "prob%":      p,
                    "time_bin":   c,
                    "direction":  strat["direction"].loc[p, c],
                    "edge":       strat["edge"].loc[p, c],
                    "ev/$1":      strat["ev_per_trade"].loc[p, c],
                    "freq/game":  strat["freq_per_game"].loc[p, c],
                    "fw_ev/game": fwev_val,
                    "½kelly":    strat["kelly_half"].loc[p, c],
                    "obs":        int(strat["raw_counts"].loc[p, c]),
                })

    top = pd.DataFrame(records).sort_values("fw_ev/game", ascending=False).head(30)
    print("  Top 30 Cells by Frequency-Weighted EV (most profitable per game):")
    print("  " + "─" * 95)
    print(f"  {'Prob%':>5}  {'Time Bin':<12} {'Dir':>3}  {'Edge':>7}  {'EV/$1':>7}  "
          f"{'Freq/Gm':>8}  {'FW-EV/Gm':>9}  {'½Kelly':>7}  {'Obs':>6}")
    print("  " + "─" * 95)
    for _, r in top.iterrows():
        print(f"  {r['prob%']:>5}  {r['time_bin']:<12} {r['direction']:>3}  "
              f"{r['edge']:>+7.3f}  {r['ev/$1']:>7.4f}  "
              f"{r['freq/game']:>8.3f}  {r['fw_ev/game']:>9.5f}  "
              f"{r['½kelly']:>7.3f}  {r['obs']:>6}")
    print()

    # ── Aggregate by time bin ────────────────────────────────────────────
    print("  Profit by Time Bin (sum of freq-weighted EV across all probs):")
    print("  " + "─" * 60)
    time_ev = strat["freq_wt_ev"].sum(axis=0).sort_values(ascending=False)
    for t, v in time_ev.items():
        bar = "█" * int(v * 5000)
        print(f"    {t:<12}  {v*100:>6.2f}¢  {bar}")
    print()

    # ── Aggregate by probability band ────────────────────────────────────
    print("  Profit by Probability Band (5-pct groups):")
    print("  " + "─" * 60)
    bands = [(1,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,99)]
    for lo, hi in bands:
        mask = (strat["freq_wt_ev"].index >= lo) & (strat["freq_wt_ev"].index <= hi)
        band_ev = strat["freq_wt_ev"].loc[mask].sum().sum()
        bar = "█" * int(band_ev * 5000)
        print(f"    {lo:>2}–{hi:<2}%  {band_ev*100:>6.2f}¢  {bar}")
    print()

    # ── Backtest estimate ────────────────────────────────────────────────
    games_per_day = 8  # rough estimate during college basketball season
    daily_ev = total_freq_wt_ev * games_per_day
    monthly_ev = daily_ev * 30

    # Capital needed: $1 per trade × trades per game × games in flight
    capital_per_game = trades_per_game  # $1 per trade
    daily_capital = capital_per_game * games_per_day
    monthly_roi_pct = (monthly_ev / daily_capital * 100) if daily_capital > 0 else 0

    print(f"  Backtest Projections  (max 1 trade per cell per game)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"    EV per game:                   {total_freq_wt_ev*100:>7.2f}¢  ({trades_per_game:.0f} trades × {avg_ev*100:.2f}¢ avg)")
    print(f"    Games/day (est):               {games_per_day}")
    print(f"    EV per day:                    {daily_ev*100:>7.2f}¢")
    print(f"    Capital deployed per day:      ~${daily_capital:.0f}  ($1/trade × {trades_per_game:.0f} trades × {games_per_day} games)")
    print(f"    EV per month:                  ${monthly_ev:.2f}")
    print(f"    Monthly ROI on deployed capital: {monthly_roi_pct:.1f}%")
    print()
    # Average contract cost per trade (the cost to enter, which ties up capital)
    # YES bet costs p_kalshi, NO bet costs (1 - p_kalshi)
    prob_vals = strat["ev_per_trade"].index.values / 100.0
    dir_vals  = direction
    # Cost matrix: p for YES, (1-p) for NO
    cost_yes = prob_vals[:, None] * np.ones_like(ev)
    cost_no  = (1 - prob_vals[:, None]) * np.ones_like(ev)
    cost_per = np.where(dir_vals == "YES", cost_yes, cost_no)
    avg_cost = np.average(cost_per[tradeable], weights=opp[tradeable]) if total_opp > 0 else 0
    capital_per_game = avg_cost * trades_per_game  # $ tied up per game at $1 per contract

    print(f"  Realistic Projections ($1 per contract)")
    print(f"  ────────────────────────────────────────")
    print(f"    Avg contract cost (capital per trade): ${avg_cost:.2f}")
    print(f"    Capital tied up per game:              ${capital_per_game:.2f}  ({trades_per_game:.0f} trades × ${avg_cost:.2f})")
    print(f"    Capital per day ({games_per_day} games):           ${capital_per_game * games_per_day:.0f}")
    print()
    for n_contracts in [1, 5, 10]:
        ev_game  = total_freq_wt_ev * n_contracts
        ev_day   = ev_game * games_per_day
        ev_month = ev_day * 30
        cap_day  = capital_per_game * games_per_day * n_contracts
        roi_mo   = (ev_month / cap_day * 100) if cap_day > 0 else 0
        print(f"    {n_contracts:>2} contracts/trade:  "
              f"${ev_game:.2f}/game  ${ev_day:.2f}/day  "
              f"${ev_month:.0f}/mo  (needs ~${cap_day:.0f} capital, {roi_mo:.1f}% monthly ROI)")
    print()
    print(f"  ⚠  CAVEATS:")
    print(f"     • Assumes you can fill at the quoted Kalshi mid-price (no spread)")
    print(f"     • Assumes sufficient liquidity at each price level")
    print(f"     • Does not account for execution latency or slippage")
    print(f"     • GAM model has estimation uncertainty (±1-3 pp)")
    print(f"     • Past calibration patterns may not persist into future games")
    print()

    return top


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Save actionable strategy CSV
# ══════════════════════════════════════════════════════════════════════════════
def save_strategy_csv(strat, out_dir=OUTPUT_DIR):
    """Save a flat CSV of all tradeable cells, sorted by freq-weighted EV."""
    probs = strat["freq_wt_ev"].index
    cols  = strat["freq_wt_ev"].columns
    rows = []
    for p in probs:
        for c in cols:
            if strat["ev_per_trade"].loc[p, c] > 0:
                rows.append({
                    "kalshi_prob_pct":    p,
                    "time_bin":           c,
                    "direction":          strat["direction"].loc[p, c],
                    "calibrated_prob":    strat["edge"].loc[p, c] + p/100.0,
                    "edge":              strat["edge"].loc[p, c],
                    "ev_per_dollar":      strat["ev_per_trade"].loc[p, c],
                    "freq_per_game":      strat["freq_per_game"].loc[p, c],
                    "freq_weighted_ev":   strat["freq_wt_ev"].loc[p, c],
                    "half_kelly":         strat["kelly_half"].loc[p, c],
                    "total_observations": int(strat["raw_counts"].loc[p, c]),
                })

    df = pd.DataFrame(rows).sort_values("freq_weighted_ev", ascending=False)
    path = os.path.join(out_dir, "strategy_signals.csv")
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Saved {len(df):,} tradeable signals → {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_strategy():
    """Full strategy pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print("  KALSHI BASKETBALL BETTING STRATEGY BUILDER")
    print(f"{'='*70}")

    # 1. Frequency matrix from raw data
    print("\n  [1/5] Building frequency matrix from game data …")
    freq_matrix, raw_counts, n_games = build_frequency_matrix()

    # 2. Load calibrated probabilities
    print("\n  [2/5] Loading GAM-calibrated probability matrix …")
    cal_matrix = load_calibrated_matrix()

    # 3. Compute strategy matrices
    print("\n  [3/5] Computing edge, EV, and Kelly for every cell …")
    strat = compute_strategy_matrices(cal_matrix, freq_matrix, raw_counts)

    # 4. Visualizations
    print("\n  [4/5] Generating strategy heatmaps …")
    plot_strategy_heatmaps(strat, n_games)

    # 5. Summary
    print("\n  [5/5] Strategy analysis …")
    top = print_strategy_summary(strat, n_games)

    # Save CSV
    save_strategy_csv(strat)

    print(f"\n{'='*70}")
    print("  STRATEGY COMPLETE — all outputs in GeneratedDataFiles/")
    print(f"{'='*70}\n")

    return strat, top


if __name__ == "__main__":
    run_strategy()
