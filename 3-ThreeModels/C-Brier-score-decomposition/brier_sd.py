"""
brier_decomposition.py — Murphy (1973) Brier Score Decomposition
for Kalshi live in-game NCAAB win probabilities.

WHAT IT COMPUTES
----------------
The Brier Score (BS) measures mean squared error between forecast
probabilities and binary outcomes.  Murphy's decomposition splits it into
three interpretable pieces:

    BS  =  Reliability  −  Resolution  +  Uncertainty

  • Reliability  (lower is better)
      Measures calibration: how far the average quoted probability in each
      bin is from the actual win rate in that bin.  A perfectly calibrated
      market has Reliability = 0.

  • Resolution  (higher is better)
      Measures informativeness: how much the per-bin win rates vary around
      the overall base rate.  A market that only ever quotes 50% has
      Resolution = 0.

  • Uncertainty  (fixed for a given dataset)
      The irreducible variance of the outcomes (ō × (1 − ō)).  The market
      cannot do better than BS = Uncertainty − Resolution.

  • Skill Score  (higher is better, >0 means beats climatology)
      SS = 1 − BS / Uncertainty
      Equivalent to asking: how much better than always quoting the base
      rate is Kalshi doing?

SAMPLING CONVENTION
-------------------
  • One observation per 60-second game-clock bucket per game
    (first row whose elapsed_seconds falls in that minute-bucket).
    Eliminates per-second autocorrelation while preserving the full
    timeline; consistent with the heatmap pipeline.
  • Favoured-team perspective only (probability ≥ 50) to avoid
    double-counting.

OUTPUT
------
  Prints a formatted numeric summary table to stdout.
  Saves a 4-panel visual dashboard to a .png file.

USAGE
-----
  python brier_decomposition.py
  python brier_decomposition.py --data /path/to/all_games_merged_clean.csv
  python brier_decomposition.py --bins 15  # number of probability bins
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Default path (same convention as the rest of the project) ─────────────────
DEFAULT_DATA_PATH = (
    Path("/Users/benherbst/kalshi_ds_project/data-science-project/")
    / "1-GatheringPreprocessingTransformation"
    / "GeneratedDataFiles"
    / "all_games_merged_clean.csv"
)

# ── Probability bins for Murphy decomposition ─────────────────────────────────
# 10 equal-width bins over [0, 1].  Increase N_PROB_BINS for finer resolution.
DEFAULT_N_PROB_BINS = 10

# ── Time-bucket definitions ───────────────────────────────────────────────────
# Coarse: five narrative stages of a college basketball game.
# Edges are in game_elapsed_seconds (0 = tip-off, 1200 = halftime, 2400 = end regulation).
COARSE_TIME_BINS = [
    (0,    600,  "Pre-half early   (0–10 min)"),
    (600,  1200, "Pre-half late   (10–20 min)"),
    (1200, 1800, "2nd-half early  (20–30 min)"),
    (1800, 2400, "2nd-half late   (30–40 min)"),
    (2400, 9999, "Overtime        (40+ min)  "),
]

# Fine: one bucket per minute of regulation + OT up to ~60 min
# Generated dynamically; each bucket is [60k, 60(k+1)) seconds.
FINE_BIN_WIDTH_SECONDS = 60
FINE_MAX_SECONDS = 3600          # covers regulation + ~2 OT periods


# ── Core Murphy decomposition ─────────────────────────────────────────────────

def brier_decompose(
    forecasts: np.ndarray,   # probabilities in [0, 1]
    outcomes: np.ndarray,    # binary 0/1
    n_bins: int = DEFAULT_N_PROB_BINS,
) -> dict:
    """
    Murphy (1973) Brier Score decomposition.
    """
    n = len(forecasts)
    if n == 0:
        return _empty_result()

    forecasts = np.asarray(forecasts, dtype=float)
    outcomes  = np.asarray(outcomes,  dtype=float)

    # Overall Brier score (raw MSE)
    bs = float(np.mean((forecasts - outcomes) ** 2))

    # Base rate (climatological mean)
    o_bar = float(np.mean(outcomes))

    # Uncertainty = irreducible component
    uncertainty = o_bar * (1.0 - o_bar)

    # Bin edges: 0, 1/K, 2/K, …, 1 (right-inclusive for last bin)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_records = []
    reliability = 0.0
    resolution  = 0.0

    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if k < n_bins - 1:
            mask = (forecasts >= lo) & (forecasts < hi)
        else:
            mask = (forecasts >= lo) & (forecasts <= hi)

        n_k = int(mask.sum())
        if n_k == 0:
            continue

        f_k   = float(forecasts[mask].mean())    # mean forecast in bin
        o_k   = float(outcomes[mask].mean())     # empirical win rate in bin
        edge_frac = n_k / n                      # weight

        reliability += edge_frac * (f_k - o_k) ** 2
        resolution  += edge_frac * (o_k - o_bar) ** 2

        bin_records.append(
            {
                "bin_lo":        round(lo * 100, 1),
                "bin_hi":        round(hi * 100, 1),
                "n":             n_k,
                "mean_forecast": round(f_k * 100, 2),
                "empirical_wr":  round(o_k * 100, 2),
                "edge_pp":       round((o_k - f_k) * 100, 2),   # +ve = underpriced
                "reliability":   round(edge_frac * (f_k - o_k) ** 2, 6),
                "resolution":    round(edge_frac * (o_k - o_bar) ** 2, 6),
            }
        )

    skill_score = 1.0 - (bs / uncertainty) if uncertainty > 0 else float("nan")

    return {
        "n":            n,
        "brier_score":  round(bs, 6),
        "reliability":  round(reliability, 6),
        "resolution":   round(resolution, 6),
        "uncertainty":  round(uncertainty, 6),
        "skill_score":  round(skill_score, 4),
        "base_rate":    round(o_bar, 4),
        "bin_details":  pd.DataFrame(bin_records),
    }


def _empty_result() -> dict:
    return {
        "n": 0, "brier_score": float("nan"), "reliability": float("nan"),
        "resolution": float("nan"), "uncertainty": float("nan"),
        "skill_score": float("nan"), "base_rate": float("nan"),
        "bin_details": pd.DataFrame(),
    }


# ── Data loading and sampling ─────────────────────────────────────────────────

def load_observations(data_path: Path) -> pd.DataFrame:
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
 
    required = {"kalshi_event", "team", "game_elapsed_seconds",
                "win_prob_pct", "team_won"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: Missing columns in CSV: {missing}\n"
            "Check that you're pointing at all_games_merged_clean.csv"
        )
 
    df["game_elapsed_seconds"] = pd.to_numeric(
        df["game_elapsed_seconds"], errors="coerce"
    )
    df["win_prob_pct"] = pd.to_numeric(df["win_prob_pct"], errors="coerce")
    df["team_won"]     = pd.to_numeric(df["team_won"],     errors="coerce")
    df = df.dropna(subset=["game_elapsed_seconds", "win_prob_pct", "team_won"])
 
    if "realworld_timestamp" in df.columns:
        df["realworld_timestamp"] = pd.to_datetime(
            df["realworld_timestamp"], errors="coerce"
        )
        df = df.sort_values(["kalshi_event", "realworld_timestamp"])
    else:
        df = df.sort_values(["kalshi_event", "game_elapsed_seconds"])
 
    # Favoured-team filter
    df = df[df["win_prob_pct"] >= 50.0].copy()
 
    # 60-second bucket sampling
    df["_minute_bucket"] = (df["game_elapsed_seconds"] // 60).astype(int)
    df = (
        df.groupby(["kalshi_event", "_minute_bucket"], sort=False)
        .first()
        .reset_index()
    )
 
    df["forecast"] = df["win_prob_pct"] / 100.0
    df["outcome"]  = df["team_won"].astype(int)
 
    df = df[(df["forecast"] >= 0.0) & (df["forecast"] <= 1.0)]
 
    print(
        f"  {len(df['kalshi_event'].unique()):,} games  |  "
        f"{len(df):,} observations (after 60-s sampling, favoured-team only)"
    )
    return df


# ── Printing helpers ──────────────────────────────────────────────────────────

def _sep(width: int = 70) -> str:
    return "─" * width

def _print_decomposition(result: dict, label: str, indent: int = 0) -> None:
    pad = " " * indent
    n   = result["n"]
    if n == 0:
        print(f"{pad}{label}: no data")
        return

    bs   = result["brier_score"]
    rel  = result["reliability"]
    res_ = result["resolution"]
    unc  = result["uncertainty"]
    ss   = result["skill_score"]
    br   = result["base_rate"]

    bs_pp  = bs * 100
    rel_pp = rel * 100
    res_pp = res_ * 100
    unc_pp = unc * 100

    print(f"{pad}{label}  (n = {n:,})")
    print(f"{pad}  Base rate (empirical win rate)  : {br*100:6.2f}%")
    print(f"{pad}  Brier Score                     : {bs:.6f}  ({bs_pp:.4f} pp²)")
    print(f"{pad}    Reliability  ↓ (calibration)  : {rel:.6f}  ({rel_pp:.4f} pp²)")
    print(f"{pad}    Resolution   ↑ (sharpness)    : {res_:.6f}  ({res_pp:.4f} pp²)")
    print(f"{pad}    Uncertainty  (irreducible)     : {unc:.6f}  ({unc_pp:.4f} pp²)")
    print(f"{pad}  Skill Score (vs climatology)    : {ss:+.4f}")


def _print_bin_details(result: dict, indent: int = 2) -> None:
    pad = " " * indent
    bd  = result.get("bin_details")
    if bd is None or bd.empty:
        return
    header = (
        f"{pad}{'Prob bin':>14}  {'n':>6}  {'Kalshi%':>8}  "
        f"{'Actual%':>8}  {'Edge pp':>8}  {'Reliability':>12}"
    )
    print(header)
    print(pad + _sep(len(header) - indent))
    for _, row in bd.iterrows():
        edge_flag = " ←" if abs(row["edge_pp"]) >= 3.0 else ""
        print(
            f"{pad}{row['bin_lo']:>5.1f}–{row['bin_hi']:>5.1f}%  "
            f"{int(row['n']):>6,}  "
            f"{row['mean_forecast']:>7.2f}%  "
            f"{row['empirical_wr']:>7.2f}%  "
            f"{row['edge_pp']:>+7.2f}  "
            f"{row['reliability']:>12.6f}{edge_flag}"
        )


# ── Plotting logic ────────────────────────────────────────────────────────────

def _style_legend_light(leg) -> None:
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor("#fafafa")
    frame.set_edgecolor("#cccccc")
    for text in leg.get_texts():
        text.set_color("#1a1a1a")


def _apply_light_axes(fig, axes) -> None:
    """White figure/axes, dark text and subtle grid (print / light mode)."""
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333", which="both")
        ax.xaxis.label.set_color("#222222")
        ax.yaxis.label.set_color("#222222")
        ax.title.set_color("#111111")
        for spine in ax.spines.values():
            spine.set_color("#888888")
        ax.grid(True, alpha=0.35, color="#888888")
    fig.suptitle(
        "Kalshi NCAAB Brier Score Decomposition",
        fontsize=16,
        fontweight="bold",
        color="#111111",
    )


def plot_brier_results(overall: dict, fine_rows: list, out_path: str = "brier_visualization.png") -> None:
    """Generates a 4-panel dashboard of the Brier Score decomposition."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _apply_light_axes(fig, axes)

    # 1. Calibration Curve
    ax1 = axes[0, 0]
    bd = overall.get("bin_details")
    if bd is not None and not bd.empty:
        # Plot perfect calibration line (reference on white)
        ax1.plot(
            [50, 100],
            [50, 100],
            linestyle="--",
            color="#555555",
            label="Perfect Calibration",
            alpha=0.9,
        )
        # Plot empirical vs predicted
        ax1.plot(
            bd["mean_forecast"],
            bd["empirical_wr"],
            marker="o",
            linestyle="-",
            color="#2563eb",
            label="Market Calibration",
        )
        ax1.set_xlim(45, 105)
        ax1.set_ylim(45, 105)
        ax1.set_xlabel("Market Implied Probability (%)")
        ax1.set_ylabel("Empirical Win Rate (%)")
        ax1.set_title("Reliability Curve (Calibration)")
        leg1 = ax1.legend()
        _style_legend_light(leg1)

    if fine_rows:
        fdf = pd.DataFrame(fine_rows)
        # Filter out extreme noise from Double/Triple OT for cleaner plots (n >= 30)
        fdf_plot = fdf[fdf['n'] >= 30].copy()

        # 2. Decomposition over time
        ax2 = axes[0, 1]
        ax2.plot(
            fdf_plot["min_start"],
            fdf_plot["brier"],
            label="Brier Score",
            color="#6d28d9",
            linewidth=2,
        )
        ax2.plot(
            fdf_plot["min_start"],
            fdf_plot["unc"],
            label="Uncertainty",
            color="#525252",
            linestyle=":",
            linewidth=2,
        )
        ax2.plot(
            fdf_plot["min_start"],
            fdf_plot["res"],
            label="Resolution",
            color="#15803d",
            linewidth=2,
        )
        ax2.plot(
            fdf_plot["min_start"],
            fdf_plot["rel"],
            label="Reliability",
            color="#dc2626",
            linewidth=2,
        )
        ax2.set_xlabel("Game Minute")
        ax2.set_ylabel("Score Component")
        ax2.set_title("Decomposition Over Time")
        leg2 = ax2.legend()
        _style_legend_light(leg2)

        # 3. Skill Score over time
        ax3 = axes[1, 0]
        ax3.plot(fdf_plot["min_start"], fdf_plot["skill"], color="#0d9488", linewidth=2)
        ax3.axhline(0, color="#737373", linestyle="--", alpha=0.9)
        ax3.set_xlabel("Game Minute")
        ax3.set_ylabel("Skill Score")
        ax3.set_title("Market Skill vs Climatology Over Time")

        # 4. Base Rate over time
        ax4 = axes[1, 1]
        ax4.plot(
            fdf_plot["min_start"],
            fdf_plot["base_rate"] * 100,
            color="#c2410c",
            linewidth=2,
        )
        ax4.set_xlabel("Game Minute")
        ax4.set_ylabel("Empirical Win Rate (%)")
        ax4.set_title("Favoured Team Win Rate Over Time")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close()


# ── Main report ───────────────────────────────────────────────────────────────

def run_report(data_path: Path, n_bins: int) -> None:
    df = load_observations(data_path)
    f  = df["forecast"].values
    o  = df["outcome"].values
    el = df["game_elapsed_seconds"].values

    W = 72
    print()
    print("=" * W)
    print("  BRIER SCORE DECOMPOSITION — Kalshi NCAAB Win Probabilities")
    print("  Murphy (1973) decomposition:  BS = Reliability − Resolution + Uncertainty")
    print("=" * W)

    # ── 1. Overall ────────────────────────────────────────────────────────────
    print()
    print(_sep(W))
    print("  1. OVERALL")
    print(_sep(W))
    overall = brier_decompose(f, o, n_bins)
    _print_decomposition(overall, "All observations", indent=2)
    print()
    print("  Per-probability-bin breakdown:")
    _print_bin_details(overall, indent=4)

    # ── 2. Coarse time buckets ────────────────────────────────────────────────
    print()
    print(_sep(W))
    print("  2. COARSE TIME BUCKETS")
    print(_sep(W))
    coarse_rows = []
    for lo, hi, label in COARSE_TIME_BINS:
        mask = (el >= lo) & (el < hi)
        if mask.sum() == 0:
            continue
        r = brier_decompose(f[mask], o[mask], n_bins)
        coarse_rows.append(
            {
                "bucket":      label,
                "n":           r["n"],
                "brier":       r["brier_score"],
                "reliability": r["reliability"],
                "resolution":  r["resolution"],
                "uncertainty": r["uncertainty"],
                "skill_score": r["skill_score"],
                "base_rate":   r["base_rate"],
            }
        )
        _print_decomposition(r, label, indent=2)
        print()

    if coarse_rows:
        cdf = pd.DataFrame(coarse_rows)
        print("  Coarse summary table:")
        col_w = max(len(r["bucket"]) for r in coarse_rows) + 2
        hdr = (
            f"  {'Bucket':<{col_w}}  {'n':>7}  {'BS':>9}  "
            f"{'Reliab':>9}  {'Resolut':>9}  {'Skill':>7}  {'BaseRate':>8}"
        )
        print(hdr)
        print("  " + _sep(len(hdr) - 2))
        for r in coarse_rows:
            print(
                f"  {r['bucket']:<{col_w}}  {r['n']:>7,}  "
                f"{r['brier']:>9.6f}  {r['reliability']:>9.6f}  "
                f"{r['resolution']:>9.6f}  {r['skill_score']:>+7.4f}  "
                f"{r['base_rate']*100:>7.2f}%"
            )

    # ── 3. Fine per-minute buckets ────────────────────────────────────────────
    print()
    print(_sep(W))
    print("  3. FINE TIME BUCKETS  (1-minute game-clock bins)")
    print(_sep(W))
    bw = FINE_BIN_WIDTH_SECONDS
    max_s = max(el.max(), FINE_MAX_SECONDS)
    minute_edges = np.arange(0, max_s + bw, bw)

    fine_rows = []
    for i in range(len(minute_edges) - 1):
        lo, hi = minute_edges[i], minute_edges[i + 1]
        mask = (el >= lo) & (el < hi)
        if mask.sum() < 5:
            continue
        r = brier_decompose(f[mask], o[mask], n_bins)
        fine_rows.append(
            {
                "min_start": int(lo // 60),
                "n":         r["n"],
                "brier":     r["brier_score"],
                "rel":       r["reliability"],
                "res":       r["resolution"],
                "unc":       r["uncertainty"],
                "skill":     r["skill_score"],
                "base_rate": r["base_rate"],
            }
        )

    if fine_rows:
        fdf = pd.DataFrame(fine_rows)
        hdr = (
            f"  {'Min':>4}  {'n':>6}  {'BS':>9}  "
            f"{'Reliab':>9}  {'Resolut':>9}  {'Skill':>7}  {'BaseRate':>8}"
        )
        print(hdr)
        print("  " + _sep(len(hdr) - 2))
        for r in fine_rows:
            flag = " ◄" if r["rel"] > overall["reliability"] * 1.5 else ""
            print(
                f"  {r['min_start']:>4}  {r['n']:>6,}  "
                f"{r['brier']:>9.6f}  {r['rel']:>9.6f}  "
                f"{r['res']:>9.6f}  {r['skill']:>+7.4f}  "
                f"{r['base_rate']*100:>7.2f}%{flag}"
            )
        print("  (◄ = reliability notably worse than overall)")

        fdf_sorted_rel = fdf.sort_values("rel", ascending=False)
        print()
        print("  Worst-calibrated minutes (highest reliability error):")
        for _, row in fdf_sorted_rel.head(5).iterrows():
            print(
                f"    Minute {int(row['min_start']):>3}  "
                f"reliability={row['rel']:.6f}  "
                f"skill={row['skill']:+.4f}  n={int(row['n']):,}"
            )
        print()
        print("  Best-calibrated minutes (lowest reliability error, min n=30):")
        fdf_dense = fdf[fdf["n"] >= 30].sort_values("rel")
        for _, row in fdf_dense.head(5).iterrows():
            print(
                f"    Minute {int(row['min_start']):>3}  "
                f"reliability={row['rel']:.6f}  "
                f"skill={row['skill']:+.4f}  n={int(row['n']):,}"
            )

    print()
    print("=" * W)
    print("  INTERPRETATION GUIDE")
    print("=" * W)
    print(
        f"""
  Brier Score:   pure MSE between quoted probabilities and outcomes.
                 Perfect = 0.  Uninformative (always quote base rate) = Uncertainty.

  Reliability:   calibration error.  If Kalshi quotes 70% and those games
                 are actually won 70% of the time, Reliability = 0.
                 Any positive value means systematic over- or under-confidence.

  Resolution:    how much the market's probabilities spread out from the
                 base rate.  A market that always quotes 50% has Resolution = 0.
                 Higher is better — it means the market is moving its numbers
                 around meaningfully in response to game events.

  Uncertainty:   purely a function of the base rate; the market cannot change
                 it.  For a balanced sport near 50% base rate, Uncertainty ≈ 0.25.

  Skill Score:   positive means Kalshi beats a naive "always quote base rate"
                 forecast.  Negative means it's worse.

  Edge column:   in the per-bin table, +pp means the true win rate EXCEEDED
                 the quoted probability (Kalshi underpriced the favourite);
                 −pp means Kalshi overpriced.  ← flags |edge| ≥ 3 pp.
"""
    )
    
    # Generate the visualization
    plot_brier_results(overall, fine_rows, "brier_visualization.png")
    print(f"\n[+] Generated visualization dashboard saved to: brier_visualization.png\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Brier Score decomposition for Kalshi NCAAB win probabilities."
    )
    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to all_games_merged_clean.csv",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_N_PROB_BINS,
        help="Number of equal-width probability bins for Murphy decomposition (default 10)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.data.exists():
        sys.exit(
            f"ERROR: Data file not found: {args.data}\n"
            "Run 1-GatheringPreprocessingTransformation/main.py first, or pass "
            "--data /path/to/all_games_merged_clean.csv"
        )
    run_report(args.data, args.bins)