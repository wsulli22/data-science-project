"""
merged_time_and_prob_data.py

Takes a Kalshi game ID and a Kalshi team abbreviation, then:
  1. Looks up the corresponding ESPN game ID from the mappings CSV.
  2. Fetches Kalshi OHLC candlestick win-probability data.
  3. Fetches ESPN play-by-play timestamps (wallclock → game-clock mapping).
  4. Merges the two on wallclock time so every candlestick gets a game-clock value.
  5. Plots win probability (0–100 %) against game-clock time (0–2 400 s).

Usage:
    python merged_time_and_prob_data.py <kalshi_game_id> <team_abbr>

Example:
    python merged_time_and_prob_data.py KXNCAAMBGAME-26FEB14CLEMDUKE CLEM
"""

import sys
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from get_kalshi_game_data import get_kalshi_game_data
from get_espn_game_timestamp_mapings import get_espn_game_timestamp_mapping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAPPINGS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "kalshi_espn_game_mappings.csv")
REGULATION_SECONDS = 2400          # 2 halves × 20 min
HALF_SECONDS = 1200                # 20 min


# ---------------------------------------------------------------------------
# Helper: look up ESPN game ID from the Kalshi ↔ ESPN mappings CSV
# ---------------------------------------------------------------------------
def lookup_espn_game_id(kalshi_game_id: str) -> str:
    """Return the ESPN game ID that corresponds to *kalshi_game_id*."""
    with open(MAPPINGS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["kalshi_game_id"] == kalshi_game_id:
                return row["espn_game_id"]
    raise KeyError(
        f"Kalshi game '{kalshi_game_id}' not found in {MAPPINGS_CSV}"
    )


# ---------------------------------------------------------------------------
# Core: fetch, merge, and return a game-clock-aligned DataFrame
# ---------------------------------------------------------------------------
def build_merged_timeseries(kalshi_game_id: str,
                            team_abbr: str) -> pd.DataFrame:
    """
    Fetch Kalshi candlestick data and ESPN play-by-play data, then align
    them on wallclock time so each candlestick row gets a game-clock value.

    Returns a DataFrame with (at least) these columns:
        game_elapsed_seconds – continuous seconds from tip-off
        win_prob             – win probability for *team_abbr* (0.0–1.0)
        win_prob_pct         – same, scaled to 0–100 %
        period               – game period (1 = 1st half, 2 = 2nd half, …)
    """
    # 1. Resolve ESPN game ID
    espn_game_id = lookup_espn_game_id(kalshi_game_id)
    print(f"Kalshi {kalshi_game_id}  →  ESPN {espn_game_id}")

    # 2. Fetch Kalshi candlestick data
    kalshi_df = get_kalshi_game_data(kalshi_game_id, team_abbr)
    kalshi_df["wallclock_ts"] = (
        pd.to_datetime(kalshi_df["wallclock_ts"]).dt.tz_localize(None)
    )
    kalshi_df = kalshi_df.sort_values("wallclock_ts").reset_index(drop=True)

    # 3. Fetch ESPN play-by-play timestamps
    espn_df = get_espn_game_timestamp_mapping(espn_game_id)
    espn_df["wallclock_ts"] = (
        pd.to_datetime(espn_df["wallclock_ts"]).dt.tz_localize(None)
    )
    espn_df = espn_df.sort_values("wallclock_ts").reset_index(drop=True)

    # 4. Merge-asof: for each Kalshi candle, find the most recent ESPN play
    merged = pd.merge_asof(
        kalshi_df,
        espn_df[["wallclock_ts", "period", "clock_display",
                 "game_elapsed_seconds"]],
        on="wallclock_ts",
        direction="backward",
    )

    # Drop rows that fell before the first ESPN play (no game-clock match)
    merged = merged.dropna(subset=["game_elapsed_seconds"]).copy()

    # 5. Scale probability to percent for easier plotting
    merged["win_prob_pct"] = merged["win_prob"] * 100.0

    # Also add OHLC in percent
    for col in ("win_prob_open", "win_prob_close",
                "win_prob_mean", "win_prob_previous"):
        if col in merged.columns:
            merged[f"{col}_pct"] = merged[col] * 100.0

    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_win_probability(merged: pd.DataFrame,
                         kalshi_game_id: str,
                         team_abbr: str,
                         save_path: str | None = None) -> None:
    """
    Plot win probability (0–100 %) vs game-clock time (0–2 400 s) with
    OHLC-style candlesticks and a half-time divider.
    """
    df = merged.sort_values("game_elapsed_seconds").copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    # --- OHLC candlestick bars ---------------------------------------------------
    has_ohlc = all(
        c in df.columns
        for c in ("win_prob_open_pct", "win_prob_close_pct",
                   "win_prob_mean_pct")
    )

    if has_ohlc:
        # Draw thin high-low lines and thicker open-close bodies
        bar_width = 8  # seconds width for the body
        for _, row in df.iterrows():
            x = row["game_elapsed_seconds"]
            o = row.get("win_prob_open_pct")
            c = row.get("win_prob_close_pct")
            h = row.get("win_prob_pct")       # use close as proxy for high
            l = row.get("win_prob_open_pct")   # and open as proxy for low

            if pd.isna(o) or pd.isna(c):
                continue

            # Determine colour: green if close >= open, red otherwise
            colour = "#2ca02c" if c >= o else "#d62728"

            # Thin wick line (low → high)
            low_val = min(o, c)
            high_val = max(o, c)
            ax.plot([x, x], [low_val, high_val], color=colour,
                    linewidth=0.8, zorder=2)

            # Thick body (open → close)
            body_bottom = min(o, c)
            body_height = abs(c - o) or 0.3   # tiny body if open == close
            ax.bar(x, body_height, bottom=body_bottom, width=bar_width,
                   color=colour, edgecolor=colour, linewidth=0.5, zorder=3)

    # --- Overlay a smoothed line for readability --------------------------------
    ax.plot(
        df["game_elapsed_seconds"],
        df["win_prob_pct"],
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.85,
        label=f"{team_abbr} Win Prob (close)",
        zorder=4,
    )

    # --- Half-time line ---------------------------------------------------------
    ax.axvline(
        x=HALF_SECONDS, color="grey", linestyle="--", linewidth=1,
        label="Half-time"
    )

    # --- 50 % reference line ----------------------------------------------------
    ax.axhline(y=50, color="lightgrey", linestyle=":", linewidth=0.8)

    # --- Axes formatting --------------------------------------------------------
    ax.set_xlim(0, REGULATION_SECONDS)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Game Clock (seconds from tip-off)", fontsize=12)
    ax.set_ylabel("Win Probability (%)", fontsize=12)
    ax.set_title(
        f"Win Probability vs Game Clock\n"
        f"{kalshi_game_id}  —  {team_abbr}",
        fontsize=14,
    )

    # Nice x-axis tick labels showing minutes
    def _sec_to_label(x, _pos):
        m, s = divmod(int(x), 60)
        return f"{m}:{s:02d}"

    ax.xaxis.set_major_locator(mticker.MultipleLocator(300))   # every 5 min
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(60))    # every 1 min
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_sec_to_label))

    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # --- Save / show ------------------------------------------------------------
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"win_prob_{kalshi_game_id}_{team_abbr}.png",
        )
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) >= 3:
        kalshi_game_id = sys.argv[1]
        team_abbr = sys.argv[2].upper()
    else:
        # Default example (change as needed)
        kalshi_game_id = "KXNCAAMBGAME-26FEB14CLEMDUKE"
        team_abbr = "CLEM"
        print(f"No arguments given – using default: {kalshi_game_id} {team_abbr}")

    merged = build_merged_timeseries(kalshi_game_id, team_abbr)
    print(f"\nMerged DataFrame: {len(merged)} rows")
    print(merged[["wallclock_ts", "game_elapsed_seconds", "period",
                   "win_prob_pct"]].head(20).to_string())

    plot_win_probability(merged, kalshi_game_id, team_abbr)


if __name__ == "__main__":
    main()
