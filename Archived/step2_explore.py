"""
step2_explore.py  –  Step 2 Methodology Exploration

Demonstrates how to align Kalshi OHLC win-probabilities with ESPN
play-by-play game-clock time for game KXNCAAMBGAME-26FEB10MILWIUIN (team MILW).

Research goal: build (game_clock_time, kalshi_probability, did_team_win) triples
for calibration analysis.

Three alignment strategies are compared:
  A) merge_asof backward   – piecewise-constant, conservative
  B) merge_asof nearest    – picks closest ESPN play
  C) Piecewise-linear interpolation of game clock between ESPN plays

Conclusion (see bottom): Strategy A is best for the calibration research question.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── project imports ──────────────────────────────────────────────────────
from get_kalshi_game_data import get_kalshi_game_data
from get_espn_game_timestamp_mapings import get_espn_game_timestamp_mapping
from data_quality_checker import assess_game_quality, print_quality_report

# ── constants ────────────────────────────────────────────────────────────
KALSHI_EVENT   = "KXNCAAMBGAME-26FEB10MILWIUIN"
TEAM           = "MILW"
# The mappings CSV has 401817616 but that's the Jan 11 game (wrong date).
# 401817686 is the second IUIN-vs-MILW game, matching the Feb 10 Kalshi market.
ESPN_GAME_ID   = "401817686"
HALF_SEC       = 1200                  # 20-min half
REG_SEC        = 2400                  # full regulation

# =====================================================================
# 1.  FETCH RAW DATA
# =====================================================================
print("=" * 70)
print("1.  FETCHING RAW DATA")
print("=" * 70)

# ── 1a. Kalshi candlestick data ──────────────────────────────────────
kalshi_raw = get_kalshi_game_data(KALSHI_EVENT, TEAM)
kalshi_raw["wallclock_ts"] = (
    pd.to_datetime(kalshi_raw["wallclock_ts"]).dt.tz_localize(None)
)
kalshi_raw = kalshi_raw.sort_values("wallclock_ts").reset_index(drop=True)

print(f"\nKalshi candles fetched : {len(kalshi_raw)}")
print(f"  First candle ts     : {kalshi_raw['wallclock_ts'].iloc[0]}")
print(f"  Last  candle ts     : {kalshi_raw['wallclock_ts'].iloc[-1]}")
print(f"  Result (did {TEAM} win?): {kalshi_raw['result'].iloc[0]}")
print(f"\nKalshi sample rows:\n{kalshi_raw[['wallclock_ts','win_prob_open','win_prob_close','win_prob','volume']].head(10).to_string()}")

# ── 1b. ESPN play-by-play data ───────────────────────────────────────
espn_raw = get_espn_game_timestamp_mapping(ESPN_GAME_ID)
espn_raw["wallclock_ts"] = (
    pd.to_datetime(espn_raw["wallclock_ts"]).dt.tz_localize(None)
)
espn_raw = espn_raw.sort_values("wallclock_ts").reset_index(drop=True)

print(f"\nESPN plays fetched    : {len(espn_raw)}")
print(f"  First play ts       : {espn_raw['wallclock_ts'].iloc[0]}")
print(f"  Last  play ts       : {espn_raw['wallclock_ts'].iloc[-1]}")
print(f"  Periods present     : {sorted(espn_raw['period'].unique())}")
print(f"\nESPN sample rows:\n{espn_raw.head(10).to_string()}")

# =====================================================================
# 2.  UNDERSTAND THE WALLCLOCK → GAME-CLOCK MAPPING
# =====================================================================
print("\n" + "=" * 70)
print("2.  WALLCLOCK → GAME-CLOCK RELATIONSHIP")
print("=" * 70)

# How much wallclock time does a regulation game consume?
espn_reg = espn_raw[espn_raw["game_elapsed_seconds"] <= REG_SEC]
wallclock_span = (espn_reg["wallclock_ts"].max() - espn_reg["wallclock_ts"].min()).total_seconds()
print(f"\n  Regulation game clock : {REG_SEC} s  (40 min)")
print(f"  Wallclock span        : {wallclock_span:.0f} s  ({wallclock_span/60:.1f} min)")
print(f"  Ratio (wall / game)   : {wallclock_span / REG_SEC:.2f}x")
print(f"    → stoppages add ~{((wallclock_span / REG_SEC) - 1)*100:.0f}% extra wallclock time")

# Gap analysis: how far apart are consecutive ESPN plays?
espn_gaps = espn_raw["wallclock_ts"].diff().dt.total_seconds().dropna()
print(f"\n  ESPN play-by-play gap statistics (seconds):")
print(f"    median  : {espn_gaps.median():.1f}")
print(f"    mean    : {espn_gaps.mean():.1f}")
print(f"    p90     : {espn_gaps.quantile(0.90):.1f}")
print(f"    p99     : {espn_gaps.quantile(0.99):.1f}")
print(f"    max     : {espn_gaps.max():.1f}  (likely halftime)")

# =====================================================================
# 3.  THREE ALIGNMENT STRATEGIES
# =====================================================================
print("\n" + "=" * 70)
print("3.  COMPARING ALIGNMENT STRATEGIES")
print("=" * 70)

# ── Prepare data ─────────────────────────────────────────────────────
# We only want Kalshi candles that fall DURING the game (between first
# and last ESPN play).  Pre-game & post-game candles are noise.

game_start = espn_raw["wallclock_ts"].min()
game_end   = espn_raw["wallclock_ts"].max()
kalshi = kalshi_raw[
    (kalshi_raw["wallclock_ts"] >= game_start) &
    (kalshi_raw["wallclock_ts"] <= game_end)
].copy().reset_index(drop=True)

print(f"\n  Kalshi candles during game: {len(kalshi)} (of {len(kalshi_raw)} total)")
print(f"  Dropped {len(kalshi_raw) - len(kalshi)} pre/post-game candles")

# ── Strategy A: merge_asof backward ─────────────────────────────────
merged_A = pd.merge_asof(
    kalshi[["wallclock_ts", "win_prob", "win_prob_open", "win_prob_close",
            "volume", "result"]].copy(),
    espn_raw[["wallclock_ts", "period", "clock_display",
              "game_elapsed_seconds"]],
    on="wallclock_ts",
    direction="backward",
).dropna(subset=["game_elapsed_seconds"])
merged_A.rename(columns={"game_elapsed_seconds": "ge_backward"}, inplace=True)

# ── Strategy B: merge_asof nearest ──────────────────────────────────
merged_B = pd.merge_asof(
    kalshi[["wallclock_ts"]].copy(),
    espn_raw[["wallclock_ts", "game_elapsed_seconds"]],
    on="wallclock_ts",
    direction="nearest",
).dropna(subset=["game_elapsed_seconds"])
merged_B.rename(columns={"game_elapsed_seconds": "ge_nearest"}, inplace=True)

# ── Strategy C: piecewise-linear interpolation ───────────────────────
# Build a mapping function: wallclock → game_elapsed using np.interp
espn_wc_sec = (espn_raw["wallclock_ts"] - espn_raw["wallclock_ts"].iloc[0]).dt.total_seconds().values
espn_ge     = espn_raw["game_elapsed_seconds"].values

kalshi_wc_sec = (kalshi["wallclock_ts"] - espn_raw["wallclock_ts"].iloc[0]).dt.total_seconds().values
ge_interp = np.interp(kalshi_wc_sec, espn_wc_sec, espn_ge)

# Combine all three into one comparison frame
comparison = merged_A[["wallclock_ts", "win_prob", "win_prob_open",
                        "win_prob_close", "volume", "result",
                        "period", "clock_display", "ge_backward"]].copy()
comparison["ge_nearest"] = merged_B["ge_nearest"].values[:len(comparison)]
comparison["ge_interp"]  = ge_interp[:len(comparison)]

# Show differences
comparison["diff_B_A"] = comparison["ge_nearest"] - comparison["ge_backward"]
comparison["diff_C_A"] = comparison["ge_interp"]  - comparison["ge_backward"]

print("\n  Difference statistics (seconds) between strategies and backward merge:")
print(f"\n  Nearest vs Backward (B-A):")
print(f"    mean abs diff : {comparison['diff_B_A'].abs().mean():.1f} s")
print(f"    max  abs diff : {comparison['diff_B_A'].abs().max():.1f} s")
print(f"\n  Interpolation vs Backward (C-A):")
print(f"    mean abs diff : {comparison['diff_C_A'].abs().mean():.1f} s")
print(f"    max  abs diff : {comparison['diff_C_A'].abs().max():.1f} s")

print("\n  Sample comparison (first 15 candles):")
print(comparison[["wallclock_ts", "ge_backward", "ge_nearest", "ge_interp",
                   "diff_B_A", "diff_C_A", "win_prob"]].head(15).to_string())

# =====================================================================
# 4.  WHY MERGE_ASOF BACKWARD IS THE BEST CHOICE
# =====================================================================
print("\n" + "=" * 70)
print("4.  METHODOLOGY RECOMMENDATION")
print("=" * 70)
print("""
  For the calibration research question, merge_asof BACKWARD is best because:

  (a) Conservative & causal: a Kalshi candle's "close" price is the last
      trade during that minute.  The closest ESPN play BEFORE that moment
      is the most accurate game-clock anchor (we never "look ahead").

  (b) Stoppages handled correctly: during timeouts/halftime, multiple
      Kalshi candles map to the SAME game-clock value – which is correct
      because game time isn't advancing.

  (c) Precision is sufficient: ESPN plays occur every ~{median:.0f}s (median),
      and Kalshi candles are 1-minute wide.  The inherent 60s candle width
      already dominates the alignment error.

  (d) For the heatmap bins (typically 2–5 min wide), the ~10–20s alignment
      error from backward merge is negligible.

  The interpolation approach (C) looks appealing but introduces artifacts
  during stoppages: it would linearly interpolate game clock across a
  timeout, assigning game-clock values that never actually occurred on
  the scoreboard.
""".format(median=espn_gaps.median()))

# =====================================================================
# 5.  FINAL MERGED TIME SERIES (using backward merge)
# =====================================================================
print("=" * 70)
print("5.  FINAL MERGED TIME SERIES")
print("=" * 70)

final = merged_A.copy()
final.rename(columns={"ge_backward": "game_elapsed_seconds"}, inplace=True)
final["win_prob_pct"] = final["win_prob"] * 100.0
final["team_won"] = (final["result"] == "yes").astype(int)

# Add identifiers so this data can be stacked across many games
final["kalshi_event"] = KALSHI_EVENT
final["team"] = TEAM

# The columns that matter for calibration analysis (win probability focus)
OUTPUT_COLS = [
    "kalshi_event",          # which game
    "team",                  # which team's perspective
    "game_elapsed_seconds",  # game clock (0-2400)
    "period",                # 1 = 1st half, 2 = 2nd half
    "win_prob_pct",          # Kalshi quoted probability (0-100)
    "team_won",              # 1 if this team won, 0 if lost
]
output_df = final[OUTPUT_COLS].copy()
output_df["win_prob_pct"] = output_df["win_prob_pct"].round(2)

# Save to CSV
output_csv = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"game_data_{KALSHI_EVENT}_{TEAM}.csv"
)
output_df.to_csv(output_csv, index=False)
print(f"\n  Saved {len(output_df)} rows to {output_csv}")

print(f"\n  Rows: {len(final)}")
print(f"  Game elapsed range: {final['game_elapsed_seconds'].min():.0f} – "
      f"{final['game_elapsed_seconds'].max():.0f} s")
print(f"  Team {TEAM} won: {'YES' if final['team_won'].iloc[0] == 1 else 'NO'}")
print(f"\n  Output columns: {OUTPUT_COLS}")
print(f"\n  Head:\n{output_df.head(20).to_string()}")
print(f"\n  Tail:\n{output_df.tail(10).to_string()}")

# =====================================================================
# 6.  VISUALIZATION
# =====================================================================
print("\n" + "=" * 70)
print("6.  PLOTTING")
print("=" * 70)

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Figure 1: Wallclock → Game-clock mapping (ESPN) -----------------
fig1, ax1 = plt.subplots(figsize=(14, 6))
wc_min = (espn_raw["wallclock_ts"] - espn_raw["wallclock_ts"].iloc[0]).dt.total_seconds() / 60
ax1.plot(wc_min, espn_raw["game_elapsed_seconds"] / 60, ".", markersize=2, alpha=0.6)
ax1.set_xlabel("Wallclock time (min from tip-off)")
ax1.set_ylabel("Game clock (min elapsed)")
ax1.set_title("Wallclock → Game-Clock Mapping (from ESPN play-by-play)")
ax1.axhline(y=20, color="grey", ls="--", lw=0.8, label="Halftime (20 min)")
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
path1 = os.path.join(script_dir, f"plot1_wallclock_mapping_{KALSHI_EVENT}.png")
fig1.savefig(path1, dpi=150)
print(f"\n  Plot 1 saved → {path1}")

# --- Figure 2: Three alignment strategies compared -------------------
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(comparison["ge_backward"] / 60, comparison["win_prob"] * 100,
         "o-", markersize=3, alpha=0.6, label="A: backward merge")
ax2.plot(comparison["ge_nearest"] / 60, comparison["win_prob"] * 100,
         "x-", markersize=3, alpha=0.4, label="B: nearest merge")
ax2.plot(comparison["ge_interp"] / 60, comparison["win_prob"] * 100,
         "+-", markersize=3, alpha=0.4, label="C: interpolation")
ax2.axvline(x=20, color="grey", ls="--", lw=0.8, label="Halftime")
ax2.axhline(y=50, color="lightgrey", ls=":", lw=0.8)
ax2.set_xlabel("Game clock (min elapsed)")
ax2.set_ylabel("Win Probability (%)")
ax2.set_title(f"Three Alignment Strategies – {TEAM} Win Prob vs Game Clock")
ax2.set_ylim(0, 100)
ax2.legend(loc="best", fontsize=9)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
path2 = os.path.join(script_dir, f"plot2_strategy_comparison_{KALSHI_EVENT}.png")
fig2.savefig(path2, dpi=150)
print(f"  Plot 2 saved → {path2}")

# --- Figure 3: Final recommended time series (backward merge) --------
fig3, ax3 = plt.subplots(figsize=(14, 6))
df_plot = final.sort_values("game_elapsed_seconds")
ax3.plot(df_plot["game_elapsed_seconds"] / 60,
         df_plot["win_prob_pct"],
         color="#1f77b4", linewidth=1.5, alpha=0.85,
         label=f"{TEAM} Win Prob (close)")

# OHLC candle bodies
if "win_prob_open" in df_plot.columns and "win_prob_close" in df_plot.columns:
    for _, row in df_plot.iterrows():
        x = row["game_elapsed_seconds"] / 60
        o = row.get("win_prob_open")
        c = row.get("win_prob_close")
        if pd.isna(o) or pd.isna(c):
            continue
        colour = "#2ca02c" if c >= o else "#d62728"
        body_bottom = min(o, c) * 100
        body_height = abs(c - o) * 100 or 0.3
        ax3.bar(x, body_height, bottom=body_bottom, width=0.12,
                color=colour, edgecolor=colour, linewidth=0.5, alpha=0.7)

ax3.axvline(x=20, color="grey", ls="--", lw=0.8, label="Halftime")
ax3.axhline(y=50, color="lightgrey", ls=":", lw=0.8)
ax3.set_xlim(0, REG_SEC / 60)
ax3.set_ylim(0, 100)
ax3.set_xlabel("Game Clock (min elapsed from tip-off)")
ax3.set_ylabel("Win Probability (%)")
ax3.set_title(f"Final Aligned Time Series – {KALSHI_EVENT} – {TEAM}")

ax3.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax3.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax3.legend(loc="best", fontsize=10)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
path3 = os.path.join(script_dir, f"plot3_final_timeseries_{KALSHI_EVENT}_{TEAM}.png")
fig3.savefig(path3, dpi=150)
print(f"  Plot 3 saved → {path3}")

plt.show()

# =====================================================================
# 7.  DATA QUALITY ASSESSMENT
# =====================================================================
print("\n" + "=" * 70)
print("7.  DATA QUALITY ASSESSMENT")
print("=" * 70)

# Run comprehensive quality check
quality = assess_game_quality(final, kalshi_game_id=KALSHI_EVENT, team_abbr=TEAM)
print_quality_report(quality, kalshi_game_id=KALSHI_EVENT, team_abbr=TEAM)

print("\n" + "=" * 70)
print("DONE.  The merged DataFrame 'final' is your per-game time series")
print("with columns: game_elapsed_seconds, win_prob_pct, team_won, etc.")
if quality["is_valid"]:
    print(f"✓ This game PASSES quality checks and is suitable for calibration analysis.")
else:
    print(f"✗ This game FAILS quality checks: {', '.join(quality['rejection_reasons'])}")
print("=" * 70)
