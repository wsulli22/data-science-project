#!/usr/bin/env python3
"""
Visualize and explain the game_data CSV files.

This script reads a game_data CSV file (output from step2_explore.py) and creates
a comprehensive visualization showing:
- Kalshi win probability over game clock time
- Game stages (early/mid/late)
- Outcome (win/loss)

WHAT THE CSV REPRESENTS:
Each row = one 1-minute Kalshi candlestick aligned to game clock time.
- game_elapsed_seconds: Where in the game (0-2400 seconds = 40 minutes)
- win_prob_pct: Kalshi's quoted win probability at that moment (0-100%)
- team_won: Did this team actually win? (1 = yes, 0 = no)

This data is used for calibration analysis: comparing quoted probabilities
to actual outcomes across different game stages.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# Game stage boundaries (in seconds)
EARLY_STAGE = (0, 800)      # 0-13.3 min
MID_STAGE = (800, 1600)     # 13.3-26.7 min
LATE_STAGE = (1600, 2400)   # 26.7-40 min

REGULATION_SECONDS = 2400  # 40 minutes (2 x 20 min halves)


def visualize_game_data(csv_path: str, output_path: str | None = None) -> None:
    """
    Create a comprehensive visualization of the game data CSV.
    
    Args:
        csv_path: Path to the game_data CSV file
        output_path: Optional path to save the plot. If None, saves next to CSV.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract game info from filename or data
    kalshi_event = df["kalshi_event"].iloc[0]
    team = df["team"].iloc[0]
    team_won = df["team_won"].iloc[0] == 1
    
    # Sort by game time
    df = df.sort_values("game_elapsed_seconds").copy()
    
    # Convert seconds to minutes for plotting
    df["game_minutes"] = df["game_elapsed_seconds"] / 60.0
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])  # Main win probability plot
    ax2 = fig.add_subplot(gs[1])  # Text explanation
    
    # =====================================================================
    # PLOT 1: Win Probability Over Time
    # =====================================================================
    
    # Plot the main line (close price)
    ax1.plot(df["game_minutes"], df["win_prob_pct"],
             color="#1f77b4", linewidth=2.5, alpha=0.9,
             label=f"{team} Win Probability (Kalshi Close Price)",
             zorder=5)
    
    # Halftime line
    ax1.axvline(x=20, color="grey", linestyle="--", linewidth=1.5, label="Halftime")
    
    # 50% reference line
    ax1.axhline(y=50, color="lightgrey", linestyle=":", linewidth=1, alpha=0.7)
    
    # Outcome annotation
    outcome_text = "✓ WON" if team_won else "✗ LOST"
    outcome_color = "green" if team_won else "red"
    ax1.text(0.99, 0.02, outcome_text, transform=ax1.transAxes,
             fontsize=14, fontweight="bold", color=outcome_color,
             horizontalalignment="right", verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=outcome_color, linewidth=2))
    
    # Formatting
    ax1.set_xlim(0, REGULATION_SECONDS / 60)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Game Clock (minutes from tip-off)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Kalshi Win Probability (%)", fontsize=12, fontweight="bold")
    ax1.set_title(f"Kalshi Win Probability Over Time\n{kalshi_event} - {team}", 
                  fontsize=14, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)
    
    # Add statistics text box
    prob_range = df["win_prob_pct"].max() - df["win_prob_pct"].min()
    prob_min = df["win_prob_pct"].min()
    prob_max = df["win_prob_pct"].max()
    stats_text = f"Range: {prob_min:.1f}% - {prob_max:.1f}% ({prob_range:.1f}% span)\n"
    stats_text += f"Observations: {len(df)} candles\n"
    stats_text += f"Game coverage: {df['game_elapsed_seconds'].max()/60:.1f} min"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # =====================================================================
    # PLOT 2: Explanation Text
    # =====================================================================
    
    explanation = (
        "WHAT THIS DATA REPRESENTS:\n"
        "• Each point = one 1-minute Kalshi candlestick aligned to game clock time\n"
        "• win_prob_pct = Kalshi's quoted win probability (close price of the candle)\n"
        "• game_elapsed_seconds = position in game (0-2400s = 40 min regulation)\n"
        "• team_won = actual outcome (1 = this team won, 0 = lost)\n"
        "• This data is used for calibration: comparing quoted probabilities to actual win rates"
    )
    
    ax2.axis("off")
    ax2.text(0.5, 0.5, explanation, transform=ax2.transAxes,
             fontsize=10, verticalalignment="center", horizontalalignment="center",
             bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
    
    # Save the figure
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(os.path.dirname(csv_path), f"{base_name}_visualization.png")
    
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Visualization saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: use the example file
        default_csv = "game_data_KXNCAAMBGAME-26FEB10MILWIUIN_MILW.csv"
        if os.path.exists(default_csv):
            print(f"Using default file: {default_csv}")
            visualize_game_data(default_csv)
        else:
            print("Usage: python visualize_game_data.py <path_to_game_data.csv>")
            print(f"Or place a file named '{default_csv}' in the current directory.")
            sys.exit(1)
    else:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"Error: File not found: {csv_path}")
            sys.exit(1)
        visualize_game_data(csv_path)
