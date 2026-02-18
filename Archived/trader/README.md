# Kalshi NCAAB Trading Strategy Backtester

A comprehensive backtesting system for trading Kalshi NCAAB win markets using YES positions only.

## Overview

This system implements:
- **Probabilistic modeling** to estimate true win probability from Kalshi market data
- **Fee-aware trade selection** with realistic execution assumptions
- **Time-delay robustness** to handle live trading constraints (up to 120 second delays)
- **Comprehensive reporting** with visualizations and parameter sensitivity analysis

## Key Features

### Trading Constraints
- YES-only positions (no NO trades)
- Maximum 1 entry per game
- Hold to settlement (no cash-out)
- Marketable fills at observed YES price
- Fee calculation: `ceil(0.07 * C * P * (1 - P) * 100) / 100` where C = contracts, P = yes_price

### Model Features
- Kalshi quoted probability (binned)
- Time in game (2-minute bins for time-delay robustness)
- Period (1st half, 2nd half)
- Volume (log-transformed)
- Recent probability changes (5 and 10 observation lookback)

### Time-Delay Handling
The model uses 2-minute time bins to ensure signals remain valid under a +0 to +120 second time delay uncertainty. This makes the strategy robust to ESPN polling delays.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the main script to generate a comprehensive backtest report:

```bash
cd trader
python main.py
```

This will:
1. Load data from `../GeneratedDataFiles/all_games_merged_clean_GOOD.csv`
2. Split into train/test sets (80/20 by chronological game order)
3. Run backtests on both sets
4. Generate visualizations
5. Run parameter sweeps
6. Create ablation studies

### Custom Backtest

```python
from backtest import (
    split_train_test_by_game_id_chronological,
    run_backtest,
    summarize_results
)
import pandas as pd

# Load data
df = pd.read_csv("GeneratedDataFiles/all_games_merged_clean_GOOD.csv")

# Split train/test
train_df, test_df = split_train_test_by_game_id_chronological(df, train_pct=0.8)

# Run backtest
results = run_backtest(
    train_df, test_df,
    bet_size_dollars=50.0,
    ev_threshold=0.0,
    volume_min=0.0,
    policy='first'  # or 'max_ev'
)

# Summarize
summarize_results(results, "Test")
```

## Parameters

### Trading Parameters
- `bet_size_dollars`: Dollar amount per trade (typically $50-$100)
- `ev_threshold`: Minimum expected value per contract to enter (e.g., 0.0, 0.003, 0.005)
- `volume_min`: Minimum volume filter (e.g., 0, 100, 500)
- `policy`: Trade selection policy
  - `'first'`: First valid signal in the game
  - `'max_ev'`: Maximum EV signal in the game

### Model Parameters
The model uses isotonic regression for calibration with logistic regression base. Features include:
- `win_prob_pct`: Kalshi quoted probability (1-99)
- `time_bin`: 2-minute time bins (0-20)
- `period`: Game period (1 or 2)
- `log_volume`: Log-transformed volume
- `prob_change_5`: Change in probability over last 5 observations
- `prob_change_10`: Change in probability over last 10 observations
- `prob_bin`: Probability binned into 10% buckets

## Output Files

All outputs are saved to `trader/reports/`:

- `parameter_sweep_results.csv`: Results for all parameter combinations
- `train_cumulative_pnl.png`: Training set cumulative P&L curve
- `test_cumulative_pnl.png`: Test set cumulative P&L curve
- `train_pnl_distribution.png`: Training set P&L distribution
- `test_pnl_distribution.png`: Test set P&L distribution
- `train_trade_breakdown.png`: Training set trade breakdown by time/probability
- `test_trade_breakdown.png`: Test set trade breakdown by time/probability
- `parameter_sensitivity.png`: Sensitivity analysis plots

## Performance Metrics

The system reports:
- Number of games and trades
- Trade rate (trades per game)
- Total P&L and average P&L per trade
- Win rate
- Median, 10th, and 90th percentile P&L
- Cumulative P&L curve
- Maximum drawdown
- Trade breakdown by time bin and probability bin

## Data Requirements

Input CSV should have columns:
- `kalshi_event`: Unique game identifier (e.g., KXNCAAMBGAME-26FEB14SMCPAC)
- `team`: Team identifier
- `game_elapsed_seconds`: Time in game (0-2400)
- `period`: Period number (1 or 2)
- `win_prob_pct`: Kalshi quoted win probability (1-99)
- `volume`: Trading volume (liquidity proxy)
- `team_won`: Final outcome (0 or 1)

## Notes

- The system automatically derives `yes_price` from `win_prob_pct` (dividing by 100)
- Games are ordered chronologically by parsing dates from `kalshi_event` strings
- If date parsing fails, falls back to ordering by earliest `game_elapsed_seconds`
- All feature engineering uses only past data to prevent lookahead bias
- The model is trained only on training games, then applied to test games

## License

This code is provided as-is for research and educational purposes.
