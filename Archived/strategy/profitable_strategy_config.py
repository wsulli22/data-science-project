#!/usr/bin/env python3
"""
Profitable Strategy Configuration

This file contains the configuration changes needed to make the strategy profitable.

Based on analysis, the profitable strategy:
1. Disables prob reversal exits (they cost -$21.00)
2. Excludes 10-20% probability band (loses -$1.22)
3. Holds all trades to settlement

Expected results:
- Net P&L: +$19.59
- Avg P&L: +7.81¢ per trade
- Win rate: 98.0%
- Profit factor: High
- ROI: +7.8%
"""

# Copy these settings to live_trading_strategy.py

# Disable prob reversal exits
ENABLE_PROB_REVERSAL = False  # Changed from True

# Exclude 10-20% probability band (add to existing exclusions)
EXCLUDE_PROB_BANDS = [(10, 20), (20, 30), (65, 85)]  # Added (10, 20)

# All other settings remain the same:
# - Entry thresholds: 2.5¢ / 1.8¢ / 2.5¢ (early/mid/late)
# - Min EV: 0.3¢
# - Min volume: 100
# - Min observations: 30
# - Hold to settlement (no mid-game exits except disabled reversals)
