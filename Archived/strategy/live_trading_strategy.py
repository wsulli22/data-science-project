#!/usr/bin/env python3
"""
live_trading_strategy.py

Production-ready trading strategy for Kalshi NCAAB basketball markets.

Supports two modes:
  RESTRICTED  — one buy per game (resets if safety-sold)
  UNRESTRICTED — buy and sell freely, multiple round-trips per game

Core Design Principles
──────────────────────
1. Uses GAM-calibrated true probabilities vs. Kalshi market price
2. Accounts for ±60 s ESPN clock uncertainty (conservative edge)
3. Multi-layer exit logic (model reversal, prob reversal, profit-take)
4. Full historical backtesting with realistic fee modeling
5. Not required to trade every game — only trade when edge exists

Usage:
    python live_trading_strategy.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — all tunable parameters in one place
# ═══════════════════════════════════════════════════════════════════════════════
INPUT_CSV       = "GeneratedDataFiles/all_games_merged_clean_GOOD.csv"
CALIBRATION_CSV = "GeneratedDataFiles/calibration_heatmap_data.csv"
OUTPUT_DIR      = "GeneratedDataFiles"

# ── STRATEGY MODE ─────────────────────────────────────────────────────────────
# "restricted"   = original: one buy per game, resets on safety sell
# "unrestricted" = new: buy and sell freely, multiple round-trips per game
STRATEGY_MODE = "restricted"   # "restricted" or "unrestricted"  (see comparison below)

# Clock uncertainty
CLOCK_UNCERTAINTY_SEC = 60    # ESPN clock can be ±60 s off true game time
CLOCK_EVAL_POINTS     = 5     # evaluate calibration at this many points in window

# Fee model
# Kalshi charges trading fees on EVERY trade (buy or sell), NOT at settlement
# Fees are rounded up on the TOTAL fee, not per contract
# 
# Taker fees (immediately matched orders): round_up(0.07 × C × P × (1 − P))
# Maker fees (resting limit orders, when applicable): round_up(0.0175 × C × P × (1 − P))
#   where C = total contracts, P = contract price in dollars
#
# For sports markets, maker fees are often available, but we'll default to taker for simplicity
# User should specify which they plan to use
USE_MAKER_FEES = False  # Set to True if using resting limit orders (maker fees)
TAKER_FEE_RATE = 0.07      # 7% of P × (1-P) for taker orders
MAKER_FEE_RATE = 0.0175    # 1.75% of P × (1-P) for maker orders (when available)
SAFETY_SELL_COST = 0.01     # ~1¢ spread/slippage cost when selling mid-game (market impact)

# Entry thresholds
MIN_EDGE_EARLY     = 0.025   # min |edge| for first 18 min (wait for better)
MIN_EDGE_MID       = 0.018   # min |edge| for 18-30 min
MIN_EDGE_LATE      = 0.025   # min |edge| for 30-40 min
MIN_EV_AFTER_FEES  = 0.003   # min EV per $1 after fees (0.3¢) - NOTE: computed at DEFAULT_TRADE_SIZE
MIN_VOLUME         = 100     # minimum Kalshi volume for liquidity
MIN_HIST_OBS       = 30      # min historical observations in the (time,prob) cell
DEFAULT_TRADE_SIZE = 1       # default number of contracts per trade (affects EV calculation due to fee rounding)

# Probability filters
MAX_PROB_FOR_NO    = 45      # only buy NO when Kalshi prob ≤ 45%
MIN_PROB_FOR_YES   = 55      # only buy YES when Kalshi prob ≥ 55%
EXCLUDE_PROB_BANDS = [(20, 30), (65, 85)]  # probability ranges to avoid

# ── RESTRICTED-mode safety-sell parameters ────────────────────────────────────
ENABLE_SAFETY_SELL   = True
STOP_LOSS_CENTS      = 0.99  # disabled — basketball too volatile for price stops
MODEL_EXIT_BUFFER    = 0.99  # disabled in restricted mode
TRAILING_STOP_FRAC   = 0.99  # disabled
MIN_PROFIT_FOR_TRAIL = 0.99  # disabled
MAX_SAFETY_SELLS     = 1     # max 1 safety sell per game (restricted mode only)

# Probability-reversal exit (active in BOTH modes)
ENABLE_PROB_REVERSAL    = True
REVERSAL_PROB_THRESHOLD = 80  # sell NO only if team reaches ≥80% (truly dominating)
REVERSAL_MIN_TIME_SEC   = 2100  # only in final 5 min (30+ min elapsed)

# ── UNRESTRICTED-mode parameters ──────────────────────────────────────────────
# Active exit: sell when model says holding is now worse than exiting
# Buffer = round-trip cost ($0.01 spread + $0.005 slippage) → sell only when
# model is CONFIDENT holding is bad (otherwise hold to settlement)
UNRES_MODEL_EXIT_BUFFER = 0.02    # sell if hold_EV < sell_val by ≥ 2¢
# Profit-take: sell if unrealised gain ≥ threshold AND edge has disappeared
UNRES_PROFIT_TAKE_MIN   = 0.05   # min 5¢ unrealised gain to consider profit-take
UNRES_EDGE_GONE_THRESH  = 0.002  # edge must be < 0.2¢ to trigger profit-take
# Cooldown: wait N ticks after selling before re-entering (avoid churn)
UNRES_REENTRY_COOLDOWN  = 1      # skip 1 tick (~25s) after selling before re-buying

# Execution
SLIPPAGE_CENTS = 0.005       # 0.5¢ slippage per trade (half-penny)

# Game constants
REGULATION_SEC = 2400        # 40 min × 60 s


# ═══════════════════════════════════════════════════════════════════════════════
#  FEE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class FeeModel:
    """
    Kalshi fee calculator for NCAAB game contracts.

    Kalshi charges trading fees on EVERY trade (buy or sell), NOT at settlement.
    
    IMPORTANT: Fees are rounded up on the TOTAL fee, not per contract.
    - 1 contract at $0.95: fee = round_up(0.07 × 1 × 0.95 × 0.05) = $0.01
    - 100 contracts at $0.95: fee = round_up(0.07 × 100 × 0.95 × 0.05) = $0.34 total (0.34¢ per contract)
    
    Taker fees (immediately matched): round_up(0.07 × C × P × (1 − P))
    Maker fees (resting limit orders, when available): round_up(0.0175 × C × P × (1 − P))
    """

    def __init__(self, use_maker=USE_MAKER_FEES,
                 taker_fee_rate=TAKER_FEE_RATE,
                 maker_fee_rate=MAKER_FEE_RATE,
                 safety_sell_cost=SAFETY_SELL_COST):
        self.use_maker = use_maker
        self.taker_fee_rate = taker_fee_rate   # 0.07 = 7%
        self.maker_fee_rate = maker_fee_rate   # 0.0175 = 1.75%
        self.safety_sell_cost = safety_sell_cost   # spread/slippage

    def trading_fee(self, contract_price: float, n_contracts: int = 1, is_maker: bool = None) -> float:
        """
        Trading fee for a trade (buy or sell).
        
        Formula: round_up(rate × C × P × (1 − P))
        where C = total contracts, P = contract price
        
        IMPORTANT: Rounding is on TOTAL fee, not per contract.
        This means per-contract fee decreases as you scale up.
        
        Args:
            contract_price: Price per contract (0.0 to 1.0)
            n_contracts: Number of contracts (default 1)
            is_maker: If True, use maker fee; if False, use taker; if None, use self.use_maker
        """
        if is_maker is None:
            is_maker = self.use_maker
        
        rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        raw_fee = rate * n_contracts * contract_price * (1 - contract_price)
        # Round up to next cent on TOTAL fee
        return np.ceil(raw_fee * 100) / 100.0

    def entry_fee(self, contract_price: float, n_contracts: int = 1, is_maker: bool = None) -> float:
        """Trading fee paid at entry (alias for trading_fee for backward compatibility)."""
        return self.trading_fee(contract_price, n_contracts, is_maker)

    def exit_fee(self, contract_price: float, n_contracts: int = 1, is_maker: bool = None) -> float:
        """Trading fee paid when selling mid-game (same formula as entry)."""
        return self.trading_fee(contract_price, n_contracts, is_maker)

    def settlement_fee(self, profit: float) -> float:
        """No settlement fee — fees are paid at entry."""
        return 0.0

    def safety_sell_fee(self) -> float:
        """
        Spread/slippage cost when selling mid-game.
        Note: This is NOT a special Kalshi fee - it's just market impact (bid-ask spread).
        The actual trading fee on the sell is calculated separately using exit_fee(exit_price).
        """
        return self.safety_sell_cost

    def ev_yes(self, p_true: float, p_kalshi: float, n_contracts: int = 1) -> float:
        """
        Expected value of a YES bet PER CONTRACT.
        
        Args:
            p_true: True win probability (from GAM model)
            p_kalshi: Kalshi quoted win probability
            n_contracts: Number of contracts (affects fee calculation due to rounding)
        """
        entry_price = p_kalshi
        entry_fee_total = self.trading_fee(entry_price, n_contracts=n_contracts)
        entry_fee_per_contract = entry_fee_total / n_contracts
        total_cost_per_contract = entry_price + entry_fee_per_contract
        
        # If win: get $1.00 per contract, net = 1.00 - total_cost_per_contract
        # If lose: get $0.00, net = -total_cost_per_contract
        return p_true * (1.0 - total_cost_per_contract) - (1 - p_true) * total_cost_per_contract

    def ev_no(self, p_true: float, p_kalshi: float, n_contracts: int = 1) -> float:
        """
        Expected value of a NO bet PER CONTRACT.
        
        Args:
            p_true: True win probability (from GAM model)
            p_kalshi: Kalshi quoted win probability
            n_contracts: Number of contracts (affects fee calculation due to rounding)
        """
        entry_price = 1.0 - p_kalshi
        entry_fee_total = self.trading_fee(entry_price, n_contracts=n_contracts)
        entry_fee_per_contract = entry_fee_total / n_contracts
        total_cost_per_contract = entry_price + entry_fee_per_contract
        
        # If win (underdog loses): get $1.00 per contract, net = 1.00 - total_cost_per_contract
        # If lose (underdog wins): get $0.00, net = -total_cost_per_contract
        return (1 - p_true) * (1.0 - total_cost_per_contract) - p_true * total_cost_per_contract

    def __repr__(self):
        fee_type = "maker" if self.use_maker else "taker"
        rate = self.maker_fee_rate if self.use_maker else self.taker_fee_rate
        return (f"FeeModel({fee_type}_fee={rate*100:.2f}% of C×P×(1-P), "
                f"safety_sell=${self.safety_sell_cost:.2f})")


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION MODEL  — loads GAM heatmap, provides interpolated lookup
# ═══════════════════════════════════════════════════════════════════════════════
class CalibrationModel:
    """
    Wraps the GAM-calibrated probability surface.
    Provides interpolated lookup at any (kalshi_prob_pct, elapsed_seconds).
    """

    def __init__(self, csv_path=CALIBRATION_CSV, raw_data_csv=INPUT_CSV):
        # ── load calibration surface ──────────────────────────────────────
        df = pd.read_csv(csv_path, index_col=0)
        self.prob_grid = df.index.values.astype(float)         # 1 … 99
        n_bins = len(df.columns)
        bin_width_sec = REGULATION_SEC / n_bins                # 120 s
        self.time_grid = np.array(
            [bin_width_sec * (i + 0.5) for i in range(n_bins)]
        )  # bin centres in seconds
        self.values = df.values                                # (99, 20)

        self._interp = RegularGridInterpolator(
            (self.prob_grid, self.time_grid),
            self.values,
            method="linear",
            bounds_error=False,
            fill_value=None,     # nearest-neighbour extrapolation at edges
        )

        # ── load raw observation counts for frequency / reliability ───────
        self.raw_counts = None
        self.freq_per_game = None
        if raw_data_csv and os.path.exists(raw_data_csv):
            raw = pd.read_csv(raw_data_csv)
            n_games = raw["kalshi_event"].nunique()
            time_edges = np.linspace(0, REGULATION_SEC, n_bins + 1)
            time_labels = df.columns.tolist()
            raw = raw.dropna(subset=["win_prob_pct"])
            raw["prob_int"] = raw["win_prob_pct"].round(0).astype(int)
            raw = raw[raw["prob_int"].between(1, 99)]
            raw["time_bin"] = pd.cut(
                raw["game_elapsed_seconds"],
                bins=time_edges, labels=time_labels,
                right=False, include_lowest=True,
            )
            counts = (
                raw.groupby(["prob_int", "time_bin"], observed=False)
                .size()
                .reset_index(name="n")
            )
            self.raw_counts = counts.pivot(
                index="prob_int", columns="time_bin", values="n"
            ).fillna(0)
            self.freq_per_game = self.raw_counts / n_games
            self.n_games = n_games

    # ── point query ───────────────────────────────────────────────────────
    def query(self, kalshi_prob_pct: float, elapsed_sec: float) -> float:
        """GAM-calibrated true probability at a single (prob, time) point."""
        prob = np.clip(kalshi_prob_pct, 1, 99)
        time = np.clip(elapsed_sec, self.time_grid[0], self.time_grid[-1])
        return float(self._interp((prob, time)))

    # ── clock-uncertainty query ───────────────────────────────────────────
    def query_robust(
        self,
        kalshi_prob_pct: float,
        espn_elapsed_sec: float,
        uncertainty: float = CLOCK_UNCERTAINTY_SEC,
        n_points: int = CLOCK_EVAL_POINTS,
    ) -> Dict:
        """
        Evaluate calibrated probability across the ±uncertainty window.
        Returns dict with conservative (worst-case) and mean estimates.
        """
        t_lo = max(0, espn_elapsed_sec - uncertainty)
        t_hi = min(REGULATION_SEC, espn_elapsed_sec + uncertainty)
        times = np.linspace(t_lo, t_hi, n_points)
        cal_probs = [self.query(kalshi_prob_pct, t) for t in times]

        return {
            "cal_min":  min(cal_probs),
            "cal_max":  max(cal_probs),
            "cal_mean": float(np.mean(cal_probs)),
            "cal_at_times": list(zip(times.tolist(), cal_probs)),
        }

    # ── cell observation count ────────────────────────────────────────────
    def get_obs_count(self, kalshi_prob_pct: float, elapsed_sec: float) -> int:
        """Number of historical observations in the matching (prob, time) cell."""
        if self.raw_counts is None:
            return 9999  # no data → don't filter
        prob_int = int(round(np.clip(kalshi_prob_pct, 1, 99)))
        # find matching time bin
        n_bins = len(self.time_grid)
        bin_width = REGULATION_SEC / n_bins
        bin_idx = int(np.clip(elapsed_sec // bin_width, 0, n_bins - 1))
        col = self.raw_counts.columns[bin_idx]
        if prob_int in self.raw_counts.index:
            return int(self.raw_counts.loc[prob_int, col])
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  POSITION  — tracks a single open position
# ═══════════════════════════════════════════════════════════════════════════════
class Direction(Enum):
    YES = "YES"
    NO  = "NO"


@dataclass
class Position:
    direction:     Direction
    entry_price:   float           # contract price at entry (YES price or 1-YES for NO)
    entry_time:    float           # game_elapsed_seconds
    entry_prob:    float           # Kalshi win_prob_pct at entry
    entry_fee:     float           # fee paid at entry
    team:          str             # team we're betting on/against
    peak_value:    float = 0.0     # highest contract value seen since entry
    n_ticks_held:  int   = 0

    @property
    def cost(self) -> float:
        """Capital tied up."""
        return self.entry_price

    def current_value(self, kalshi_prob_pct: float) -> float:
        """Current mark-to-market value of the contract."""
        if self.direction == Direction.YES:
            return kalshi_prob_pct / 100.0
        else:
            return 1.0 - kalshi_prob_pct / 100.0

    def unrealised_pnl(self, kalshi_prob_pct: float) -> float:
        """Unrealised P&L (before exit fee)."""
        return self.current_value(kalshi_prob_pct) - self.entry_price - self.entry_fee


# ═══════════════════════════════════════════════════════════════════════════════
#  TRADE RECORD
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class TradeRecord:
    event_id:        str
    team:            str
    direction:       str        # "YES" or "NO"
    entry_price:     float
    entry_time:      float
    entry_prob:      float      # Kalshi prob at entry
    entry_fee:       float
    exit_price:      float      # settlement value or sell price
    exit_time:       float
    exit_prob:       float      # Kalshi prob at exit
    exit_fee:        float      # 0 if held to settlement
    exit_type:       str        # "settlement_win", "settlement_loss", "safety_sell"
    gross_pnl:       float
    net_pnl:         float
    hold_ticks:      int
    edge_at_entry:   float
    cal_prob_entry:  float
    safety_sell_count: int      # how many times we safety-sold in this game


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL EVALUATION  — entry & exit logic
# ═══════════════════════════════════════════════════════════════════════════════
def get_min_edge(elapsed_sec: float) -> float:
    """Dynamic edge threshold: higher early (we can wait), lower late (last chance)."""
    minutes = elapsed_sec / 60.0
    if minutes < 18:
        return MIN_EDGE_EARLY
    elif minutes < 30:
        return MIN_EDGE_MID
    else:
        return MIN_EDGE_LATE


def evaluate_entry(
    kalshi_prob_pct: float,
    elapsed_sec: float,
    volume: float,
    cal_model: CalibrationModel,
    fee_model: FeeModel,
    n_contracts: int = DEFAULT_TRADE_SIZE,
) -> Optional[Dict]:
    """
    Evaluate whether a trade should be entered at this (prob, time) point.

    Returns None if no trade, or a dict with trade details.
    Uses robust (conservative) calibrated probability across the ±60s
    clock-uncertainty window to ensure edge is real.
    """
    p = kalshi_prob_pct / 100.0

    # ── liquidity filter ──────────────────────────────────────────────────
    if volume < MIN_VOLUME:
        return None

    # ── probability band filter ───────────────────────────────────────────
    # Exclude mid-range probabilities where edge is weakest
    for lo, hi in EXCLUDE_PROB_BANDS:
        if lo <= kalshi_prob_pct <= hi:
            return None

    # ── reliability filter ────────────────────────────────────────────────
    obs = cal_model.get_obs_count(kalshi_prob_pct, elapsed_sec)
    if obs < MIN_HIST_OBS:
        return None

    # ── robust calibration across clock uncertainty ───────────────────────
    robust = cal_model.query_robust(kalshi_prob_pct, elapsed_sec)

    # Conservative: use the WORST-CASE calibrated prob within ±60s window
    #   For YES: use minimum → smallest edge
    #   For NO:  use maximum → smallest edge
    cal_for_yes = robust["cal_min"]
    cal_for_no  = robust["cal_max"]

    # ── compute EV using fee model (trading fees, computed at trade size) ────────────────
    ev_yes = fee_model.ev_yes(cal_for_yes, p, n_contracts=n_contracts)
    ev_no  = fee_model.ev_no(cal_for_no, p, n_contracts=n_contracts)

    edge_yes = cal_for_yes - p       # raw edge before fees
    edge_no  = p - cal_for_no        # raw edge for NO direction

    # ── pick best direction (with probability constraints) ──────────────
    yes_ok = ev_yes > 0 and kalshi_prob_pct >= MIN_PROB_FOR_YES
    no_ok  = ev_no > 0  and kalshi_prob_pct <= MAX_PROB_FOR_NO

    if yes_ok and (not no_ok or ev_yes >= ev_no):
        direction = Direction.YES
        edge      = edge_yes
        ev        = ev_yes
        cal_prob  = cal_for_yes
    elif no_ok:
        direction = Direction.NO
        edge      = edge_no
        ev        = ev_no
        cal_prob  = cal_for_no
    else:
        return None  # no viable direction

    # ── threshold checks ──────────────────────────────────────────────────
    min_edge = get_min_edge(elapsed_sec)
    if abs(edge) < min_edge:
        return None
    if ev < MIN_EV_AFTER_FEES:
        return None

    # ── compute entry price (with slippage) ───────────────────────────────
    if direction == Direction.YES:
        entry_price = p + SLIPPAGE_CENTS
    else:
        entry_price = (1 - p) + SLIPPAGE_CENTS

    return {
        "direction":   direction,
        "edge":        edge,
        "ev":          ev,
        "entry_price": entry_price,
        "cal_prob":    cal_prob,
        "obs_count":   obs,
    }



# ═══════════════════════════════════════════════════════════════════════════════
#  GAME TRADER  — manages trading for one game event
# ═══════════════════════════════════════════════════════════════════════════════
class GameTrader:
    """
    Manages trading for a single Kalshi game event.

    RESTRICTED mode:
      - One buy per game (resets after safety sell, limited by MAX_SAFETY_SELLS)
    UNRESTRICTED mode:
      - Buy and sell freely; multiple round-trips per game
      - Active exits: model-based, profit-take, probability reversal
      - Re-entry after cooldown period
    """

    def __init__(self, event_id: str, cal_model: CalibrationModel,
                 fee_model: FeeModel, mode: str = STRATEGY_MODE):
        self.event_id       = event_id
        self.cal_model      = cal_model
        self.fee_model      = fee_model
        self.mode           = mode
        self.position       = None           # current open position (or None)
        self.can_buy        = True
        self.trades: List[TradeRecord] = []
        self.safety_sell_count = 0
        self.ticks_since_exit  = 999         # cooldown counter (unrestricted)

    def process_tick(
        self,
        team: str,
        kalshi_prob_pct: float,
        elapsed_sec: float,
        volume: float,
    ) -> Optional[str]:
        """
        Process one market data tick.
        Returns action taken: "buy", "sell", or None.
        """
        # ── count ticks since last exit (for cooldown) ────────────────────
        if self.position is None:
            self.ticks_since_exit += 1

        # ── if holding, check exit conditions ─────────────────────────────
        if self.position is not None:
            cur_val = self.position.current_value(kalshi_prob_pct)
            self.position.peak_value = max(self.position.peak_value, cur_val)
            self.position.n_ticks_held += 1

            # Only evaluate exits for the SAME team we're holding
            if team == self.position.team:
                should_sell, reason = self._evaluate_exit(
                    kalshi_prob_pct, elapsed_sec
                )
                if should_sell:
                    self._execute_sell(kalshi_prob_pct, elapsed_sec, reason)
                    return "sell"

            return None  # holding, no action

        # ── try to enter ──────────────────────────────────────────────────
        if not self.can_buy:
            return None

        # Cooldown: skip a few ticks after selling (unrestricted only)
        if self.mode == "unrestricted" and self.ticks_since_exit < UNRES_REENTRY_COOLDOWN:
            return None

        signal = evaluate_entry(
            kalshi_prob_pct, elapsed_sec, volume,
            self.cal_model, self.fee_model,
        )
        if signal is not None:
            self._execute_buy(team, kalshi_prob_pct, elapsed_sec, signal)
            return "buy"

        return None

    # ── EXIT EVALUATION ───────────────────────────────────────────────────
    def _evaluate_exit(self, kalshi_prob_pct, elapsed_sec):
        """
        Evaluate whether to exit the current position.
        Combines all exit conditions.
        Returns (should_sell, reason).
        """
        pos = self.position

        # ── 0. Probability-reversal (active in BOTH modes) ───────────────
        if ENABLE_PROB_REVERSAL and elapsed_sec >= REVERSAL_MIN_TIME_SEC:
            if pos.direction == Direction.NO:
                if kalshi_prob_pct >= REVERSAL_PROB_THRESHOLD:
                    return True, (f"prob_reversal (entry={pos.entry_prob:.0f}% "
                                  f"→ now={kalshi_prob_pct:.0f}%)")
            elif pos.direction == Direction.YES:
                if kalshi_prob_pct <= (100 - REVERSAL_PROB_THRESHOLD):
                    return True, (f"prob_reversal (entry={pos.entry_prob:.0f}% "
                                  f"→ now={kalshi_prob_pct:.0f}%)")

        if self.mode == "unrestricted":
            return self._evaluate_unrestricted_exit(kalshi_prob_pct, elapsed_sec)
        else:
            return self._evaluate_restricted_exit(kalshi_prob_pct, elapsed_sec)

    def _evaluate_restricted_exit(self, kalshi_prob_pct, elapsed_sec):
        """Original conservative exit logic (restricted mode)."""
        if not ENABLE_SAFETY_SELL:
            return False, ""

        pos = self.position
        current_val = pos.current_value(kalshi_prob_pct)

        # 1. Hard stop-loss
        loss = pos.entry_price - current_val
        if loss >= STOP_LOSS_CENTS:
            return True, f"stop_loss (lost {loss:.3f})"

        # 2. Model-based exit
        robust = self.cal_model.query_robust(kalshi_prob_pct, elapsed_sec)
        if pos.direction == Direction.YES:
            p_win = robust["cal_min"]
            hold_profit = (1 - pos.entry_price)
        else:
            p_win = 1.0 - robust["cal_max"]
            hold_profit = pos.entry_price

        # Hold EV: fees already paid at entry, so no additional fees at settlement
        total_entry_cost = pos.entry_price + pos.entry_fee
        hold_ev = p_win * (1.0 - total_entry_cost) - (1 - p_win) * total_entry_cost
        # Sell value: get current market value, pay trading fee on sell + spread/slippage
        # Note: Exit fee depends on exit price - can be higher at mid prices (e.g., $0.50)
        exit_price = current_val - SLIPPAGE_CENTS
        exit_trading_fee = self.fee_model.exit_fee(exit_price, n_contracts=1)
        sell_val = (exit_price - exit_trading_fee - self.fee_model.safety_sell_cost
                    - total_entry_cost)

        if sell_val > hold_ev + MODEL_EXIT_BUFFER:
            return True, f"model_exit (hold_ev={hold_ev:.3f}, sell_val={sell_val:.3f})"

        # 3. Trailing stop
        unrealised_gain = current_val - pos.entry_price
        peak_gain       = pos.peak_value - pos.entry_price
        if peak_gain >= MIN_PROFIT_FOR_TRAIL:
            retrace = peak_gain - unrealised_gain
            if retrace >= TRAILING_STOP_FRAC * peak_gain:
                return True, f"trailing_stop (peak={peak_gain:.3f}, retrace={retrace:.3f})"

        return False, ""

    def _evaluate_unrestricted_exit(self, kalshi_prob_pct, elapsed_sec):
        """
        Active exit logic for unrestricted mode.

        Two smart exits:
          1. Model-based: hold_EV < sell_value by ≥ UNRES_MODEL_EXIT_BUFFER
             (model says continuing to hold is worse than selling)
          2. Profit-take: if sitting on ≥3¢ unrealised gain AND edge has
             disappeared → sell to lock in profit, scan for re-entry
        """
        pos = self.position
        current_val = pos.current_value(kalshi_prob_pct)
        p = kalshi_prob_pct / 100.0

        # ── 1. Model-based exit ──────────────────────────────────────────
        robust = self.cal_model.query_robust(kalshi_prob_pct, elapsed_sec)
        if pos.direction == Direction.YES:
            p_win = robust["cal_min"]
            hold_profit = (1 - pos.entry_price)
        else:
            p_win = 1.0 - robust["cal_max"]
            hold_profit = pos.entry_price

        # Hold EV: fees already paid at entry, so no additional fees at settlement
        total_entry_cost = pos.entry_price + pos.entry_fee
        hold_ev = p_win * (1.0 - total_entry_cost) - (1 - p_win) * total_entry_cost
        # Sell value: get current market value, pay trading fee on sell + spread/slippage
        # Note: Exit fee depends on exit price - can be higher at mid prices (e.g., $0.50)
        exit_price = current_val - SLIPPAGE_CENTS
        exit_trading_fee = self.fee_model.exit_fee(exit_price, n_contracts=1)
        sell_val = (exit_price - exit_trading_fee - self.fee_model.safety_sell_cost
                    - total_entry_cost)

        if sell_val > hold_ev + UNRES_MODEL_EXIT_BUFFER:
            return True, (f"model_exit (hold_ev={hold_ev:.3f}, "
                          f"sell_val={sell_val:.3f})")

        # ── 2. Profit-take when edge has disappeared ─────────────────────
        unrealised_gain = current_val - pos.entry_price - SLIPPAGE_CENTS
        if unrealised_gain >= UNRES_PROFIT_TAKE_MIN:
            # Check if the original edge has disappeared
            if pos.direction == Direction.YES:
                cal_for_yes = robust["cal_min"]
                current_edge = cal_for_yes - p
            else:
                cal_for_no = robust["cal_max"]
                current_edge = p - cal_for_no

            if current_edge < UNRES_EDGE_GONE_THRESH:
                return True, (f"profit_take (gain={unrealised_gain:.3f}, "
                              f"edge={current_edge:.4f})")

        return False, ""

    # ── EXECUTE BUY ───────────────────────────────────────────────────────
    def _execute_buy(self, team, kalshi_prob_pct, elapsed_sec, signal):
        """Open a new position."""
        entry_price = signal["entry_price"]
        entry_fee = self.fee_model.entry_fee(entry_price, n_contracts=1)
        self.position = Position(
            direction   = signal["direction"],
            entry_price = entry_price,
            entry_time  = elapsed_sec,
            entry_prob  = kalshi_prob_pct,
            entry_fee   = entry_fee,
            team        = team,
            peak_value  = entry_price,
        )
        if self.mode == "restricted":
            self.can_buy = False  # used our 1 buy (restricted mode)
        # In unrestricted mode, can_buy stays True but we can't buy while
        # holding (checked via self.position is not None above)

    # ── EXECUTE SELL (mid-game) ───────────────────────────────────────────
    def _execute_sell(self, kalshi_prob_pct, elapsed_sec, reason):
        """
        Close position mid-game, record trade, manage re-entry permission.
        
        When selling mid-game, you pay:
        1. Normal trading fee on the sell (same formula as entry: round_up(0.07 × P × (1-P)))
        2. Spread/slippage cost (~$0.01) - this is market impact, not a special Kalshi fee
        """
        pos = self.position
        exit_price = pos.current_value(kalshi_prob_pct) - SLIPPAGE_CENTS
        # Trading fee on the sell side (normal Kalshi trading fee)
        # Note: Fee depends on exit price - can be higher at mid prices (e.g., $0.50)
        exit_trading_fee = self.fee_model.exit_fee(exit_price, n_contracts=1)
        # Spread/slippage cost (market impact, not a special fee)
        exit_spread_cost = self.fee_model.safety_sell_fee()
        total_exit_cost = exit_trading_fee + exit_spread_cost
        gross_pnl = exit_price - pos.entry_price
        net_pnl   = gross_pnl - pos.entry_fee - total_exit_cost

        # Determine exit type label
        reason_label = reason.split('(')[0].strip()
        exit_type = f"sell:{reason_label}"

        self.trades.append(TradeRecord(
            event_id        = self.event_id,
            team            = pos.team,
            direction       = pos.direction.value,
            entry_price     = pos.entry_price,
            entry_time      = pos.entry_time,
            entry_prob      = pos.entry_prob,
            entry_fee       = pos.entry_fee,
            exit_price      = exit_price,
            exit_time       = elapsed_sec,
            exit_prob       = kalshi_prob_pct,
            exit_fee        = total_exit_cost,
            exit_type       = exit_type,
            gross_pnl       = gross_pnl,
            net_pnl         = net_pnl,
            hold_ticks      = pos.n_ticks_held,
            edge_at_entry   = 0,
            cal_prob_entry  = 0,
            safety_sell_count = self.safety_sell_count,
        ))

        self.position = None
        self.ticks_since_exit = 0       # start cooldown
        self.safety_sell_count += 1

        if self.mode == "restricted":
            self.can_buy = True   # RESET — can buy again after safety sell
            if self.safety_sell_count >= MAX_SAFETY_SELLS:
                self.can_buy = False  # hit cap → no more buys
        # In unrestricted mode: can_buy stays True, will re-enter after cooldown

    # ── SETTLE ────────────────────────────────────────────────────────────
    def settle(self, team_won_map: Dict[str, bool]):
        """Settle at game end.  team_won_map = {team_name: True/False}."""
        if self.position is None:
            return

        pos = self.position
        team_won = team_won_map.get(pos.team, False)

        if pos.direction == Direction.YES:
            settlement_val = 1.0 if team_won else 0.0
        else:  # NO
            settlement_val = 1.0 if not team_won else 0.0

        gross_pnl = settlement_val - pos.entry_price
        # No settlement fee — fees are paid at entry
        settle_fee = 0.0
        net_pnl    = gross_pnl - pos.entry_fee

        exit_type = "settlement_win" if settlement_val == 1.0 else "settlement_loss"

        self.trades.append(TradeRecord(
            event_id        = self.event_id,
            team            = pos.team,
            direction       = pos.direction.value,
            entry_price     = pos.entry_price,
            entry_time      = pos.entry_time,
            entry_prob      = pos.entry_prob,
            entry_fee       = pos.entry_fee,
            exit_price      = settlement_val,
            exit_time       = REGULATION_SEC,
            exit_prob       = 100.0 if team_won else 0.0,
            exit_fee        = settle_fee,
            exit_type       = exit_type,
            gross_pnl       = gross_pnl,
            net_pnl         = net_pnl,
            hold_ticks      = pos.n_ticks_held,
            edge_at_entry   = 0,
            cal_prob_entry  = 0,
            safety_sell_count = self.safety_sell_count,
        ))
        self.position = None


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTESTER  — runs the strategy on historical data
# ═══════════════════════════════════════════════════════════════════════════════
class Backtester:
    """Full historical backtest engine."""

    def __init__(self, cal_model: CalibrationModel, fee_model: FeeModel):
        self.cal_model = cal_model
        self.fee_model = fee_model

    def run(self, csv_path: str = INPUT_CSV) -> Dict:
        """
        Run the backtest on historical data.

        Returns a dict with:
          'trades'  : list of TradeRecord
          'summary' : dict of aggregate statistics
          'by_game' : per-game results
        """
        print(f"\n{'='*70}")
        print(f"  BACKTESTING LIVE TRADING STRATEGY  [{STRATEGY_MODE.upper()} MODE]")
        print(f"{'='*70}")
        print(f"  Strategy mode:        {STRATEGY_MODE.upper()}")
        print(f"  Fee model:            {self.fee_model}")
        print(f"  Clock uncertainty:    ±{CLOCK_UNCERTAINTY_SEC}s")
        print(f"  Min edge (early/mid/late): {MIN_EDGE_EARLY*100:.1f}¢ / "
              f"{MIN_EDGE_MID*100:.1f}¢ / {MIN_EDGE_LATE*100:.1f}¢")
        print(f"  Min EV after fees:    {MIN_EV_AFTER_FEES*100:.1f}¢")
        if STRATEGY_MODE == "restricted":
            print(f"  Max safety sells:     {MAX_SAFETY_SELLS}")
            print(f"  Stop loss:            {STOP_LOSS_CENTS*100:.0f}¢")
        else:
            print(f"  Model exit buffer:    {UNRES_MODEL_EXIT_BUFFER*100:.1f}¢")
            print(f"  Profit-take min:      {UNRES_PROFIT_TAKE_MIN*100:.1f}¢ (edge < {UNRES_EDGE_GONE_THRESH*100:.1f}¢)")
            print(f"  Re-entry cooldown:    {UNRES_REENTRY_COOLDOWN} ticks")

        # ── load data ─────────────────────────────────────────────────────
        print(f"\n  Loading data from {csv_path} …")
        df = pd.read_csv(csv_path)
        n_rows = len(df)
        events = df["kalshi_event"].unique()
        n_events = len(events)
        print(f"  {n_rows:,} rows across {n_events:,} game events")

        # ── filter to TEST SET ONLY (avoid data leakage) ──────────────────
        test_ids_path = os.path.join(OUTPUT_DIR, "test_game_ids.txt")
        if os.path.exists(test_ids_path):
            with open(test_ids_path, "r") as f:
                test_game_ids = set(line.strip() for line in f if line.strip())
            # Create game_id column (same format as model.py)
            df["game_id"] = df["kalshi_event"] + "_" + df["team"]
            df_before = len(df)
            df = df[df["game_id"].isin(test_game_ids)].copy()
            print(f"  Filtered to TEST SET ONLY: {len(df):,} rows "
                  f"({len(df)/df_before*100:.1f}% of data, {len(test_game_ids):,} test games)")
        else:
            print(f"  ⚠ WARNING: {test_ids_path} not found!")
            print(f"    Backtesting on ALL data (includes training set — DATA LEAKAGE)")
            print(f"    Run model.py first to generate test set split.")

        # ── deduplicate: keep last obs per (event, team, time) ────────────
        df = df.sort_values(["kalshi_event", "team", "game_elapsed_seconds"])
        df = df.drop_duplicates(
            subset=["kalshi_event", "team", "game_elapsed_seconds"],
            keep="last",
        )
        print(f"  After dedup: {len(df):,} rows")

        # ── run game by game ──────────────────────────────────────────────
        all_trades = []
        games_traded = 0
        games_skipped = 0

        game_groups = df.groupby("kalshi_event")
        total_games = len(game_groups)

        for i, (event_id, game_df) in enumerate(game_groups):
            if (i + 1) % 500 == 0:
                print(f"    Processing game {i+1:,}/{total_games:,} …")

            # Build team_won map for settlement
            teams = game_df["team"].unique()
            team_won_map = {}
            for t in teams:
                team_rows = game_df[game_df["team"] == t]
                won = team_rows["team_won"].iloc[0]
                team_won_map[t] = bool(won)

            # Create game trader
            trader = GameTrader(event_id, self.cal_model, self.fee_model)

            # Sort all ticks chronologically
            ticks = game_df.sort_values("game_elapsed_seconds")

            for _, row in ticks.iterrows():
                trader.process_tick(
                    team            = row["team"],
                    kalshi_prob_pct = row["win_prob_pct"],
                    elapsed_sec     = row["game_elapsed_seconds"],
                    volume          = row.get("volume", 9999),
                )

            # Settle at game end
            trader.settle(team_won_map)

            if trader.trades:
                games_traded += 1
                all_trades.extend(trader.trades)
            else:
                games_skipped += 1

        print(f"\n  Backtest complete:")
        print(f"    Games traded:  {games_traded:,}")
        print(f"    Games skipped: {games_skipped:,} (no signal met thresholds)")
        print(f"    Total trades:  {len(all_trades):,}")

        return {
            "trades":       all_trades,
            "n_games":      total_games,
            "games_traded": games_traded,
            "games_skipped": games_skipped,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
def build_trades_df(trades: List[TradeRecord]) -> pd.DataFrame:
    """Convert trade records to a DataFrame."""
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        rows.append({
            "event_id":        t.event_id,
            "team":            t.team,
            "direction":       t.direction,
            "entry_price":     t.entry_price,
            "entry_time_sec":  t.entry_time,
            "entry_time_min":  t.entry_time / 60.0,
            "entry_prob":      t.entry_prob,
            "entry_fee":       t.entry_fee,
            "exit_price":      t.exit_price,
            "exit_time_sec":   t.exit_time,
            "exit_time_min":   t.exit_time / 60.0,
            "exit_prob":       t.exit_prob,
            "exit_fee":        t.exit_fee,
            "exit_type":       t.exit_type,
            "gross_pnl":       t.gross_pnl,
            "net_pnl":         t.net_pnl,
            "hold_ticks":      t.hold_ticks,
            "safety_sell_ct":  t.safety_sell_count,
        })
    return pd.DataFrame(rows)


def print_backtest_report(results: Dict):
    """Print comprehensive backtest statistics."""
    trades_df = build_trades_df(results["trades"])
    if trades_df.empty:
        print("\n  ⚠ No trades were generated. Try lowering thresholds.")
        return trades_df

    n_games  = results["n_games"]
    n_traded = results["games_traded"]

    # ── Categorise trades ─────────────────────────────────────────────────
    settlements  = trades_df[trades_df["exit_type"].str.startswith("settlement")]
    safety_sells = trades_df[trades_df["exit_type"].str.startswith("safety_sell")]
    wins         = settlements[settlements["exit_type"] == "settlement_win"]
    losses       = settlements[settlements["exit_type"] == "settlement_loss"]

    total_trades  = len(trades_df)
    n_settle_win  = len(wins)
    n_settle_loss = len(losses)
    n_safety      = len(safety_sells)

    total_net_pnl  = trades_df["net_pnl"].sum()
    total_gross    = trades_df["gross_pnl"].sum()
    total_fees     = trades_df["entry_fee"].sum() + trades_df["exit_fee"].sum()
    avg_net_pnl    = trades_df["net_pnl"].mean()
    median_net_pnl = trades_df["net_pnl"].median()

    win_rate_settle = n_settle_win / max(1, n_settle_win + n_settle_loss)
    avg_win  = wins["net_pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["net_pnl"].mean() if len(losses) > 0 else 0
    avg_safety_pnl = safety_sells["net_pnl"].mean() if len(safety_sells) > 0 else 0

    # Capital metrics
    avg_entry_cost = trades_df["entry_price"].mean()
    total_capital_deployed = trades_df["entry_price"].sum()

    # ── ROI ───────────────────────────────────────────────────────────────
    roi_total = total_net_pnl / total_capital_deployed * 100 if total_capital_deployed > 0 else 0
    roi_per_trade = avg_net_pnl / avg_entry_cost * 100 if avg_entry_cost > 0 else 0

    # ── Direction breakdown ───────────────────────────────────────────────
    yes_trades = trades_df[trades_df["direction"] == "YES"]
    no_trades  = trades_df[trades_df["direction"] == "NO"]

    # ── Profit factor ─────────────────────────────────────────────────────
    gross_wins  = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
    gross_losses = abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum())
    profit_factor = gross_wins / max(0.001, gross_losses)

    print(f"\n{'='*70}")
    print("  BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"""
  Overview
  ────────
    Total games in dataset:    {n_games:,}
    Games with trades:         {n_traded:,}  ({n_traded/n_games*100:.1f}%)
    Total trades executed:     {total_trades:,}
      → Settled (win):          {n_settle_win:,}
      → Settled (loss):         {n_settle_loss:,}
      → Safety sells:           {n_safety:,}

  P&L Summary
  ───────────
    Total gross P&L:          ${total_gross:>+10.2f}
    Total fees paid:          ${total_fees:>10.2f}
    Total net P&L:            ${total_net_pnl:>+10.2f}
    Avg net P&L per trade:    ${avg_net_pnl:>+10.4f}  ({avg_net_pnl*100:+.2f}¢)
    Median net P&L per trade: ${median_net_pnl:>+10.4f}  ({median_net_pnl*100:+.2f}¢)
    Profit factor:              {profit_factor:.2f}x

  Win Rates
  ─────────
    Settlement win rate:      {win_rate_settle*100:.1f}%  ({n_settle_win}/{n_settle_win + n_settle_loss})
    Avg winning trade:        ${avg_win:>+.4f}  ({avg_win*100:+.2f}¢)
    Avg losing trade:         ${avg_loss:>+.4f}  ({avg_loss*100:+.2f}¢)
    Avg safety-sell P&L:      ${avg_safety_pnl:>+.4f}  ({avg_safety_pnl*100:+.2f}¢)

  Capital & ROI
  ─────────────
    Avg entry cost per trade: ${avg_entry_cost:.4f}
    Total capital deployed:   ${total_capital_deployed:.2f}
    ROI (total):              {roi_total:+.2f}%
    ROI (per trade avg):      {roi_per_trade:+.2f}%

  Direction Breakdown
  ───────────────────
    YES trades: {len(yes_trades):,}  (net P&L: ${yes_trades['net_pnl'].sum():+.2f})
    NO  trades: {len(no_trades):,}  (net P&L: ${no_trades['net_pnl'].sum():+.2f})
""")

    # ── By exit type ──────────────────────────────────────────────────────
    print("  P&L by Exit Type:")
    print("  " + "─" * 65)
    for etype, grp in trades_df.groupby("exit_type"):
        print(f"    {etype:<35}  n={len(grp):>5}  "
              f"net=${grp['net_pnl'].sum():>+8.2f}  "
              f"avg=${grp['net_pnl'].mean():>+.4f}")
    print()

    # ── By entry time ─────────────────────────────────────────────────────
    print("  P&L by Entry Time (game phase):")
    print("  " + "─" * 65)
    time_bins = [(0, 10, "0-10 min"), (10, 20, "10-20 min (halftime)"),
                 (20, 30, "20-30 min"), (30, 40, "30-40 min")]
    for lo, hi, label in time_bins:
        mask = (trades_df["entry_time_min"] >= lo) & (trades_df["entry_time_min"] < hi)
        grp = trades_df[mask]
        if len(grp) > 0:
            print(f"    {label:<25}  n={len(grp):>5}  "
                  f"net=${grp['net_pnl'].sum():>+8.2f}  "
                  f"avg=${grp['net_pnl'].mean():>+.4f}  "
                  f"winrate={len(grp[grp['net_pnl']>0])/len(grp)*100:.0f}%")
    print()

    # ── By entry probability band ─────────────────────────────────────────
    print("  P&L by Entry Probability Band:")
    print("  " + "─" * 65)
    prob_bins = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                 (50, 60), (60, 70), (70, 80), (80, 90), (90, 99)]
    for lo, hi in prob_bins:
        mask = (trades_df["entry_prob"] >= lo) & (trades_df["entry_prob"] <= hi)
        grp = trades_df[mask]
        if len(grp) > 0:
            print(f"    {lo:>2}-{hi:<2}%  n={len(grp):>5}  "
                  f"net=${grp['net_pnl'].sum():>+8.2f}  "
                  f"avg=${grp['net_pnl'].mean():>+.4f}  "
                  f"dir={grp['direction'].mode().iloc[0] if len(grp) > 0 else '—'}")
    print()

    # ── Projections ───────────────────────────────────────────────────────
    trades_per_game = total_trades / n_games
    pnl_per_game    = total_net_pnl / n_games
    games_per_day   = 8  # rough estimate during season

    print(f"  Projections (based on backtest)")
    print(f"  {'─'*50}")
    print(f"    Trades per game (avg):    {trades_per_game:.2f}")
    print(f"    Net P&L per game:         ${pnl_per_game:.4f}  ({pnl_per_game*100:.2f}¢)")
    print(f"    Games per day (est):      {games_per_day}")
    print(f"    Net P&L per day (est):    ${pnl_per_game * games_per_day:.4f}")
    print(f"    Net P&L per month (est):  ${pnl_per_game * games_per_day * 30:.2f}")

    for n_contracts in [1, 5, 10, 25]:
        monthly = pnl_per_game * games_per_day * 30 * n_contracts
        capital = avg_entry_cost * trades_per_game * games_per_day * n_contracts
        roi_m   = monthly / max(0.01, capital) * 100
        print(f"      {n_contracts:>2} contracts/trade: "
              f"${monthly:>+8.2f}/mo  (needs ~${capital:.0f} capital, {roi_m:+.1f}% monthly ROI)")
    print()

    return trades_df


def plot_backtest_results(trades_df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Generate backtest visualization plots."""
    if trades_df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Live Trading Strategy — Backtest Results", fontsize=18, fontweight="bold", y=1.02)

    # ── 1. Cumulative P&L ─────────────────────────────────────────────────
    ax = axes[0, 0]
    cum_pnl = trades_df["net_pnl"].cumsum()
    ax.plot(cum_pnl.values, color="#2E86C1", lw=1.5)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.fill_between(range(len(cum_pnl)), cum_pnl.values, 0,
                    where=cum_pnl.values >= 0, color="#27AE60", alpha=0.3)
    ax.fill_between(range(len(cum_pnl)), cum_pnl.values, 0,
                    where=cum_pnl.values < 0, color="#E74C3C", alpha=0.3)
    ax.set_xlabel("Trade #", fontsize=12)
    ax.set_ylabel("Cumulative Net P&L ($)", fontsize=12)
    ax.set_title("Cumulative P&L (equity curve)", fontsize=14)
    ax.grid(True, alpha=0.3)

    # ── 2. P&L distribution ───────────────────────────────────────────────
    ax = axes[0, 1]
    pnl_cents = trades_df["net_pnl"] * 100
    ax.hist(pnl_cents, bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", lw=1, ls="--")
    ax.axvline(pnl_cents.mean(), color="green", lw=2, ls="-", label=f"Mean={pnl_cents.mean():.1f}¢")
    ax.set_xlabel("Net P&L per trade (¢)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("P&L Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── 3. Entry time distribution ────────────────────────────────────────
    ax = axes[0, 2]
    ax.hist(trades_df["entry_time_min"], bins=20, color="#F39C12", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Entry Time (minutes)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("When Trades Are Entered", fontsize=14)
    ax.grid(True, alpha=0.3)

    # ── 4. Entry probability distribution ─────────────────────────────────
    ax = axes[1, 0]
    yes_mask = trades_df["direction"] == "YES"
    no_mask  = trades_df["direction"] == "NO"
    ax.hist(trades_df.loc[yes_mask, "entry_prob"], bins=50, color="#27AE60",
            alpha=0.6, label="YES", edgecolor="white")
    ax.hist(trades_df.loc[no_mask, "entry_prob"], bins=50, color="#E74C3C",
            alpha=0.6, label="NO", edgecolor="white")
    ax.set_xlabel("Kalshi Win Probability at Entry (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Entry Probability Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── 5. Exit type breakdown ────────────────────────────────────────────
    ax = axes[1, 1]
    exit_types = trades_df["exit_type"].apply(lambda x: x.split(":")[0]).value_counts()
    colors_map = {"settlement_win": "#27AE60", "settlement_loss": "#E74C3C", "safety_sell": "#F39C12"}
    colors = [colors_map.get(et, "#95A5A6") for et in exit_types.index]
    ax.bar(exit_types.index, exit_types.values, color=colors, edgecolor="white")
    ax.set_xlabel("Exit Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Trade Outcomes", fontsize=14)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 6. P&L by entry time ─────────────────────────────────────────────
    ax = axes[1, 2]
    trades_df["time_bin_5min"] = (trades_df["entry_time_min"] // 5) * 5
    time_pnl = trades_df.groupby("time_bin_5min")["net_pnl"].agg(["sum", "count", "mean"])
    bars = ax.bar(time_pnl.index, time_pnl["sum"] * 100, width=4,
                  color=["#27AE60" if v > 0 else "#E74C3C" for v in time_pnl["sum"]],
                  edgecolor="white", alpha=0.8)
    ax.set_xlabel("Entry Time (5-min bins)", fontsize=12)
    ax.set_ylabel("Total Net P&L (¢)", fontsize=12)
    ax.set_title("Profit by Game Phase", fontsize=14)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "backtest_results.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved backtest plots → {path}")

    # ── Save trades CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "backtest_trades.csv")
    trades_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved trades CSV    → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE STRATEGY SUMMARY  — printable rules for actual trading
# ═══════════════════════════════════════════════════════════════════════════════
def print_live_strategy_rules():
    """Print the complete rule set for live execution."""
    mode = STRATEGY_MODE.upper()
    print(f"""
{'='*70}
  LIVE TRADING RULES — Kalshi NCAAB Basketball  [{mode} MODE]
{'='*70}""")

    if STRATEGY_MODE == "restricted":
        print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  RULE 1: ONE BUY PER GAME                                      │
  │  • You get exactly ONE buy per game event                       │
  │  • EXCEPTION: if you safety-sell, you may buy again             │
  │  • Max {MAX_SAFETY_SELLS} safety sell(s) per game                              │
  └─────────────────────────────────────────────────────────────────┘""")
    else:
        print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  UNRESTRICTED MODE                                              │
  │  • Buy and sell freely — multiple round-trips per game          │
  │  • Not required to trade every game                             │
  │  • Enter on every good signal, exit when edge disappears        │
  │  • {UNRES_REENTRY_COOLDOWN}-tick cooldown after selling before re-entering           │
  └─────────────────────────────────────────────────────────────────┘""")

    print(f"""
  ENTRY CONDITIONS (ALL must be true):
  ─────────────────────────────────────
    1. Edge exceeds time-dependent threshold:
         • 0-18 min:  edge ≥ {MIN_EDGE_EARLY*100:.1f}¢
         • 18-30 min: edge ≥ {MIN_EDGE_MID*100:.1f}¢
         • 30-40 min: edge ≥ {MIN_EDGE_LATE*100:.1f}¢

    2. EV after fees ≥ {MIN_EV_AFTER_FEES*100:.1f}¢ per contract

    3. Volume ≥ {MIN_VOLUME} contracts (liquidity check)

    4. Historical cell has ≥ {MIN_HIST_OBS} observations (reliability)

    5. Edge is ROBUST across ±{CLOCK_UNCERTAINTY_SEC}s clock uncertainty
       (conservative evaluation at worst-case time in window)

  HOW TO COMPUTE EDGE:
  ────────────────────
    • Look up Kalshi win probability (p) and ESPN game clock
    • Query GAM calibration at 5 points across ±60s window
    • For YES: edge = min(calibrated probs in window) − p
    • For NO:  edge = p − max(calibrated probs in window)
    • Best direction = whichever has higher EV after fees""")

    if STRATEGY_MODE == "unrestricted":
        print(f"""
  EXIT CONDITIONS (UNRESTRICTED MODE):
  ────────────────────────────────────
    1. MODEL-BASED EXIT: sell if expected value of holding to settlement
       is worse than selling now by ≥ {UNRES_MODEL_EXIT_BUFFER*100:.1f}¢
       → GAM model confirms position is losing, exit and re-scan

    2. PROFIT-TAKE: sell if unrealised gain ≥ {UNRES_PROFIT_TAKE_MIN*100:.1f}¢
       AND the edge that motivated entry has disappeared (< {UNRES_EDGE_GONE_THRESH*100:.1f}¢)
       → Lock in profit, look for next entry

    3. PROBABILITY REVERSAL: sell if we bought NO on underdog
       and they reach ≥{REVERSAL_PROB_THRESHOLD}% in final {(REGULATION_SEC - REVERSAL_MIN_TIME_SEC)//60} min
       → Game truly flipped, cut losses

    4. OTHERWISE: hold to settlement (most profitable default)

  RE-ENTRY AFTER SELLING:
  ──────────────────────
    • Wait {UNRES_REENTRY_COOLDOWN} ticks (~{UNRES_REENTRY_COOLDOWN * 25}s) cooldown after any sell
    • Then scan for new entry signal as normal
    • No limit on number of round-trips per game""")
    else:
        print(f"""
  EXIT CONDITIONS (RESTRICTED MODE):
  ──────────────────────────────────
    ★ PROBABILITY REVERSAL: sell if we bought NO on a ≤{MAX_PROB_FOR_NO}% team
       and they reach ≥{REVERSAL_PROB_THRESHOLD}% in the last {(REGULATION_SEC - REVERSAL_MIN_TIME_SEC)//60} min

    • STOP LOSS / TRAILING STOP: Disabled (basketball too volatile)
    • After safety sell, 1-buy rule resets (max {MAX_SAFETY_SELLS} per game)""")

    print(f"""
  FEE AWARENESS:
  ──────────────
    • Trading fees are charged on EVERY trade (buy or sell), NOT at settlement
    • Taker fees (immediately matched): round_up(0.07 × C × P × (1-P))
    • Maker fees (resting limit orders, when available): round_up(0.0175 × C × P × (1-P))
    • IMPORTANT: Rounding is on TOTAL fee, not per contract
      - 1 contract at $0.95: fee = $0.01 (1¢ per contract)
      - 100 contracts at $0.95: fee = $0.34 total (0.34¢ per contract)
    • When selling mid-game: pay trading fee on exit price + ~${SAFETY_SELL_COST:.2f} spread/slippage
    • Exit fees can be HIGHER at mid prices (e.g., $0.50 → $0.02 fee vs $0.95 → $0.01 fee)
    • Fees are highest near 50/50 (P × (1-P) is maximized at P=0.5) → avoid trading there

  ⚠  CAVEATS:
  ──────────
    • Edge is SMALL (~1-3¢ per trade) — need volume to make money
    • Past calibration patterns may not persist
    • Assumes fills at quoted Kalshi mid-price (no spread)
    • GAM model has estimation uncertainty (±1-3 pp)
    • ESPN clock delay (±60s) handled conservatively but not perfectly
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """Run the full strategy pipeline: backtest + analysis + rules."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load models
    print(f"\n{'='*70}")
    print(f"  KALSHI NCAAB LIVE TRADING STRATEGY  [{STRATEGY_MODE.upper()} MODE]")
    print(f"{'='*70}")

    print("\n  [1/5] Loading calibration model …")
    cal_model = CalibrationModel()
    print(f"    Calibration surface: {len(cal_model.prob_grid)} probs × "
          f"{len(cal_model.time_grid)} time bins")
    if cal_model.raw_counts is not None:
        print(f"    Historical games:    {cal_model.n_games:,}")

    print("\n  [2/5] Initialising fee model …")
    fee_model = FeeModel()
    print(f"    {fee_model}")
    print(f"    Entry fee on $0.05 contract (5% underdog): ${fee_model.entry_fee(0.05):.3f}")
    print(f"    Entry fee on $0.95 contract (95% favorite): ${fee_model.entry_fee(0.95):.3f}")
    print(f"    Mid-game sell cost per trade:   ${fee_model.safety_sell_fee():.3f}")

    # 2. Backtest
    print("\n  [3/5] Running backtest …")
    bt = Backtester(cal_model, fee_model)
    results = bt.run()

    # 3. Report
    print("\n  [4/5] Analysing results …")
    trades_df = print_backtest_report(results)

    # 4. Plots
    if not trades_df.empty:
        print("\n  [5/5] Generating plots …")
        plot_backtest_results(trades_df)

    # 5. Print live rules
    print_live_strategy_rules()

    print(f"\n{'='*70}")
    print(f"  STRATEGY COMPLETE [{STRATEGY_MODE.upper()}] — all outputs in GeneratedDataFiles/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
