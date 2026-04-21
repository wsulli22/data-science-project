"""
optimizer.py — Bayesian hyper-parameter search for the Kalshi NCAAB betting strategy.

OVERVIEW
--------
Uses Optuna (TPE/Bayesian sampler) to find the combination of strategy parameters
that maximises total profit across all out-of-sample test weeks.

Walk-forward cross-validation prevents look-ahead bias:
  - Test week N is evaluated using a calibration surface built only from weeks 1..(N-1).
  - The calibration surface estimates the TRUE win rate for each (probability, game-time)
    cell from historical data, enabling Kelly-style bet sizing.
  - Weeks 1 and 2 are used only as training data; testing starts at week 3.

Bankroll: $1000 per week (resets each week, treated independently).

NEW PARAMETERS vs. BASELINE evaluator.py
-----------------------------------------
  prob_floor        min Kalshi win-probability to consider betting      (%)
  prob_ceiling      skip bets where the probability is already this high (%)
  min_elapsed_s     earliest allowed game-clock position                (seconds)
  max_elapsed_s     latest allowed game-clock position                  (seconds)
  momentum_window_s look-back window for the momentum filter            (seconds)
  min_momentum_pp   minimum probability rise over that window           (pp)
  min_volume        minimum cumulative volume at bet time
  kelly_fraction    weight on full-Kelly sizing  (0 = use flat_bet_pct)
  kelly_cap_pct     hard ceiling on fraction of bankroll per bet        (%)
  flat_bet_pct      bet size as % of bankroll when kelly_fraction = 0   (%)

USAGE
-----
  pip install optuna pandas numpy
  python ben-modified-by-will.py
  python ben-modified-by-will.py --fast          # quick run (~40 trials, 4 workers)
  python ben-modified-by-will.py --trials 80 --workers 4

  The script expects week_1_games.csv … week_19_games.csv to live in a
  subdirectory called "Data/" relative to this file (same layout as evaluator.py).

OUTPUT
------
  Prints a per-week walk-forward summary for the best trial, then the
  optimal parameter set, then reruns that set explicitly so you can
  copy the parameters straight into evaluator.py.

CHANGES FROM ben_model.py
-------------------------
  1. _buy() now matches fee_calculator.buy() — target_dollars is the max total
     debit (notional + fee), not just notional.  Uses iterative sizing so the
     contract count respects the budget inclusive of fees.
  2. Momentum filter uses actual elapsed-time differences instead of treating
     row indices as seconds.
  3. All dollar amounts routed through _usd2() (Decimal ROUND_HALF_UP) to match
     fee_calculator.py's financial rounding.
  4. Payout rounded to cents for parity with the evaluator.
  5. Volume filter is skipped from Optuna search when no volume column exists
     in the data, avoiding wasted optimisation budget.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
import warnings
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from heapq import heappop, heappush
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import optuna.trial
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── User-adjustable configuration ─────────────────────────────────────────────

DATA_DIRECTORY = Path(__file__).resolve().parent / "Data"

NUM_WEEKS = 19
STARTING_BANKROLL = 1000.0            # dollars, resets each week
SETTLEMENT_BUFFER_SECONDS = 2 * 60 * 60
TAKER_FEE_RATE = 0.07

# Optimisation
N_OPTUNA_TRIALS = 500                  # increase for more thorough search
N_OPTUNA_WORKERS = 10                  # parallel Optuna workers (threads)
OPTUNA_SEED = 42
MIN_TRAIN_WEEKS = 2                    # need at least this many weeks before testing

# Degenerate-solution guard: penalise if fewer bets made across all test weeks
MIN_TOTAL_BETS_REQUIRED = 0

# Calibration surface resolution
# Edges are right-exclusive except the last bucket which catches everything above
PROB_BIN_EDGES  = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 101]
TIME_BIN_EDGES  = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 3600, 99999]

N_PROB_BINS = len(PROB_BIN_EDGES)
N_TIME_BINS = len(TIME_BIN_EDGES)


# ── Fee helpers (matching fee_calculator.py exactly) ─────────────────────────

def _usd2(x: float) -> float:
    """Financial-style rounding to cents (half-up), matching fee_calculator.py."""
    return float(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _taker_fee(price: float, contracts: int) -> float:
    p = min(1.0, max(0.0, price))
    notional = contracts * p
    return _usd2(math.ceil(100 * TAKER_FEE_RATE * notional * (1.0 - p)) / 100)


def _total_buy_cost(contracts: int, price: float) -> float:
    if contracts <= 0:
        return _usd2(0.0)
    actual = _usd2(contracts * price)
    fee = _taker_fee(price, contracts)
    return _usd2(actual + fee)


def _buy(target_dollars: float, contract_price: float) -> tuple[float, float, int]:
    """
    Budget-aware buy matching fee_calculator.buy(): target_dollars is the max
    total debit (notional + fee).  Iteratively finds the largest contract count
    whose total cost fits within budget.
    """
    price = min(1.0, max(0.01, contract_price))
    budget = _usd2(max(0.0, target_dollars))
    if budget <= 0.0:
        return 0.0, 0.0, 0
    contracts = 0
    while _total_buy_cost(contracts + 1, price) <= budget:
        contracts += 1
    if contracts <= 0:
        return 0.0, 0.0, 0
    stake = _usd2(contracts * price)
    fee = _taker_fee(price, contracts)
    return stake, fee, contracts


# ── Data loading and preprocessing ────────────────────────────────────────────

_HAS_VOLUME_DATA = False  # set at load time; controls whether Optuna tunes min_volume


def _detect_volume_column(df: pd.DataFrame) -> Optional[str]:
    """Find the volume column; returns None if not present."""
    candidates = ["volume", "total_volume", "trade_volume", "vol"]
    for c in candidates:
        if c in df.columns:
            return c
    vol_cols = [c for c in df.columns if "volume" in c.lower()]
    return vol_cols[0] if vol_cols else None


def load_and_preprocess_weeks(data_dir: Path) -> list[list[dict]]:
    """
    Load all 19 week CSVs and convert each game into a dict of numpy arrays
    for fast vectorised processing.

    Returns a list of 19 elements (one per week); each element is a list of
    game-dicts (one per game in that week).
    """
    global _HAS_VOLUME_DATA
    all_weeks: list[list[dict]] = []
    found_volume = False

    for w in range(1, NUM_WEEKS + 1):
        path = data_dir / f"week_{w}_games.csv"
        df = pd.read_csv(path)
        df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"])
        df = df.sort_values(
            ["kalshi_event", "realworld_timestamp"], kind="mergesort"
        ).reset_index(drop=True)

        vol_col = _detect_volume_column(df)
        if vol_col is not None:
            found_volume = True
        week_games: list[dict] = []

        for event_id, grp in df.groupby("kalshi_event", sort=False):
            grp = grp.sort_values("realworld_timestamp").reset_index(drop=True)
            n = len(grp)

            volume_arr = (
                grp[vol_col].astype(float).values
                if vol_col is not None
                else np.zeros(n)
            )

            week_games.append(
                {
                    "event_id": str(event_id),
                    "team_1":   str(grp["team_1"].iloc[0]),
                    "team_2":   str(grp["team_2"].iloc[0]),
                    "winner":   str(grp["winning_team"].iloc[0]),
                    "p1":       grp["team_1_win_prob_pct"].astype(float).values,
                    "p2":       grp["team_2_win_prob_pct"].astype(float).values,
                    "elapsed":  grp["game_elapsed_seconds"].astype(float).values,
                    "ts":       grp["realworld_timestamp"].values,         # numpy datetime64
                    "game_end": grp["realworld_timestamp"].max(),
                    "volume":   volume_arr,
                }
            )

        all_weeks.append(week_games)
        print(f"  Loaded week {w:>2d}: {len(week_games):3d} games")

    _HAS_VOLUME_DATA = found_volume
    if not found_volume:
        print("  [NOTE] No volume column found in data — min_volume filter disabled.")

    return all_weeks


# ── Calibration surface ────────────────────────────────────────────────────────

def precompute_week_contributions(
    all_weeks: list[list[dict]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    For each week, build (wins_arr, counts_arr) of shape (N_PROB_BINS, N_TIME_BINS).

    We sample every 60 seconds per game (vectorised per week) to avoid per-row Python
    loops while still capturing calibration signal across the full game timeline.
    """
    prob_edges = np.array(PROB_BIN_EDGES, dtype=float)
    time_edges = np.array(TIME_BIN_EDGES, dtype=float)
    contributions: list[tuple[np.ndarray, np.ndarray]] = []

    for week_games in all_weeks:
        wins_arr   = np.zeros((N_PROB_BINS, N_TIME_BINS))
        counts_arr = np.zeros((N_PROB_BINS, N_TIME_BINS))

        for game in week_games:
            p1, p2, elapsed = game["p1"], game["p2"], game["elapsed"]
            winner, team_1  = game["winner"], game["team_1"]
            n = len(p1)

            # Sample at most one row per 60-second bucket per game
            buckets  = (elapsed // 60).astype(int)
            _, first = np.unique(buckets, return_index=True)
            idx = first  # one index per 60-second bucket

            s_p1  = p1[idx]
            s_p2  = p2[idx]
            s_el  = elapsed[idx]
            is_t1_win = winner == team_1  # scalar bool

            # --- Team-1 perspective (favoured when p1 >= 50) ---
            mask_t1 = s_p1 >= 50
            if mask_t1.any():
                probs_t1 = s_p1[mask_t1]
                elt_t1   = s_el[mask_t1]
                pb = (np.searchsorted(prob_edges, probs_t1, side="right") - 1).clip(0, N_PROB_BINS - 1)
                tb = (np.searchsorted(time_edges, elt_t1,   side="right") - 1).clip(0, N_TIME_BINS - 1)
                np.add.at(counts_arr, (pb, tb), 1)
                np.add.at(wins_arr,   (pb, tb), int(is_t1_win))

            # --- Team-2 perspective (favoured when p2 >= 50, i.e. p1 < 50) ---
            mask_t2 = s_p2 >= 50
            if mask_t2.any():
                probs_t2 = s_p2[mask_t2]
                elt_t2   = s_el[mask_t2]
                pb = (np.searchsorted(prob_edges, probs_t2, side="right") - 1).clip(0, N_PROB_BINS - 1)
                tb = (np.searchsorted(time_edges, elt_t2,   side="right") - 1).clip(0, N_TIME_BINS - 1)
                np.add.at(counts_arr, (pb, tb), 1)
                np.add.at(wins_arr,   (pb, tb), int(not is_t1_win))

        contributions.append((wins_arr, counts_arr))

    return contributions


def build_surface(
    contributions: list[tuple[np.ndarray, np.ndarray]],
    up_to_week_idx: int,
) -> np.ndarray:
    """
    Aggregate win/count arrays from weeks 0..(up_to_week_idx - 1) and return
    the empirical win-rate surface.  Cells with no data are NaN.
    """
    total_wins   = np.zeros((N_PROB_BINS, N_TIME_BINS))
    total_counts = np.zeros((N_PROB_BINS, N_TIME_BINS))
    for wins, counts in contributions[:up_to_week_idx]:
        total_wins   += wins
        total_counts += counts
    return np.where(total_counts > 0, total_wins / total_counts, np.nan)


def build_surface_excluding_week(
    contributions: list[tuple[np.ndarray, np.ndarray]],
    exclude_week_idx: int,
) -> np.ndarray:
    """
    Aggregate win/count arrays from all weeks except exclude_week_idx and return
    the empirical win-rate surface. Cells with no data are NaN.
    """
    total_wins   = np.zeros((N_PROB_BINS, N_TIME_BINS))
    total_counts = np.zeros((N_PROB_BINS, N_TIME_BINS))
    for i, (wins, counts) in enumerate(contributions):
        if i == exclude_week_idx:
            continue
        total_wins   += wins
        total_counts += counts
    return np.where(total_counts > 0, total_wins / total_counts, np.nan)


def lookup_true_win_prob(
    surface: np.ndarray, prob_pct: float, elapsed_s: float
) -> Optional[float]:
    """
    Look up the empirical win rate for a given (prob, elapsed) coordinate.
    Falls back to the nearest populated cell (expanding-ring search) if the
    exact bin is empty.  Returns None only if the surface is entirely empty.
    """
    prob_edges = np.array(PROB_BIN_EDGES, dtype=float)
    time_edges = np.array(TIME_BIN_EDGES, dtype=float)

    pb = int((np.searchsorted(prob_edges, prob_pct, side="right") - 1).clip(0, N_PROB_BINS - 1))
    tb = int((np.searchsorted(time_edges, elapsed_s, side="right") - 1).clip(0, N_TIME_BINS - 1))

    if not np.isnan(surface[pb, tb]):
        return float(surface[pb, tb])

    # Expanding ring search
    for radius in range(1, max(N_PROB_BINS, N_TIME_BINS)):
        vals = []
        for dp in range(-radius, radius + 1):
            for dt in range(-radius, radius + 1):
                if max(abs(dp), abs(dt)) != radius:
                    continue
                np_ = pb + dp
                nt  = tb + dt
                if 0 <= np_ < N_PROB_BINS and 0 <= nt < N_TIME_BINS:
                    if not np.isnan(surface[np_, nt]):
                        vals.append(surface[np_, nt])
        if vals:
            return float(np.mean(vals))
    return None


# ── Kelly bet sizing ───────────────────────────────────────────────────────────

def compute_bet_size(
    p_true: Optional[float],
    p_kalshi_frac: float,       # Kalshi price in [0, 1]
    params: dict,
) -> float:
    """
    Compute target bet size in dollars from the starting bankroll.

    If kelly_fraction > 0 and p_true is available:
      - Full Kelly fraction f* = (p_true - p_kalshi) / (1 - p_kalshi)
        (derived from standard Kelly for binary payoff bets)
      - Bet = kelly_fraction * f* * STARTING_BANKROLL, capped at kelly_cap_pct %
    Otherwise:
      - Bet = flat_bet_pct % of STARTING_BANKROLL, capped at kelly_cap_pct %
    """
    cap = (params["kelly_cap_pct"] / 100.0) * STARTING_BANKROLL

    if params["kelly_fraction"] > 0.0 and p_true is not None:
        edge = p_true - p_kalshi_frac
        if edge <= 0:
            return 0.0                          # no positive edge → skip
        if p_kalshi_frac >= 1.0:
            return 0.0
        f_star = edge / (1.0 - p_kalshi_frac)
        bet = params["kelly_fraction"] * f_star * STARTING_BANKROLL
    else:
        bet = (params["flat_bet_pct"] / 100.0) * STARTING_BANKROLL

    return min(bet, cap)


# ── Core strategy: find the first qualifying bet in a single game ──────────────

def find_first_bet(
    game: dict,
    params: dict,
    surface: Optional[np.ndarray],
) -> Optional[dict]:
    """
    Vectorised scan over a game's timeline.  Returns the first row that
    satisfies ALL of the following conditions, or None.

    Conditions
    ----------
    1. game_elapsed_seconds in [min_elapsed_s, max_elapsed_s]
    2. At least one team's probability crosses UP through prob_floor
    3. That team's probability is <= prob_ceiling
    4. Probability has risen by >= min_momentum_pp over the last momentum_window_s
       of actual elapsed game time
    5. Volume >= min_volume
    6. When kelly_fraction > 0: estimated true win prob > Kalshi price (positive edge)
    """
    p1      = game["p1"]
    p2      = game["p2"]
    elapsed = game["elapsed"]
    n       = len(p1)
    if n == 0:
        return None

    floor   = params["prob_floor"]
    ceiling = params["prob_ceiling"]
    min_el  = params["min_elapsed_s"]
    max_el  = params["max_elapsed_s"]
    win_s   = float(params["momentum_window_s"])
    min_mom = params["min_momentum_pp"]
    min_vol = params["min_volume"]

    # ── 1. Clock gate ──────────────────────────────────────────────────────────
    clock_ok = (elapsed >= min_el) & (elapsed <= max_el)

    # ── 2. Threshold crossing (upward) ────────────────────────────────────────
    prev_p1 = np.empty(n)
    prev_p1[0] = 0.0
    prev_p1[1:] = p1[:-1]

    prev_p2 = np.empty(n)
    prev_p2[0] = 0.0
    prev_p2[1:] = p2[:-1]

    t1_cross = (prev_p1 < floor) & (p1 >= floor)
    t2_cross = (prev_p2 < floor) & (p2 >= floor)
    any_cross = t1_cross | t2_cross

    # ── 3. Ceiling gate (per-team) ─────────────────────────────────────────────
    t1_ceil_ok = p1 <= ceiling
    t2_ceil_ok = p2 <= ceiling
    ceiling_ok = (
        (t1_cross & t1_ceil_ok)
        | (t2_cross & t2_ceil_ok)
    )

    # ── 4. Momentum filter (using actual elapsed time, not row indices) ───────
    if min_mom > 0.0 and win_s > 0:
        p1_past = np.full(n, np.nan)
        p2_past = np.full(n, np.nan)

        for i in range(n):
            target_time = elapsed[i] - win_s
            if target_time < 0:
                continue
            # Find the last row at or before (elapsed[i] - window) via binary search
            j = np.searchsorted(elapsed[:i + 1], target_time, side="right") - 1
            if j >= 0:
                p1_past[i] = p1[j]
                p2_past[i] = p2[j]

        with np.errstate(invalid="ignore"):
            p1_mom_ok = (p1 - p1_past) >= min_mom
            p2_mom_ok = (p2 - p2_past) >= min_mom

        momentum_ok = (t1_cross & p1_mom_ok) | (t2_cross & p2_mom_ok)
    else:
        momentum_ok = np.ones(n, dtype=bool)

    # ── 5. Volume filter ───────────────────────────────────────────────────────
    if min_vol > 0.0:
        vol_ok = game["volume"] >= min_vol
    else:
        vol_ok = np.ones(n, dtype=bool)

    # ── Combined pre-Kelly mask ────────────────────────────────────────────────
    candidate = clock_ok & any_cross & ceiling_ok & momentum_ok & vol_ok
    candidate_indices = np.where(candidate)[0]

    if len(candidate_indices) == 0:
        return None

    # ── 6. Kelly / edge check (first candidate that passes) ───────────────────
    for i in candidate_indices:
        t1_ok = bool(t1_cross[i]) and bool(t1_ceil_ok[i])
        t2_ok = bool(t2_cross[i]) and bool(t2_ceil_ok[i])

        if t1_ok and t2_ok:
            if p1[i] >= p2[i]:
                bet_team, bet_prob = game["team_1"], p1[i]
            else:
                bet_team, bet_prob = game["team_2"], p2[i]
        elif t1_ok:
            bet_team, bet_prob = game["team_1"], p1[i]
        elif t2_ok:
            bet_team, bet_prob = game["team_2"], p2[i]
        else:
            continue

        p_kalshi_frac = bet_prob / 100.0
        p_true: Optional[float] = None

        if params["kelly_fraction"] > 0.0 and surface is not None:
            p_true = lookup_true_win_prob(surface, bet_prob, float(elapsed[i]))
            if p_true is None or p_true <= p_kalshi_frac:
                continue                       # no positive edge at this row

        bet_size = compute_bet_size(p_true, p_kalshi_frac, params)
        if bet_size <= 0.0:
            continue

        return {
            "event_id":    game["event_id"],
            "bet_ts":      pd.Timestamp(game["ts"][i]),
            "game_end_ts": pd.Timestamp(game["game_end"]),
            "bet_team":    bet_team,
            "prob_pct":    bet_prob,
            "winner":      game["winner"],
            "p_true":      p_true,
            "bet_size":    bet_size,
        }

    return None


# ── Weekly simulation ──────────────────────────────────────────────────────────

def simulate_week(
    week_games: list[dict],
    params: dict,
    surface: Optional[np.ndarray],
) -> dict:
    """
    Run one week of the strategy.  Returns a summary dict.
    One bet per game maximum.  FIFO bankroll with 2-hour settlement buffer.
    """
    candidates: list[dict] = []
    for game in week_games:
        bet = find_first_bet(game, params, surface)
        if bet is not None:
            candidates.append(bet)

    candidates.sort(key=lambda b: (b["bet_ts"], b["event_id"]))

    bankroll = STARTING_BANKROLL
    pending: list[tuple] = []
    total_bets = wins = losses = skipped = 0
    total_profit = 0.0
    team_won_bets = 0
    sum_return_multiple = 0.0  # sum of payout/total_cost per placed bet (0 on losses)

    for bet in candidates:
        while pending and pending[0][0] <= bet["bet_ts"]:
            _, payout, _ = heappop(pending)
            bankroll += payout

        stake, fee, contracts = _buy(bet["bet_size"], bet["prob_pct"] / 100.0)
        total_cost = _usd2(stake + fee)

        if contracts <= 0 or total_cost > bankroll:
            skipped += 1
            continue

        bankroll -= total_cost
        payout = _usd2(float(contracts)) if bet["bet_team"] == bet["winner"] else 0.0
        release_ts = bet["game_end_ts"] + timedelta(seconds=SETTLEMENT_BUFFER_SECONDS)
        heappush(pending, (release_ts, payout, bet["event_id"]))

        profit = payout - total_cost
        total_profit += profit
        total_bets += 1
        if profit >= 0:
            wins += 1
        else:
            losses += 1

        if bet["bet_team"] == bet["winner"]:
            team_won_bets += 1
        if total_cost > 0:
            sum_return_multiple += payout / total_cost

    while pending:
        _, payout, _ = heappop(pending)
        bankroll += payout

    return {
        "games":                 len(week_games),
        "games_with_opportunity": len(candidates),
        "profit":                bankroll - STARTING_BANKROLL,
        "bets":                  total_bets,
        "wins":                  wins,
        "losses":                losses,
        "skipped":               skipped,
        "final_bank":            bankroll,
        "team_won_bets":         team_won_bets,
        "sum_return_multiple":   sum_return_multiple,
    }


# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(
    all_weeks: list[list[dict]],
    contributions: list[tuple[np.ndarray, np.ndarray]],
):
    """
    Returns a closure that Optuna calls for each trial.

    Leave-one-week-out schedule over test weeks 3..19:
      - Test week N → train on all weeks except week N
    """
    first_test_week = MIN_TRAIN_WEEKS + 1   # 1-indexed

    def objective(trial: optuna.Trial) -> float:
        # ── Sample parameters ──────────────────────────────────────────────────
        prob_floor = trial.suggest_float("prob_floor", 55.0, 92.0)

        ceil_lo = min(prob_floor + 3.0, 98.0)
        prob_ceiling = trial.suggest_float("prob_ceiling", ceil_lo, 99.0)

        min_elapsed = trial.suggest_int("min_elapsed_s", 0, 2400, step=60)

        max_el_lo = min(min_elapsed + 300, 4800)
        max_elapsed = trial.suggest_int("max_elapsed_s", max_el_lo, 4800, step=60)

        momentum_window = trial.suggest_int("momentum_window_s", 30, 600, step=30)
        min_momentum    = trial.suggest_float("min_momentum_pp", 0.0, 20.0)

        if _HAS_VOLUME_DATA:
            min_volume = trial.suggest_float("min_volume", 0.0, 1000.0)
        else:
            min_volume = 0.0

        kelly_fraction  = trial.suggest_float("kelly_fraction", 0.0, 1.0)
        kelly_cap_pct   = trial.suggest_float("kelly_cap_pct", 2.0, 50.0)
        flat_bet_pct    = trial.suggest_float("flat_bet_pct", 1.0, 30.0)

        params = {
            "prob_floor":        prob_floor,
            "prob_ceiling":      prob_ceiling,
            "min_elapsed_s":     min_elapsed,
            "max_elapsed_s":     max_elapsed,
            "momentum_window_s": momentum_window,
            "min_momentum_pp":   min_momentum,
            "min_volume":        min_volume,
            "kelly_fraction":    kelly_fraction,
            "kelly_cap_pct":     kelly_cap_pct,
            "flat_bet_pct":      flat_bet_pct,
        }

        # ── Walk-forward evaluation ────────────────────────────────────────────
        total_profit = 0.0
        total_bets   = 0

        for test_week_num in range(first_test_week, NUM_WEEKS + 1):
            test_idx   = test_week_num - 1
            surface    = build_surface_excluding_week(contributions, test_idx)
            week_games = all_weeks[test_week_num - 1]
            result     = simulate_week(week_games, params, surface)
            total_profit += result["profit"]
            total_bets   += result["bets"]

        # Penalise degenerate "never bet" solutions
        if total_bets < MIN_TOTAL_BETS_REQUIRED:
            return -STARTING_BANKROLL * NUM_WEEKS

        return total_profit

    return objective


# ── Best-params re-run with verbose output ────────────────────────────────────

def run_best_params(
    params: dict,
    all_weeks: list[list[dict]],
    contributions: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Re-run leave-one-week-out evaluation with the best params and print a detailed
    week-by-week breakdown.
    """
    first_test_week = MIN_TRAIN_WEEKS + 1
    total_profit = 0.0
    total_games = total_bets = total_wins = total_losses = 0
    total_opportunities = 0
    total_team_won_bets = 0
    total_sum_return_multiple = 0.0

    print("\n" + "=" * 60)
    print("LEAVE-ONE-WEEK-OUT DETAIL  (best parameter set)")
    print("=" * 60)
    print(
        f"{'Week':>4}  {'Games':>5}  {'Bets':>4}  {'W':>4}  {'L':>4}  "
        f"{'Acc%':>6}  {'ROI/G all%':>10}  {'ROI/G bet%':>10}  {'Profit':>10}"
    )
    print("-" * 60)

    for test_week_num in range(first_test_week, NUM_WEEKS + 1):
        test_idx   = test_week_num - 1
        surface    = build_surface_excluding_week(contributions, test_idx)
        week_games = all_weeks[test_week_num - 1]
        r          = simulate_week(week_games, params, surface)

        win_pct = 100.0 * r["wins"] / r["bets"] if r["bets"] > 0 else 0.0
        roi_per_game_all = (
            (r["profit"] / STARTING_BANKROLL) * 100.0 / r["games"]
            if r["games"] > 0
            else 0.0
        )
        roi_per_game_bet = (
            (r["profit"] / STARTING_BANKROLL) * 100.0 / r["bets"]
            if r["bets"] > 0
            else 0.0
        )
        total_profit += r["profit"]
        total_games  += r["games"]
        total_bets   += r["bets"]
        total_wins   += r["wins"]
        total_losses += r["losses"]
        total_opportunities += r["games_with_opportunity"]
        total_team_won_bets += r["team_won_bets"]
        total_sum_return_multiple += r["sum_return_multiple"]

        print(
            f"{test_week_num:>4}  {r['games']:>5}  {r['bets']:>4}  {r['wins']:>4}  {r['losses']:>4}  "
            f"{win_pct:>5.1f}%  {roi_per_game_all:>9.3f}%  {roi_per_game_bet:>9.3f}%  "
            f"${r['profit']:>9.2f}"
        )

    overall_win_pct = (
        100.0 * total_wins / total_bets if total_bets > 0 else 0.0
    )
    overall_roi_per_game_all = (
        (total_profit / STARTING_BANKROLL) * 100.0 / total_games
        if total_games > 0
        else 0.0
    )
    overall_roi_per_game_bet = (
        (total_profit / STARTING_BANKROLL) * 100.0 / total_bets
        if total_bets > 0
        else 0.0
    )
    print("-" * 60)
    print(
        f"{'TOT':>4}  {total_games:>5}  {total_bets:>4}  {total_wins:>4}  {total_losses:>4}  "
        f"{overall_win_pct:>5.1f}%  {overall_roi_per_game_all:>9.3f}%  "
        f"{overall_roi_per_game_bet:>9.3f}%  ${total_profit:>9.2f}"
    )

    n_test_weeks = NUM_WEEKS - first_test_week + 1
    avg_profit = total_profit / n_test_weeks if n_test_weeks > 0 else 0
    roi = (total_profit / (STARTING_BANKROLL * n_test_weeks)) * 100

    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Test weeks evaluated : {n_test_weeks}")
    print(f"  Total games observed : {total_games}")
    print(f"  Total bets placed    : {total_bets}")
    print(f"  Overall win rate     : {overall_win_pct:.1f}%")
    print(f"  Avg ROI per game (all games) : {overall_roi_per_game_all:.3f}%")
    print(f"  Avg ROI per game (bets only) : {overall_roi_per_game_bet:.3f}%")
    print(f"  Total profit         : ${total_profit:.2f}")
    print(f"  Average weekly profit: ${avg_profit:.2f}")
    print(f"  ROI on bankroll      : {roi:.2f}%")

    print("\n── Bet-level summary (all test weeks, placed bets only where noted) ──")
    print(
        f"  {total_opportunities} of {total_games}  games had a betting opportunity "
        f"(qualifying signal before bankroll / timing skips)."
    )
    if total_bets > 0:
        pick_win_pct = 100.0 * total_team_won_bets / total_bets
        avg_mult = total_sum_return_multiple / total_bets
        print(
            f"  {pick_win_pct:.1f}%  of placed bets: team we bet on won the game."
        )
        print(
            f"  {avg_mult:.2f}x  average cash return multiple per bet "
            f"(payout ÷ total cost, counting losses as 0 payout → 0x in that bet’s term)."
        )
    else:
        print("  (No placed bets — win rate and return multiple N/A.)")


# ── Main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian hyper-parameter search for the Kalshi NCAAB strategy."
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Short Optuna run (40 trials, 4 workers) for quick iteration.",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=None,
        metavar="N",
        help=f"Optuna trials (default: {N_OPTUNA_TRIALS}, or 40 with --fast if not set).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=f"Parallel Optuna workers (default: {N_OPTUNA_WORKERS}, or 4 with --fast if not set).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n_trials = args.trials
    n_workers = args.workers
    if args.fast:
        n_trials = n_trials if n_trials is not None else 40
        n_workers = n_workers if n_workers is not None else 4
    else:
        n_trials = n_trials if n_trials is not None else N_OPTUNA_TRIALS
        n_workers = n_workers if n_workers is not None else N_OPTUNA_WORKERS
    if n_trials < 1:
        raise SystemExit("--trials must be >= 1")
    if n_workers < 1:
        raise SystemExit("--workers must be >= 1")

    _script_start = time.perf_counter()
    print("=" * 60)
    print("Ben's Kalshi NCAAB Betting Strategy Optimiser (with modifications by Will)")
    print("=" * 60)
    if args.fast or args.trials is not None or args.workers is not None:
        print(f"  Optuna: {n_trials} trials, {n_workers} workers")

    # 1. Load data
    print("\n[1/4] Loading weekly data files …")
    all_weeks = load_and_preprocess_weeks(DATA_DIRECTORY)

    # 2. Precompute per-week calibration surface contributions
    print("\n[2/4] Precomputing calibration surface contributions …")
    contributions = precompute_week_contributions(all_weeks)
    print("  Done.")

    # 3. Run Bayesian optimisation
    print(f"\n[3/4] Running Optuna optimisation ({n_trials} trials, {n_workers} workers) …")
    print("  (One line per trial as each worker finishes; completion order can be out of order.)\n")

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = make_objective(all_weeks, contributions)

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        finished = sum(1 for t in study.trials if t.state.is_finished())
        thr = threading.current_thread().name
        if trial.duration is not None:
            dur_txt = f"{trial.duration.total_seconds():.1f}s"
        else:
            dur_txt = "?"

        if trial.value is not None:
            obj_txt = f"${trial.value:.2f}"
        else:
            obj_txt = trial.state.name

        try:
            best_txt = f"${study.best_value:.2f}"
        except ValueError:
            best_txt = "n/a"

        print(
            f"  [{finished:>{len(str(n_trials))}}/{n_trials}] "
            f"trial #{trial.number:<4}  {trial.state.name:<9}  "
            f"objective={obj_txt:<12}  best={best_txt:<12}  "
            f"({dur_txt})  {thr}",
            flush=True,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_workers,
        callbacks=[_callback],
    )

    # 4. Report results
    print("\n[4/4] Optimisation complete.\n")
    print("=" * 60)
    print("BEST PARAMETERS FOUND")
    print("=" * 60)

    best = study.best_params
    best_val = study.best_value

    print(f"  Optimised total profit (test weeks 3–19): ${best_val:.2f}\n")
    print("  ── Copy these into evaluator.py ────────────────────────")
    print(f"  MIN_INCLUSIVE_KALSHI_PROBABILITY_TO_BET = {best['prob_floor']:.1f}")
    print(f"  # prob_ceiling (skip bets above)        = {best['prob_ceiling']:.1f}")
    print(f"  MIN_GAME_ELAPSED_SECONDS_TO_BET         = {best['min_elapsed_s']}")
    print(f"  # max_elapsed_s (stop betting after)    = {best['max_elapsed_s']}")
    print(f"  # momentum_window_s                     = {best['momentum_window_s']}")
    print(f"  # min_momentum_pp                       = {best['min_momentum_pp']:.2f}")
    if _HAS_VOLUME_DATA:
        print(f"  # min_volume                            = {best['min_volume']:.1f}")
    else:
        print(f"  # min_volume                            = N/A (no volume data)")
    print(f"  # kelly_fraction  (0 = flat sizing)     = {best['kelly_fraction']:.3f}")
    print(f"  # kelly_cap_pct   (% of bankroll cap)   = {best['kelly_cap_pct']:.1f}")
    print(f"  # flat_bet_pct    (% bankroll flat bet)  = {best['flat_bet_pct']:.1f}")
    print()
    print("  Full param dict for direct use in simulate_week():")
    print(f"  {best}")

    # Detailed walk-forward breakdown with best params
    run_best_params(best, all_weeks, contributions)

    # Also print top-5 trials for comparison
    print("\n── Top 5 trials ────────────────────────────────────────")
    trials_df = study.trials_dataframe().sort_values("value", ascending=False).head(5)
    for _, row in trials_df.iterrows():
        print(
            f"  Trial {int(row['number']):>4} | profit ${row['value']:>9.2f} | "
            f"floor={row['params_prob_floor']:.1f}  "
            f"ceil={row['params_prob_ceiling']:.1f}  "
            f"min_el={int(row['params_min_elapsed_s'])}  "
            f"kelly={row['params_kelly_fraction']:.2f}"
        )

    elapsed = time.perf_counter() - _script_start
    if elapsed >= 60:
        mins, secs = divmod(elapsed, 60)
        print(f"\n── Total wall time: {int(mins)}m {secs:.1f}s ({elapsed:.2f}s)")
    else:
        print(f"\n── Total wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
