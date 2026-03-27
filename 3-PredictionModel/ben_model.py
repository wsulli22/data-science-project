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

Bankroll: $1,000 per week (resets each week, treated independently).

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
  python optimizer.py

  The script expects week_1_games.csv … week_19_games.csv to live in a
  subdirectory called "GeneratedDataFiles/" relative to this file (same layout as evaluator.py).

OUTPUT
------
  Prints a per-week walk-forward summary for the best trial, then the
  optimal parameter set, then reruns that set explicitly so you can
  copy the parameters straight into evaluator.py.
"""

from __future__ import annotations

import math
import warnings
from datetime import timedelta
from heapq import heappop, heappush
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── User-adjustable configuration ─────────────────────────────────────────────

DATA_DIRECTORY = Path(__file__).resolve().parent / "GeneratedDataFiles"

NUM_WEEKS = 19
STARTING_BANKROLL = 1_000.0           # dollars, resets each week
SETTLEMENT_BUFFER_SECONDS = 2 * 60 * 60
TAKER_FEE_RATE = 0.07

# Optimisation
N_OPTUNA_TRIALS = 500                  # increase for more thorough search
OPTUNA_SEED = 42
MIN_TRAIN_WEEKS = 2                    # need at least this many weeks before testing

# Degenerate-solution guard: penalise if fewer bets made across all test weeks
MIN_TOTAL_BETS_REQUIRED = 10

# Calibration surface resolution
# Edges are right-exclusive except the last bucket which catches everything above
PROB_BIN_EDGES  = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 101]
TIME_BIN_EDGES  = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 3600, 99999]

N_PROB_BINS = len(PROB_BIN_EDGES)
N_TIME_BINS = len(TIME_BIN_EDGES)


# ── Fee helpers (mirror helper_functions.py exactly) ──────────────────────────

def _round_up_cent(x: float) -> float:
    return math.ceil(max(0.0, x) * 100) / 100


def _taker_fee(price: float, contracts: int) -> float:
    p = min(1.0, max(0.0, price))
    return _round_up_cent(TAKER_FEE_RATE * contracts * p * (1.0 - p))


def _buy(target_dollars: float, contract_price: float) -> tuple[float, float, int]:
    """
    Returns (actual_stake, fee, contract_count).
    Mirrors the buy() function in helper_functions.py.
    """
    price = min(1.0, max(0.01, contract_price))
    contracts = int(math.floor(target_dollars / price))
    if contracts <= 0:
        return 0.0, 0.0, 0
    stake = contracts * price
    fee = _taker_fee(price, contracts)
    return stake, fee, contracts


# ── Data loading and preprocessing ────────────────────────────────────────────

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
    all_weeks: list[list[dict]] = []

    for w in range(1, NUM_WEEKS + 1):
        path = data_dir / f"week_{w}_games.csv"
        df = pd.read_csv(path)
        df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"])
        df = df.sort_values(
            ["kalshi_event", "realworld_timestamp"], kind="mergesort"
        ).reset_index(drop=True)

        vol_col = _detect_volume_column(df)
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
    4. Probability has risen by >= min_momentum_pp over the last momentum_window_s rows
       (rows ≈ seconds in the per-second data)
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
    win_s   = int(params["momentum_window_s"])
    min_mom = params["min_momentum_pp"]
    min_vol = params["min_volume"]

    # ── 1. Clock gate ──────────────────────────────────────────────────────────
    clock_ok = (elapsed >= min_el) & (elapsed <= max_el)

    # ── 2. Threshold crossing (upward) ────────────────────────────────────────
    # Treat the very first row as crossing from 0 (allows betting immediately if
    # conditions are met and prob already above floor at game start).
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
    # A crossing is valid if the crossing team is also below the ceiling.
    # When both cross: at least one must be below ceiling.
    ceiling_ok = (
        (t1_cross & t1_ceil_ok)
        | (t2_cross & t2_ceil_ok)
    )

    # ── 4. Momentum filter ─────────────────────────────────────────────────────
    if min_mom > 0.0 and win_s > 0:
        p1_past = np.empty(n)
        p2_past = np.empty(n)
        p1_past[:win_s] = np.nan
        p2_past[:win_s] = np.nan
        p1_past[win_s:] = p1[:n - win_s]
        p2_past[win_s:] = p2[:n - win_s]

        # nan comparisons evaluate to False in numpy — correct behaviour here
        with np.errstate(invalid="ignore"):
            p1_mom_ok = (p1 - p1_past) >= min_mom
            p2_mom_ok = (p2 - p2_past) >= min_mom

        # Valid only if the crossing team has sufficient momentum
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
        # Determine which team to bet on
        t1_ok = bool(t1_cross[i]) and bool(t1_ceil_ok[i])
        t2_ok = bool(t2_cross[i]) and bool(t2_ceil_ok[i])

        # Choose higher-probability team when both qualify
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
    # Collect candidate bets (one per game)
    candidates: list[dict] = []
    for game in week_games:
        bet = find_first_bet(game, params, surface)
        if bet is not None:
            candidates.append(bet)

    # Sort by bet time (ascending) to respect bankroll FIFO ordering
    candidates.sort(key=lambda b: (b["bet_ts"], b["event_id"]))

    bankroll = STARTING_BANKROLL
    pending: list[tuple] = []
    total_bets = wins = losses = skipped = 0
    total_profit = 0.0

    for bet in candidates:
        # Release matured settlements before this bet's timestamp
        while pending and pending[0][0] <= bet["bet_ts"]:
            _, payout, _ = heappop(pending)
            bankroll += payout

        stake, fee, contracts = _buy(bet["bet_size"], bet["prob_pct"] / 100.0)
        total_cost = stake + fee

        if contracts <= 0 or total_cost > bankroll:
            skipped += 1
            continue

        bankroll -= total_cost
        payout = float(contracts) if bet["bet_team"] == bet["winner"] else 0.0
        release_ts = bet["game_end_ts"] + timedelta(seconds=SETTLEMENT_BUFFER_SECONDS)
        heappush(pending, (release_ts, payout, bet["event_id"]))

        profit = payout - total_cost
        total_profit += profit
        total_bets += 1
        if profit >= 0:
            wins += 1
        else:
            losses += 1

    # Flush all remaining settlements
    while pending:
        _, payout, _ = heappop(pending)
        bankroll += payout

    return {
        "profit":      bankroll - STARTING_BANKROLL,
        "bets":        total_bets,
        "wins":        wins,
        "losses":      losses,
        "skipped":     skipped,
        "final_bank":  bankroll,
    }


# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(
    all_weeks: list[list[dict]],
    contributions: list[tuple[np.ndarray, np.ndarray]],
):
    """
    Returns a closure that Optuna calls for each trial.

    Walk-forward schedule:
      - Test week 3  → train on weeks 1–2
      - Test week 4  → train on weeks 1–3
      - …
      - Test week 19 → train on weeks 1–18
    """
    first_test_week = MIN_TRAIN_WEEKS + 1   # 1-indexed

    def objective(trial: optuna.Trial) -> float:
        # ── Sample parameters ──────────────────────────────────────────────────
        prob_floor = trial.suggest_float("prob_floor", 55.0, 92.0)

        # Ceiling must be strictly above floor; cap so range is always valid
        ceil_lo = min(prob_floor + 3.0, 98.0)
        prob_ceiling = trial.suggest_float("prob_ceiling", ceil_lo, 99.0)

        min_elapsed = trial.suggest_int("min_elapsed_s", 0, 2400, step=60)

        # Max elapsed must be above min; cap at 4800 (covers ~2 OT periods)
        max_el_lo = min(min_elapsed + 300, 4800)
        max_elapsed = trial.suggest_int("max_elapsed_s", max_el_lo, 4800, step=60)

        momentum_window = trial.suggest_int("momentum_window_s", 30, 600, step=30)
        min_momentum    = trial.suggest_float("min_momentum_pp", 0.0, 20.0)
        min_volume      = trial.suggest_float("min_volume", 0.0, 1000.0)

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
            train_idx  = test_week_num - 1          # weeks 0..(train_idx-1) are training
            surface    = build_surface(contributions, train_idx)
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
    Re-run walk-forward evaluation with the best params and print a detailed
    week-by-week breakdown.
    """
    first_test_week = MIN_TRAIN_WEEKS + 1
    total_profit = 0.0
    total_bets = total_wins = total_losses = 0

    print("\n" + "=" * 60)
    print("WALK-FORWARD DETAIL  (best parameter set)")
    print("=" * 60)
    print(
        f"{'Week':>4}  {'Bets':>4}  {'W':>4}  {'L':>4}  "
        f"{'Profit':>10}  {'End Bank':>10}  {'Win%':>6}"
    )
    print("-" * 60)

    for test_week_num in range(first_test_week, NUM_WEEKS + 1):
        train_idx  = test_week_num - 1
        surface    = build_surface(contributions, train_idx)
        week_games = all_weeks[test_week_num - 1]
        r          = simulate_week(week_games, params, surface)

        win_pct = 100.0 * r["wins"] / r["bets"] if r["bets"] > 0 else float("nan")
        total_profit += r["profit"]
        total_bets   += r["bets"]
        total_wins   += r["wins"]
        total_losses += r["losses"]

        print(
            f"{test_week_num:>4}  {r['bets']:>4}  {r['wins']:>4}  {r['losses']:>4}  "
            f"${r['profit']:>9.2f}  ${r['final_bank']:>9.2f}  {win_pct:>5.1f}%"
        )

    overall_win_pct = (
        100.0 * total_wins / total_bets if total_bets > 0 else float("nan")
    )
    print("-" * 60)
    print(
        f"{'TOT':>4}  {total_bets:>4}  {total_wins:>4}  {total_losses:>4}  "
        f"${total_profit:>9.2f}  {'':>10}  {overall_win_pct:>5.1f}%"
    )

    n_test_weeks = NUM_WEEKS - first_test_week + 1
    avg_profit = total_profit / n_test_weeks if n_test_weeks > 0 else 0
    roi = (total_profit / (STARTING_BANKROLL * n_test_weeks)) * 100

    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Test weeks evaluated : {n_test_weeks}")
    print(f"  Total bets placed    : {total_bets}")
    print(f"  Overall win rate     : {overall_win_pct:.1f}%")
    print(f"  Total profit         : ${total_profit:.2f}")
    print(f"  Average weekly profit: ${avg_profit:.2f}")
    print(f"  ROI on bankroll      : {roi:.2f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Kalshi NCAAB Betting Strategy Optimiser")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading weekly data files …")
    all_weeks = load_and_preprocess_weeks(DATA_DIRECTORY)

    # 2. Precompute per-week calibration surface contributions
    print("\n[2/4] Precomputing calibration surface contributions …")
    contributions = precompute_week_contributions(all_weeks)
    print("  Done.")

    # 3. Run Bayesian optimisation
    print(f"\n[3/4] Running Optuna optimisation ({N_OPTUNA_TRIALS} trials) …")
    print("  (This may take several minutes — progress shown every 50 trials)\n")

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = make_objective(all_weeks, contributions)

    # Progress callback
    def _callback(study: optuna.Study, trial: optuna.Trial) -> None:
        if (trial.number + 1) % 50 == 0:
            print(
                f"  Trial {trial.number + 1:>4}/{N_OPTUNA_TRIALS} | "
                f"best profit so far: ${study.best_value:.2f}"
            )

    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, callbacks=[_callback])

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
    print(f"  # min_volume                            = {best['min_volume']:.1f}")
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


if __name__ == "__main__":
    main()