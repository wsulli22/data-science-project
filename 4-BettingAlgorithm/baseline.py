"""
baseline.py — Simple "always bet the pre-game favorite" benchmark.

PURPOSE
-------
Provides a straightforward gambling-style benchmark to compare against the
targeted strategy in ``algorithm_with_testing.py``. For every game we:

  1. Open a position on the team with the higher win probability at the first
     available timestamp (the pre-game / tip-off favorite).
  2. Pay Kalshi taker buy-side fees (identical to the main algorithm).
  3. Hold to settlement — no selling, no cash-out penalty (mirrors the main
     algorithm, which is also buy-and-hold).

Two bet-sizing variants are evaluated side-by-side:

  * FIXED    — flat ``FIXED_BET_PCT`` % of the starting bankroll per game.
  * DYNAMIC  — risk-adjusted half-Kelly sizing that uses the leave-one-week-out
               empirical win-rate surface (same surface the main algorithm uses)
               to estimate the favorite's true win probability, then sizes the
               bet proportional to the edge. Falls back to the fixed bet when
               the surface shows no positive edge.

Testing follows the same leave-one-week-out walk-forward schedule as
``algorithm_with_testing.py`` (test weeks 3–19). Bankroll is $1000 and resets
each week; bets settle with a 2-hour buffer after the final timestamp.

USAGE
-----
    python baseline.py
"""

from __future__ import annotations

import time
from datetime import timedelta
from heapq import heappop, heappush
from typing import Optional

import numpy as np
import pandas as pd

from algorithm_with_testing import (
    DATA_DIRECTORY,
    MIN_TRAIN_WEEKS,
    NUM_WEEKS,
    SETTLEMENT_BUFFER_SECONDS,
    STARTING_BANKROLL,
    _buy,
    _usd2,
    build_surface_excluding_week,
    load_and_preprocess_weeks,
    lookup_true_win_prob,
    precompute_week_contributions,
)


# ── Baseline-specific bet-sizing knobs ────────────────────────────────────────

FIXED_BET_PCT  = 5.0     # % of starting bankroll per bet in the fixed variant
KELLY_FRACTION = 0.5     # half-Kelly for the dynamic variant
KELLY_CAP_PCT  = 20.0    # hard ceiling on bet size (% of starting bankroll)


# ── Pick the pre-game favorite for one game ───────────────────────────────────

def pick_starting_favorite(game: dict) -> Optional[dict]:
    """Return a bet dict for the first timestamp of the game.

    The favored team is the one with the higher probability at that timestamp
    (ties break toward team_1). Returns None if the game has no rows.
    """
    p1 = game["p1"]
    p2 = game["p2"]
    if len(p1) == 0:
        return None

    p1_0 = float(p1[0])
    p2_0 = float(p2[0])
    elapsed_0 = float(game["elapsed"][0])
    ts_0 = pd.Timestamp(game["ts"][0])

    if p1_0 >= p2_0:
        bet_team, bet_prob = game["team_1"], p1_0
    else:
        bet_team, bet_prob = game["team_2"], p2_0

    return {
        "event_id":    game["event_id"],
        "bet_ts":      ts_0,
        "game_end_ts": pd.Timestamp(game["game_end"]),
        "bet_team":    bet_team,
        "prob_pct":    bet_prob,
        "winner":      game["winner"],
        "elapsed_s":   elapsed_0,
    }


# ── Weekly simulator ──────────────────────────────────────────────────────────

def simulate_week_baseline(
    week_games: list[dict],
    mode: str,
    surface: Optional[np.ndarray] = None,
) -> dict:
    """Simulate one week of the baseline strategy.

    Parameters
    ----------
    week_games : list[dict]
        Output of ``load_and_preprocess_weeks`` for a single week.
    mode : {"fixed", "dynamic"}
        Bet-sizing rule.
    surface : np.ndarray or None
        Required when ``mode == "dynamic"``; supplies p_true estimates for Kelly.
    """
    candidates: list[dict] = []
    for game in week_games:
        bet = pick_starting_favorite(game)
        if bet is not None:
            candidates.append(bet)

    candidates.sort(key=lambda b: (b["bet_ts"], b["event_id"]))

    bankroll = STARTING_BANKROLL
    pending: list[tuple] = []
    total_bets = wins = losses = skipped = 0
    total_profit = 0.0
    team_won_bets = 0
    sum_return_multiple = 0.0
    fixed_target = (FIXED_BET_PCT / 100.0) * STARTING_BANKROLL
    kelly_cap = (KELLY_CAP_PCT / 100.0) * STARTING_BANKROLL

    for bet in candidates:
        while pending and pending[0][0] <= bet["bet_ts"]:
            _, payout, _ = heappop(pending)
            bankroll += payout

        p_kalshi_frac = bet["prob_pct"] / 100.0

        if mode == "fixed":
            target = fixed_target
        elif mode == "dynamic":
            target = fixed_target  # fallback when no edge is detected
            if surface is not None and p_kalshi_frac < 1.0:
                p_true = lookup_true_win_prob(
                    surface, bet["prob_pct"], bet["elapsed_s"]
                )
                if p_true is not None and p_true > p_kalshi_frac:
                    edge = p_true - p_kalshi_frac
                    f_star = edge / (1.0 - p_kalshi_frac)
                    target = KELLY_FRACTION * f_star * STARTING_BANKROLL
            target = min(target, kelly_cap)
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        if target <= 0.0:
            skipped += 1
            continue

        stake, fee, contracts = _buy(target, p_kalshi_frac)
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
        "games":               len(week_games),
        "profit":              bankroll - STARTING_BANKROLL,
        "bets":                total_bets,
        "wins":                wins,
        "losses":              losses,
        "skipped":             skipped,
        "final_bank":          bankroll,
        "team_won_bets":       team_won_bets,
        "sum_return_multiple": sum_return_multiple,
    }


# ── Leave-one-week-out driver ─────────────────────────────────────────────────

def run_baseline(
    mode: str,
    all_weeks: list[list[dict]],
    contributions: Optional[list[tuple[np.ndarray, np.ndarray]]],
) -> dict:
    """Run test weeks 3..19 and print a per-week breakdown."""
    first_test_week = MIN_TRAIN_WEEKS + 1

    total_profit = 0.0
    total_games = total_bets = total_wins = total_losses = 0
    total_team_won_bets = 0
    total_sum_return_multiple = 0.0

    label = "FIXED" if mode == "fixed" else "DYNAMIC (Kelly)"
    print("\n" + "=" * 72)
    print(f"BASELINE — always bet starting favorite [{label}]")
    print("=" * 72)
    print(
        f"{'Week':>4}  {'Games':>5}  {'Bets':>4}  {'W':>4}  {'L':>4}  "
        f"{'Acc%':>6}  {'ROI/G all%':>10}  {'ROI/G bet%':>10}  {'Profit':>10}"
    )
    print("-" * 72)

    for test_week_num in range(first_test_week, NUM_WEEKS + 1):
        test_idx = test_week_num - 1
        if mode == "dynamic":
            assert contributions is not None, "dynamic mode needs contributions"
            surface = build_surface_excluding_week(contributions, test_idx)
        else:
            surface = None
        week_games = all_weeks[test_idx]
        r = simulate_week_baseline(week_games, mode, surface)

        win_pct = 100.0 * r["wins"] / r["bets"] if r["bets"] > 0 else 0.0
        roi_all = (
            (r["profit"] / STARTING_BANKROLL) * 100.0 / r["games"]
            if r["games"] > 0 else 0.0
        )
        roi_bet = (
            (r["profit"] / STARTING_BANKROLL) * 100.0 / r["bets"]
            if r["bets"] > 0 else 0.0
        )

        total_profit += r["profit"]
        total_games  += r["games"]
        total_bets   += r["bets"]
        total_wins   += r["wins"]
        total_losses += r["losses"]
        total_team_won_bets += r["team_won_bets"]
        total_sum_return_multiple += r["sum_return_multiple"]

        print(
            f"{test_week_num:>4}  {r['games']:>5}  {r['bets']:>4}  {r['wins']:>4}  {r['losses']:>4}  "
            f"{win_pct:>5.1f}%  {roi_all:>9.3f}%  {roi_bet:>9.3f}%  ${r['profit']:>9.2f}"
        )

    overall_win_pct = 100.0 * total_wins / total_bets if total_bets else 0.0
    overall_roi_all = (
        (total_profit / STARTING_BANKROLL) * 100.0 / total_games
        if total_games else 0.0
    )
    overall_roi_bet = (
        (total_profit / STARTING_BANKROLL) * 100.0 / total_bets
        if total_bets else 0.0
    )
    print("-" * 72)
    print(
        f"{'TOT':>4}  {total_games:>5}  {total_bets:>4}  {total_wins:>4}  {total_losses:>4}  "
        f"{overall_win_pct:>5.1f}%  {overall_roi_all:>9.3f}%  {overall_roi_bet:>9.3f}%  "
        f"${total_profit:>9.2f}"
    )

    n_test_weeks = NUM_WEEKS - first_test_week + 1
    avg_profit = total_profit / n_test_weeks if n_test_weeks else 0.0
    roi_bank = (total_profit / (STARTING_BANKROLL * n_test_weeks)) * 100 if n_test_weeks else 0.0

    rule = (
        f"flat {FIXED_BET_PCT:.1f}% of starting bankroll"
        if mode == "fixed"
        else f"{KELLY_FRACTION:.2f}-Kelly via LOO surface, capped at {KELLY_CAP_PCT:.1f}%"
    )

    print(f"\n── {label} baseline summary ─────────────────────────────")
    print(f"  Bet-size rule        : {rule}")
    print(f"  Test weeks evaluated : {n_test_weeks}")
    print(f"  Total games observed : {total_games}")
    print(f"  Total bets placed    : {total_bets}")
    print(f"  Overall win rate     : {overall_win_pct:.1f}%")
    print(f"  Avg ROI per game (all games) : {overall_roi_all:.3f}%")
    print(f"  Avg ROI per game (bets only) : {overall_roi_bet:.3f}%")
    print(f"  Total profit         : ${total_profit:.2f}")
    print(f"  Average weekly profit: ${avg_profit:.2f}")
    print(f"  ROI on bankroll      : {roi_bank:.2f}%")
    if total_bets > 0:
        avg_mult = total_sum_return_multiple / total_bets
        pick_win_pct = 100.0 * total_team_won_bets / total_bets
        print(f"  Pick win rate        : {pick_win_pct:.1f}%")
        print(f"  Avg return multiple  : {avg_mult:.2f}x")

    return {
        "mode":               mode,
        "label":              label,
        "rule":               rule,
        "total_profit":       total_profit,
        "total_bets":         total_bets,
        "total_games":        total_games,
        "wins":               total_wins,
        "losses":             total_losses,
        "win_pct":            overall_win_pct,
        "roi_pct":            roi_bank,
        "avg_weekly_profit":  avg_profit,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()
    print("=" * 72)
    print("BASELINE EVALUATOR — bet the pre-game favorite on every game")
    print("=" * 72)

    print("\n[1/3] Loading weekly data files …")
    all_weeks = load_and_preprocess_weeks(DATA_DIRECTORY)

    print("\n[2/3] Precomputing calibration surface contributions …")
    contributions = precompute_week_contributions(all_weeks)
    print("  Done.")

    print("\n[3/3] Running baseline strategies (leave-one-week-out, weeks 3–19) …")
    fixed_res   = run_baseline("fixed",   all_weeks, contributions=None)
    dynamic_res = run_baseline("dynamic", all_weeks, contributions=contributions)

    print("\n" + "=" * 72)
    print("HEAD-TO-HEAD COMPARISON (baseline variants)")
    print("=" * 72)
    print(
        f"  {'Variant':<28}  {'Bets':>5}  {'Win%':>6}  {'Profit':>11}  {'ROI':>7}"
    )
    print("  " + "-" * 64)
    for r in (fixed_res, dynamic_res):
        nice_label = (
            f"Fixed  ({FIXED_BET_PCT:.1f}% flat)"
            if r["mode"] == "fixed"
            else f"Dynamic ({KELLY_FRACTION:.2f}-Kelly)"
        )
        print(
            f"  {nice_label:<28}  {r['total_bets']:>5}  {r['win_pct']:>5.1f}%  "
            f"${r['total_profit']:>9.2f}  {r['roi_pct']:>6.2f}%"
        )

    elapsed = time.perf_counter() - t0
    if elapsed >= 60:
        mins, secs = divmod(elapsed, 60)
        print(f"\n── Total wall time: {int(mins)}m {secs:.1f}s ({elapsed:.2f}s)")
    else:
        print(f"\n── Total wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
