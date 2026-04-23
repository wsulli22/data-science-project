"""
baseline.py — Benchmark: always bet the team favored at game start (opening line).

Uses the same weekly CSVs, fee model, settlement buffer, and leave-one-week-out
test weeks (3–19) as ``algorithm_with_testing.py``.

Two variants for comparison with the targeted strategy:
  * **Fixed** — each intended bet uses a flat percentage of the *starting* weekly
    bankroll (default 10%), subject to the usual per-bet budget / fee sizing.
  * **Dynamic** — fractional Kelly sizing from the empirical win-rate surface
    (all training weeks except the test week), capped as a % of starting
    bankroll, using the same ``compute_bet_size`` logic as the optimiser (positive
    edge required when a calibrated ``p_true`` is available). By default
    ``flat_bet_pct`` is 0 in this mode so games with no ``p_true`` are skipped;
    pass ``--dynamic-flat-fallback-pct 10`` to allow a flat stake when the surface
    lookup cannot supply ``p_true``.

Run from ``4-BettingAlgorithm/``:
  python baseline.py
  python baseline.py --fixed-flat-pct 10 --dynamic-kelly 0.5 --dynamic-cap 20
"""

from __future__ import annotations

import argparse
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
    STARTING_BANKROLL,
    SETTLEMENT_BUFFER_SECONDS,
    _buy,
    _usd2,
    build_surface_excluding_week,
    compute_bet_size,
    load_and_preprocess_weeks,
    lookup_true_win_prob,
    precompute_week_contributions,
)


def opening_favorite_bet(
    game: dict,
    surface: Optional[np.ndarray],
    params: dict,
) -> Optional[dict]:
    """
    One bet per game: the team with higher Kalshi win % at the earliest observed
    game-clock row (minimum ``game_elapsed_seconds``). True 50–50 openings skip.
    """
    p1 = game["p1"]
    p2 = game["p2"]
    elapsed = game["elapsed"]
    n = len(p1)
    if n == 0:
        return None

    i = int(np.argmin(elapsed))
    t1p, t2p = float(p1[i]), float(p2[i])

    if t1p > t2p:
        bet_team, bet_prob = game["team_1"], t1p
    elif t2p > t1p:
        bet_team, bet_prob = game["team_2"], t2p
    else:
        return None

    p_kalshi_frac = bet_prob / 100.0
    p_true: Optional[float] = None
    if params["kelly_fraction"] > 0.0 and surface is not None:
        p_true = lookup_true_win_prob(surface, bet_prob, float(elapsed[i]))

    bet_size = compute_bet_size(p_true, p_kalshi_frac, params)
    if bet_size <= 0.0:
        return None

    return {
        "event_id": game["event_id"],
        "bet_ts": pd.Timestamp(game["ts"][i]),
        "game_end_ts": pd.Timestamp(game["game_end"]),
        "bet_team": bet_team,
        "prob_pct": bet_prob,
        "winner": game["winner"],
        "p_true": p_true,
        "bet_size": bet_size,
    }


def simulate_week_baseline(
    week_games: list[dict],
    surface: Optional[np.ndarray],
    params: dict,
) -> dict:
    """
    Same FIFO bankroll + 2-hour settlement queue as ``simulate_week`` in the
    main optimiser, but every game uses at most one opening-favorite bet.
    """
    candidates: list[dict] = []
    for game in week_games:
        bet = opening_favorite_bet(game, surface, params)
        if bet is not None:
            candidates.append(bet)

    candidates.sort(key=lambda b: (b["bet_ts"], b["event_id"]))

    bankroll = STARTING_BANKROLL
    pending: list[tuple] = []
    total_bets = wins = losses = skipped = 0
    total_profit = 0.0
    team_won_bets = 0
    sum_return_multiple = 0.0

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
        "games": len(week_games),
        "games_with_opportunity": len(candidates),
        "profit": bankroll - STARTING_BANKROLL,
        "bets": total_bets,
        "wins": wins,
        "losses": losses,
        "skipped": skipped,
        "final_bank": bankroll,
        "team_won_bets": team_won_bets,
        "sum_return_multiple": sum_return_multiple,
    }


def _params_fixed(flat_bet_pct: float) -> dict:
    return {
        "kelly_fraction": 0.0,
        "kelly_cap_pct": 100.0,
        "flat_bet_pct": flat_bet_pct,
    }


def _params_dynamic(kelly_fraction: float, kelly_cap_pct: float, flat_bet_pct: float) -> dict:
    return {
        "kelly_fraction": kelly_fraction,
        "kelly_cap_pct": kelly_cap_pct,
        "flat_bet_pct": flat_bet_pct,
    }


def run_leave_one_week_out(
    label: str,
    all_weeks: list[list[dict]],
    contributions: list,
    params: dict,
    bet_rule_line: str,
) -> dict:
    first_test_week = MIN_TRAIN_WEEKS + 1
    total_profit = 0.0
    total_games = total_bets = total_wins = total_losses = 0
    total_team_won_bets = 0
    total_sum_return_multiple = 0.0
    per_week: list[tuple[int, dict]] = []

    print(f"\n{'=' * 72}")
    print(f"BASELINE — always bet starting favorite [{label}]")
    print(f"{'=' * 72}")
    print(
        f"{'Week':>4}  {'Games':>5}  {'Bets':>4}  {'W':>4}  {'L':>4}  "
        f"{'Acc%':>6}  {'ROI/G all%':>10}  {'ROI/G bet%':>10}  {'Profit':>10}"
    )
    print("-" * 72)

    for test_week_num in range(first_test_week, NUM_WEEKS + 1):
        test_idx = test_week_num - 1
        surface = build_surface_excluding_week(contributions, test_idx)
        week_games = all_weeks[test_week_num - 1]
        r = simulate_week_baseline(week_games, surface, params)

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
        total_games += r["games"]
        total_bets += r["bets"]
        total_wins += r["wins"]
        total_losses += r["losses"]
        total_team_won_bets += r["team_won_bets"]
        total_sum_return_multiple += r["sum_return_multiple"]
        per_week.append((test_week_num, r))

        print(
            f"{test_week_num:>4}  {r['games']:>5}  {r['bets']:>4}  {r['wins']:>4}  {r['losses']:>4}  "
            f"{win_pct:>5.1f}%  {roi_per_game_all:>9.3f}%  {roi_per_game_bet:>9.3f}%  "
            f"${r['profit']:>9.2f}"
        )

    overall_win_pct = 100.0 * total_wins / total_bets if total_bets > 0 else 0.0
    overall_roi_all = (
        (total_profit / STARTING_BANKROLL) * 100.0 / total_games
        if total_games > 0
        else 0.0
    )
    overall_roi_bet = (
        (total_profit / STARTING_BANKROLL) * 100.0 / total_bets
        if total_bets > 0
        else 0.0
    )

    print("-" * 72)
    print(
        f"{'TOT':>4}  {total_games:>5}  {total_bets:>4}  {total_wins:>4}  {total_losses:>4}  "
        f"{overall_win_pct:>5.1f}%  {overall_roi_all:>9.3f}%  {overall_roi_bet:>9.3f}%  "
        f"${total_profit:>9.2f}"
    )

    n_test_weeks = NUM_WEEKS - first_test_week + 1
    avg_profit = total_profit / n_test_weeks if n_test_weeks > 0 else 0.0
    roi_bankroll = (total_profit / (STARTING_BANKROLL * n_test_weeks)) * 100.0
    pick_win_pct = 100.0 * total_team_won_bets / total_bets if total_bets > 0 else 0.0
    avg_mult = total_sum_return_multiple / total_bets if total_bets > 0 else 0.0

    print(f"\n── {label} baseline summary ─────────────────────────────")
    print(f"  Bet-size rule        : {bet_rule_line}")
    print(f"  Test weeks evaluated : {n_test_weeks}")
    print(f"  Total games observed : {total_games}")
    print(f"  Total bets placed    : {total_bets}")
    print(f"  Overall win rate     : {overall_win_pct:.1f}%")
    print(f"  Avg ROI per game (all games) : {overall_roi_all:.3f}%")
    print(f"  Avg ROI per game (bets only) : {overall_roi_bet:.3f}%")
    print(f"  Total profit         : ${total_profit:.2f}")
    print(f"  Average weekly profit: ${avg_profit:.2f}")
    print(f"  ROI on bankroll      : {roi_bankroll:.2f}%")
    print(f"  Pick win rate        : {pick_win_pct:.1f}%")
    print(f"  Avg return multiple  : {avg_mult:.2f}x")

    return {
        "label": label,
        "bet_rule_line": bet_rule_line,
        "total_profit": total_profit,
        "total_bets": total_bets,
        "overall_win_pct": overall_win_pct,
        "roi_bankroll_pct": roi_bankroll,
        "per_week": per_week,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline: bet the opening favorite every game (fixed vs Kelly sizing)."
    )
    p.add_argument(
        "--fixed-flat-pct",
        type=float,
        default=10.0,
        help="Fixed mode: bet size as %% of starting weekly bankroll (default 10).",
    )
    p.add_argument(
        "--dynamic-kelly",
        type=float,
        default=0.5,
        help="Dynamic mode: Kelly fraction weight on f* (default 0.5).",
    )
    p.add_argument(
        "--dynamic-cap",
        type=float,
        default=20.0,
        help="Dynamic mode: max %% of starting bankroll per bet (default 20).",
    )
    p.add_argument(
        "--dynamic-flat-fallback-pct",
        type=float,
        default=0.0,
        help=(
            "Passed as ``flat_bet_pct`` while Kelly is on: when no calibrated "
            "p_true is available, ``compute_bet_size`` uses this %% of starting "
            "bankroll (default 0 = skip those games; set e.g. 10 to mirror the "
            "optimizer’s flat fallback when the surface cell is empty)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
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

    fixed_params = _params_fixed(args.fixed_flat_pct)
    fixed_rule = f"flat {args.fixed_flat_pct:.1f}% of starting bankroll"
    fixed_summary = run_leave_one_week_out(
        "FIXED", all_weeks, contributions, fixed_params, fixed_rule
    )

    dyn_params = _params_dynamic(
        args.dynamic_kelly,
        args.dynamic_cap,
        args.dynamic_flat_fallback_pct,
    )
    dyn_rule = (
        f"{args.dynamic_kelly:.2f}-Kelly via LOO surface, capped at {args.dynamic_cap:.1f}%"
    )
    if args.dynamic_flat_fallback_pct > 0:
        dyn_rule += f"; flat fallback {args.dynamic_flat_fallback_pct:.1f}% if no p_true"
    dynamic_summary = run_leave_one_week_out(
        "DYNAMIC (Kelly)", all_weeks, contributions, dyn_params, dyn_rule
    )

    print(f"\n{'=' * 72}")
    print("HEAD-TO-HEAD COMPARISON (baseline variants)")
    print(f"{'=' * 72}")
    print(f"  {'Variant':<30}  {'Bets':>6}  {'Win%':>8}  {'Profit':>12}  {'ROI':>8}")
    print("  " + "-" * 64)
    print(
        f"  {'Fixed  (' + f'{args.fixed_flat_pct:.1f}% flat)':<30}  "
        f"{fixed_summary['total_bets']:>6}  "
        f"{fixed_summary['overall_win_pct']:>7.1f}%  "
        f"${fixed_summary['total_profit']:>10.2f}  "
        f"{fixed_summary['roi_bankroll_pct']:>7.2f}%"
    )
    print(
        f"  {'Dynamic (' + f'{args.dynamic_kelly:.2f}-Kelly)':<30}  "
        f"{dynamic_summary['total_bets']:>6}  "
        f"{dynamic_summary['overall_win_pct']:>7.1f}%  "
        f"${dynamic_summary['total_profit']:>10.2f}  "
        f"{dynamic_summary['roi_bankroll_pct']:>7.2f}%"
    )

    elapsed = time.perf_counter() - t0
    if elapsed >= 60:
        m, s = divmod(elapsed, 60)
        print(f"\n── Total wall time: {int(m)}m {s:.2f}s ({elapsed:.2f}s)")
    else:
        print(f"\n── Total wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
